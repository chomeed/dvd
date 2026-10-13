"""MULTI-QUERY ATTENTION POOLING -- AttnPool's raw math with K queries, per camera or jointly."""
from __future__ import annotations

import torch
import torch.nn as nn

FEATURE_DIM = 384
N_VIEWS = 2
N_PATCHES = 196
MQATTN_QUERIES = 1
MQATTN_INIT_SCALE = 0.02


def concat_dim(queries: int = MQATTN_QUERIES, per_view: bool = False,
               dim: int = FEATURE_DIM, n_views: int = N_VIEWS) -> int:
    """Width of the CONCATENATED query outputs, before any output projection."""
    k, d, v = int(queries), int(dim), int(n_views)
    return (v * k * d) if per_view else (k * d)


def pooled_dim(queries: int = MQATTN_QUERIES, per_view: bool = False,
               out_dim: int | None = None, dim: int = FEATURE_DIM,
               n_views: int = N_VIEWS) -> int:
    """Width V actually reads, BEFORE the 9 proprio dims:"""
    return int(out_dim) if out_dim else concat_dim(queries, per_view, dim, n_views)


class MultiQueryAttnPool(nn.Module):
    """K raw attention queries over the patch grid, per camera or jointly."""

    def __init__(self, dim: int = FEATURE_DIM, queries: int = MQATTN_QUERIES,
                 per_view: bool = False, out_dim: int | None = None,
                 n_views: int = N_VIEWS, query_norm: bool = False,
                 share_queries: bool = False, patches: int | None = None,
                 hidden: int | None = None):
        super().__init__()
        dim, queries, n_views = int(dim), int(queries), int(n_views)
        per_view, query_norm, share_queries = bool(per_view), bool(query_norm), bool(share_queries)
        if dim < 1 or queries < 1 or n_views < 1:
            raise ValueError(f"MultiQueryAttnPool needs dim/queries/n_views >= 1, got "
                             f"{dim}/{queries}/{n_views}")
        if out_dim is not None:
            out_dim = int(out_dim)
            if out_dim < 1:
                raise ValueError(f"MultiQueryAttnPool out_dim must be >= 1 or None, got {out_dim}")
        if share_queries and not per_view:
            raise ValueError("MultiQueryAttnPool: share_queries only means anything with "
                             "per_view=True (a joint softmax already sees both cameras)")
        if hidden is not None:
            hidden = int(hidden)
            if hidden < 1:
                raise ValueError(f"MultiQueryAttnPool hidden must be >= 1 or None, got {hidden}")
            if out_dim is None:
                raise ValueError("MultiQueryAttnPool: hidden needs an out_dim -- a hidden stage "
                                 "that projects back to the raw concatenation width is not a "
                                 "readout this module defines")
        if patches is not None:
            patches = int(patches)
            if patches < 1:
                raise ValueError(f"MultiQueryAttnPool patches must be >= 1 or None, got {patches}")
        self.dim, self.k, self.per_view, self.n_views = dim, queries, per_view, n_views
        self.query_norm, self.share_queries, self.patches = query_norm, share_queries, patches
        self.hidden = hidden
        self.concat_dim = concat_dim(queries, per_view, dim, n_views)
        self.out_dim = out_dim if out_dim else self.concat_dim
        self.projected = out_dim is not None


        qshape = ((1 if share_queries else n_views), queries, dim) if per_view else (queries, dim)
        self.q = nn.Parameter(torch.randn(*qshape) * MQATTN_INIT_SCALE)
        self.q_ln = nn.LayerNorm(dim) if query_norm else nn.Identity()


        self.out_proj = (nn.Identity() if not self.projected else
                         nn.Linear(self.concat_dim, self.out_dim) if hidden is None else
                         nn.Sequential(nn.Linear(self.concat_dim, hidden), nn.SiLU(),
                                       nn.Linear(hidden, self.out_dim)))
        self.register_buffer("meta", torch.tensor(
            [dim, queries, int(per_view), n_views, (out_dim or 0), int(query_norm),
             int(share_queries), (patches or 0)], dtype=torch.long))


        if hidden is not None:
            self.register_buffer("hidden_meta", torch.tensor([hidden], dtype=torch.long))


    @property
    def pooled_dim(self) -> int:
        return self.out_dim

    def weights(self, feats: torch.Tensor) -> torch.Tensor:
        """The attention weights themselves, for inspection:"""
        feats = self._check(feats)
        q = self.q_ln(self.q).to(feats.dtype)
        if not self.per_view:
            return torch.softmax((feats.flatten(1, 2) @ q.T) / self.dim ** 0.5, dim=1)
        if self.share_queries:
            q = q.expand(self.n_views, -1, -1)
        return torch.softmax(torch.einsum("bvpd,vkd->bvpk", feats, q) / self.dim ** 0.5, dim=2)

    def _check(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 4:
            raise ValueError(f"MultiQueryAttnPool expects (B,V,P,d), got {tuple(feats.shape)}")
        b, v, p, d = feats.shape
        if d != self.dim:
            raise ValueError(f"MultiQueryAttnPool was built for d={self.dim}, got {d}")
        if v != self.n_views:
            raise ValueError(f"MultiQueryAttnPool was built for {self.n_views} views, got {v}")
        if self.patches is not None and p != self.patches:
            raise ValueError(f"MultiQueryAttnPool was built for P={self.patches}, got P={p}")
        if feats.dtype != self.q.dtype:


            feats = feats.to(self.q.dtype)
        return feats

    def forward(self, feats: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """(B, V, P, d) -> (B, out_dim)."""
        feats = self._check(feats)
        q = self.q_ln(self.q).to(feats.dtype)
        if not self.per_view:


            t = feats.flatten(1, 2)
            w = torch.softmax((t @ q.T) / self.dim ** 0.5, dim=1)
            z = torch.einsum("bpd,bpk->bkd", t, w).flatten(1)
        else:


            if self.share_queries:
                q = q.expand(self.n_views, -1, -1)
            w = torch.softmax(torch.einsum("bvpd,vkd->bvpk", feats, q) / self.dim ** 0.5, dim=2)
            z = torch.einsum("bvpd,bvpk->bvkd", feats, w).flatten(1)
        return self.out_proj(z) if self.projected else z

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, queries={self.k}, per_view={self.per_view}, "
                f"n_views={self.n_views}, share_queries={self.share_queries}, "
                f"query_norm={self.query_norm}, concat_dim={self.concat_dim}, "
                f"out_dim={self.out_dim}, hidden={self.hidden}, projected={self.projected}")


    @staticmethod
    def spec_from_state_dict(state_dict: dict) -> dict:
        """The shape as recorded IN the state dict, so loading needs no flags."""
        meta = state_dict.get("meta")
        if meta is None:
            raise KeyError("state dict has no 'meta' buffer; it was not written by MultiQueryAttnPool")
        dim, queries, per_view, n_views, out_dim, query_norm, share, patches = (
            int(x) for x in torch.as_tensor(meta).tolist())
        spec = dict(dim=dim, queries=queries, per_view=bool(per_view), n_views=n_views,
                    out_dim=(out_dim or None), query_norm=bool(query_norm),
                    share_queries=bool(share), patches=(patches or None))
        h = state_dict.get("hidden_meta")
        if h is not None:


            spec["hidden"] = int(torch.as_tensor(h).flatten()[0]) or None
        return spec

    @classmethod
    def from_blob(cls, blob, device=None) -> "MultiQueryAttnPool":
        """Rebuild from what blob() wrote (or from a bare state dict)."""
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        try:
            spec = cls.spec_from_state_dict(sd)
        except KeyError:

            spec = dict(dim=int(blob.get("dim", FEATURE_DIM)),
                        queries=int(blob.get("queries", MQATTN_QUERIES)),
                        per_view=bool(blob.get("per_view", False)),
                        n_views=int(blob.get("n_views", N_VIEWS)),
                        out_dim=(int(blob["out_dim"]) if blob.get("projected") else None),
                        query_norm=bool(blob.get("query_norm", False)),
                        share_queries=bool(blob.get("share_queries", False)),
                        patches=(int(blob["patches"]) if blob.get("patches") else None),
                        hidden=(int(blob["hidden"]) if blob.get("hidden") else None))
        module = cls(**spec)
        module.load_state_dict(sd)
        return module if device is None else module.to(device)

    def blob(self, **extra) -> dict:
        """The dict the offline fit saves as mqattn_pool.pt."""
        out = {"state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
               "dim": self.dim, "queries": self.k, "per_view": self.per_view,
               "n_views": self.n_views, "share_queries": self.share_queries,
               "query_norm": self.query_norm, "patches": self.patches,
               "concat_dim": self.concat_dim, "out_dim": self.out_dim, "hidden": self.hidden,
               "projected": self.projected, "pooled_dim": self.out_dim,
               "params": int(sum(p.numel() for p in self.parameters()))}
        out.update(extra)
        return out


    @classmethod
    def from_attnpool(cls, attnpool, device=None) -> "MultiQueryAttnPool":
        """A K=1 joint module holding dino_wm.AttnPool's query -- the exact-equivalence witness."""
        q = attnpool.q if isinstance(attnpool, nn.Module) else torch.as_tensor(attnpool)
        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(f"expected AttnPool.q of shape (1, d), got {tuple(q.shape)}")
        module = cls(dim=int(q.shape[1]), queries=1, per_view=False, out_dim=None)
        with torch.no_grad():
            module.q.copy_(q.detach().to(module.q.dtype))
        return module if device is None else module.to(device)
