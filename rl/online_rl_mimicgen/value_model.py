"""Frozen (or optionally fine-tuned) state-value model V(s) for the DICE-RL online loop."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

VARIANTS = ("v_td", "v_relabel_hinge")
SEEDS = (0, 1, 2)


class VNet(nn.Module):
    """Verbatim from the HF repo's modeling_v.py, so the state dicts load unchanged."""

    def __init__(self, state_dim: int = 66, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        current = state_dim
        for width in hidden:
            layers += [nn.Linear(current, width), nn.LayerNorm(width), nn.Mish()]
            current = width
        self.trunk = nn.Sequential(*layers)
        self.v_head = nn.Linear(current, 1)
        self.p_head = nn.Linear(current, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v_head(self.trunk(state))


class MCPooledMLP(nn.Module):
    """PooledV of the research tree's patch-V fitter (not shipped), verbatim keys ('net.*'), forward -> (B,1) like VNet."""
    def __init__(self, input_dim: int, hidden=(256, 256, 256)):
        super().__init__()
        layers, w = [], input_dim
        for h in hidden: layers += [nn.Linear(w, h), nn.ReLU()]; w = h
        self.net = nn.Sequential(*layers, nn.Linear(w, 1))
    def forward(self, state): return self.net(state)


class MCPatchTransformer(nn.Module):
    """PatchV of the same fitter, verbatim keys."""
    def __init__(self, d=384, layers=4, heads=6, cams=2, proprio=9):
        super().__init__()
        self.cams, self.d, self.proprio = cams, d, proprio
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.cam_emb = nn.Parameter(torch.zeros(cams, 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(1, 196, d))
        self.prop = nn.Linear(proprio, d)
        enc = nn.TransformerEncoderLayer(d, heads, 2048, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, state):
        pr = state[:, -self.proprio:]
        patches = state[:, :-self.proprio].reshape(len(state), self.cams, 196, self.d)
        toks = (patches + self.cam_emb.unsqueeze(0) + self.pos_emb.unsqueeze(1)).flatten(1, 2)
        toks = torch.cat([self.cls.expand(len(state), 1, -1), self.prop(pr)[:, None], toks], 1)
        if len(state) >= 256 and torch.is_grad_enabled():


            from torch.utils.checkpoint import checkpoint as _ckpt
            x = toks
            for _layer in self.tr.layers:
                x = _ckpt(_layer, x, use_reentrant=False)
            return self.head(x[:, 0])
        return self.head(self.tr(toks)[:, 0])


def _assert_scaler_matches(root: Path, scaler) -> None:
    """Fail loudly if V's normalisation is not the one the training loop already applied."""
    shipped = json.loads((root / "state_scaler.json").read_text())
    for key, training in (("q01", scaler.low), ("q99", scaler.high)):
        expected = np.asarray(shipped[key], dtype=np.float32)
        if expected.shape != training.shape or not np.allclose(expected, training, atol=0.0):
            raise ValueError(
                f"V's {key} does not match the training StateScaler's. The loop feeds V an "
                f"already-scaled state, so a mismatch means V is being evaluated in the wrong "
                f"input space. Refit V against {scaler.__class__.__name__}'s dataset stats."
            )


class ValueEnsemble(nn.Module):
    """Seed-averaged V(s), returning (batch, 1) to match `critics.conservative`."""

    def __init__(
        self,
        root: str | Path,
        variant: str,
        scaler,
        state_dim: int = 66,
        device: str | torch.device = "cuda",
        finetune: bool = False,
        aggregate: str = "mean",
        lcb_lambda: float = 2.0,
        scratch: bool = False,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown value variant {variant!r}, expected one of {VARIANTS}")
        root = Path(root)
        _assert_scaler_matches(root, scaler)
        config = json.loads((root / "config.json").read_text())
        if int(config["state_dim"]) != state_dim:
            raise ValueError(
                f"V was fit at state_dim={config['state_dim']} but this task is {state_dim}. "
                "The shipped checkpoints are coffee_d0 only."
            )


        input_dim = int(config.get("input_dim", config["state_dim"]))

        self.variant = variant
        self.input_dim = input_dim
        self.finetune = finetune
        self.aggregate = aggregate
        self.lcb_lambda = float(lcb_lambda)


        arch = config.get("arch", "vnet")


        _sd = config.get("seeds", SEEDS)
        seeds = range(int(_sd)) if isinstance(_sd, (int, float, str)) else tuple(_sd)
        nets = []
        for seed in seeds:
            if arch == "mc_pooled_mlp":
                net = MCPooledMLP(input_dim, tuple(config["hidden"]))
            elif arch == "mc_patch_transformer":
                net = MCPatchTransformer()
            else:
                net = VNet(input_dim, tuple(config["hidden"]))
            if not scratch:
                weights = root / "weights" / f"{variant}_seed{seed}.pt"
                net.load_state_dict(torch.load(weights, map_location="cpu"))
            nets.append(net)
        self.scratch = scratch
        self.nets = nn.ModuleList(nets)
        self.to(device)


        self.requires_grad_(finetune)
        self.train(finetune)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base = self._reduce_nets(state)
        d = getattr(self, "_delta", None)
        if d is None:
            return base
        if getattr(self, "_delta_cond", False):


            return base + d(torch.cat([state, base], dim=-1))
        return base + d(state)

    def _reduce_nets(self, state: torch.Tensor) -> torch.Tensor:
        values = torch.stack([net(state) for net in self.nets], dim=0)
        if self.aggregate == "lcb":


            return values.mean(dim=0) - self.lcb_lambda * values.std(dim=0)
        if self.aggregate == "max":


            return values.max(dim=0).values
        if self.aggregate == "min":


            return values.min(dim=0).values
        return values.mean(dim=0)

    @torch.no_grad()
    def frozen(self, state: torch.Tensor) -> torch.Tensor:
        """Value with gradients cut regardless of `finetune` -- for TD targets and shaping."""
        with torch.no_grad():
            return self(state)

    def enable_residual(self, device, hidden: int = 256, layers: int = 2,
                        freeze_base: bool = True, condition_on_base: bool = False) -> None:
        """V(s) = V_frozen(s) + dV_theta(s); only dV is optimised."""
        import torch.nn as nn


        in_dim = self.input_dim + (1 if condition_on_base else 0)
        dims = [in_dim] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.Mish()]
        head = nn.Linear(dims[-1], 1)
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        delta = nn.Sequential(*mods, head).to(device)
        object.__setattr__(self, "_delta_cond", bool(condition_on_base))


        if freeze_base:
            self.nets.requires_grad_(False)
        object.__setattr__(self, "_delta", delta)
        object.__setattr__(self, "_freeze_base", freeze_base)

    def residual_parameters(self):
        d = getattr(self, "_delta", None)
        if d is None:
            return []
        if getattr(self, "_freeze_base", True):
            return list(d.parameters())
        return list(d.parameters()) + list(self.nets.parameters())

    def enable_target(self, device) -> None:
        """Attach a frozen Polyak copy of this ensemble, used only as the TD bootstrap."""
        import copy as _copy


        target = _copy.deepcopy(self).to(device)
        target.requires_grad_(False)
        target.eval()
        object.__setattr__(self, "_target", target)

    def soft_update_target(self, tau: float) -> None:
        target = getattr(self, "_target", None)
        if target is None:
            return
        with torch.no_grad():
            for src, dst in zip(self.parameters(), target.parameters(), strict=True):
                dst.lerp_(src, tau)

    def _td_loss_fn(self):
        """--value-loss:"""
        return nn.functional.mse_loss if getattr(self, "value_loss", "smooth_l1") == "mse" else nn.functional.smooth_l1_loss

    def fit_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer,
        discount_key: str = "discount",
    ) -> float:
        """One online TD step on V, used only when --value-finetune is set."""
        if not self.finetune:
            return 0.0


        bootstrap = self._target if getattr(self, "_target", None) is not None else self
        with torch.no_grad():
            target = batch["reward"] + (1.0 - batch["done"]) * batch[discount_key] * bootstrap(
                batch["next_state"]
            )
        optimizer.zero_grad(set_to_none=True)
        if self.input_dim > 20000:


            n = len(target); chunk = 48; loss_val = 0.0
            for i in range(0, n, chunk):
                sl = slice(i, i + chunk)
                part = self._td_loss_fn()(self(batch["state"][sl]), target[sl])
                (part * (sl.indices(n)[1] - sl.indices(n)[0]) / n).backward()
                loss_val += float(part) * (min(i + chunk, n) - i) / n
            loss = torch.tensor(loss_val)
        else:
            loss = self._td_loss_fn()(self(batch["state"]), target)
            loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()
        return float(loss.detach())


    def mc_fit_step(self, batch: dict[str, torch.Tensor], optimizer, weight: float,
                    key: str = "mc_return") -> float:
        """One regression step of V onto the REALISED discounted return-to-go (`mc_return`)."""
        target = batch[key]
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        have = (target > -1e8).squeeze(-1)
        if not bool(have.any()):
            return float("nan")
        x, y = batch["state"][have], target[have]
        optimizer.zero_grad(set_to_none=True)
        nets = getattr(self, "nets", None)
        if nets is None or getattr(self, "_delta", None) is not None:
            pred = self(x)
        else:
            pred = torch.stack([net(x) for net in nets], dim=0)
            y = y.unsqueeze(0).expand_as(pred)
        loss = weight * nn.functional.smooth_l1_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()
        return float(loss.detach())


def _finite_abs_mean(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """mean|t| over the FINITE entries; 1.0 when there are none."""
    v = t.detach().abs().to(like.dtype)
    fin = torch.isfinite(v)
    n = fin.sum()
    s = torch.where(fin, v, torch.zeros_like(v)).sum()
    return torch.where(n > 0, s / n.clamp_min(1).to(like.dtype), torch.ones_like(s))


class DualValue(nn.Module):
    """Two value ensembles read as one:"""

    MIN_ROWS = 64
    MOMENTUM = 0.99
    COMBINES = ("min", "mean", "max")
    GRAD_DIAG_EVERY = 500

    def __init__(self, primary: nn.Module, td: nn.Module, combine: str = "min",
                 eps: float = 1e-6, max_pending: int = 512):
        super().__init__()
        if combine not in self.COMBINES:
            raise ValueError(f"combine must be one of {self.COMBINES}, got {combine!r}")
        if int(primary.input_dim) != int(td.input_dim):
            raise ValueError(
                f"primary V takes input_dim={primary.input_dim} but the td V takes "
                f"{td.input_dim}: both are called on the SAME vector, so they must be fitted on "
                "the same input space")
        self.primary = primary
        self.td = td
        self.combine = combine
        self.eps = float(eps)
        dev = next(primary.parameters()).device
        self.register_buffer("scale_primary", torch.ones((), device=dev))
        self.register_buffer("scale_td", torch.ones((), device=dev))
        self.register_buffer("scale_init", torch.zeros((), dtype=torch.bool, device=dev))
        from collections import deque


        self._pending = deque(maxlen=int(max_pending))
        self._armed = False
        self._consumed = False
        self._keep_graph = False
        self._graph = None
        self._gdiag_bucket = -1
        self._warned_missed = False
        self.scale_updates = 0
        self.scale_missed = 0


    @property
    def finetune(self) -> bool:
        return bool(getattr(self.primary, "finetune", False))

    @property
    def input_dim(self) -> int:
        return int(self.primary.input_dim)

    @property
    def variant(self) -> str:
        return (f"dual[{self.combine}]({getattr(self.primary, 'variant', '?')} "
                f"| td {getattr(self.td, 'variant', '?')})")

    def _refuse(self, name: str):
        raise RuntimeError(
            f"DualValue.{name} is refused: it is a per-ensemble operation. Use "
            f"dual.primary.{name} or dual.td.{name} explicitly (the RoboMeter distiller must be "
            "handed the primary; each member runs its own TD fit and Polyak target).")

    @property
    def nets(self):
        self._refuse("nets")

    @property
    def _target(self):
        self._refuse("_target")

    def fit_step(self, *a, **k):
        self._refuse("fit_step")

    def enable_target(self, *a, **k):
        self._refuse("enable_target")

    def soft_update_target(self, *a, **k):
        self._refuse("soft_update_target")

    def enable_residual(self, *a, **k):
        self._refuse("enable_residual")

    def residual_parameters(self, *a, **k):
        self._refuse("residual_parameters")


    def arm_scale_update(self, keep_graph: bool = False) -> None:
        """Open a one-shot window:"""
        self._armed = True
        self._consumed = False
        self._keep_graph = bool(keep_graph)
        self._graph = None

    def disarm_scale_update(self, expected: bool = False) -> bool:
        """Close the window; True if a qualifying call consumed it."""
        consumed = self._consumed
        self._armed = False
        self._consumed = False
        self._keep_graph = False
        if expected and not consumed:
            self.scale_missed += 1
            if not self._warned_missed:
                self._warned_missed = True
                print("\n" + "!" * 96 + "\n"
                      "[dual-v] the actor's DVD read had < %d rows, so the DualValue scales were "
                      "NOT updated\n  this update. They stay frozen (every read uses the last "
                      "stored scales) while the td V\n  keeps TD-training. Counted in "
                      "dual_scale_missed; this banner prints ONCE per run.\n" % self.MIN_ROWS
                      + "!" * 96 + "\n", file=sys.stderr, flush=True)
        return consumed

    @contextlib.contextmanager
    def scale_update(self, keep_graph: bool = False):
        """arm_scale_update() ..."""
        self.arm_scale_update(keep_graph)
        try:
            yield self
        finally:
            self.disarm_scale_update()

    def take_graph(self):
        """(a/s_a, b/s_b) from the last keep_graph window, graph attached; None otherwise."""
        g, self._graph = self._graph, None
        return g

    def claim_grad_diag(self, env_steps: int) -> bool:
        """True once per GRAD_DIAG_EVERY env steps (the first update of each 500-step record)."""
        bucket = int(env_steps) // self.GRAD_DIAG_EVERY
        if bucket == self._gdiag_bucket:
            return False
        self._gdiag_bucket = bucket
        return True


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = self.primary(state)
        b = self.td(state)
        rows = int(a.shape[0]) if a.dim() else 1
        update = self._armed and rows >= self.MIN_ROWS
        with torch.no_grad():
            if update:
                self._armed = False
                self._consumed = True
                self.scale_updates += 1
                ma = a.detach().abs().mean().to(self.scale_primary.dtype)
                mb = b.detach().abs().mean().to(self.scale_td.dtype)
                m = self.MOMENTUM
                init = self.scale_init
                ok = torch.isfinite(ma) & torch.isfinite(mb)
                new_a = torch.where(init, m * self.scale_primary + (1.0 - m) * ma, ma)
                new_b = torch.where(init, m * self.scale_td + (1.0 - m) * mb, mb)
                self.scale_primary.copy_(torch.where(ok, new_a, self.scale_primary))
                self.scale_td.copy_(torch.where(ok, new_b, self.scale_td))
                self.scale_init.copy_(init | ok)

            sa = torch.where(self.scale_init, self.scale_primary,
                             _finite_abs_mean(a, self.scale_primary)).clone().clamp_min(self.eps)
            sb = torch.where(self.scale_init, self.scale_td,
                             _finite_abs_mean(b, self.scale_td)).clone().clamp_min(self.eps)
        na = a / sa.to(a.dtype)
        nb = b / sb.to(b.dtype)
        if self.combine == "min":
            c = torch.minimum(na, nb)
        elif self.combine == "max":
            c = torch.maximum(na, nb)
        else:
            c = 0.5 * (na + nb)
        if update:
            self._record(a, b, na, nb)
            if self._keep_graph:
                self._graph = (na, nb)
                self._keep_graph = False
        return c * sa.to(c.dtype)

    @torch.no_grad()
    def frozen(self, state: torch.Tensor) -> torch.Tensor:
        return self(state)


    @torch.no_grad()
    def _record(self, a, b, na, nb) -> None:
        xa = na.detach().float().flatten()
        xb = nb.detach().float().flatten()
        ca, cb = xa - xa.mean(), xb - xb.mean()
        den = ca.norm() * cb.norm()
        nan = torch.full_like(den, float("nan"))
        corr = torch.where(den > 1e-12, (ca * cb).sum() / den.clamp_min(1e-12), nan)
        if self.combine == "mean":
            pick = lo = mid = hi = nan
        else:
            picked = (xb < xa).float() if self.combine == "min" else (xb > xa).float()
            pick = picked.mean()


            n = xa.numel()
            k1, k2 = n // 3, (2 * n) // 3
            ps = picked[torch.argsort(xa)]
            lo, mid, hi = ps[:k1].mean(), ps[k1:k2].mean(), ps[k2:].mean()
        self._pending.append(torch.stack([
            a.detach().float().mean(), b.detach().float().mean(), corr, pick,
            xa.std(), xb.std(), lo, mid, hi]))

    def pop_stats(self) -> dict[str, float]:
        """Mean over the recorded updates since the last pop, plus the scales and counters."""
        out: dict[str, float] = {"dual_scale_updates": float(self.scale_updates),
                                 "dual_scale_missed": float(self.scale_missed)}
        if bool(self.scale_init):
            out["dual_scale_primary"] = float(self.scale_primary)
            out["dual_scale_td"] = float(self.scale_td)
        if not self._pending:
            return out
        with torch.no_grad():
            arr = torch.stack(list(self._pending)).cpu().numpy()
        self._pending.clear()

        def _m(col: int) -> float:
            v = arr[:, col]
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else float("nan")

        out.update({"dual_v_primary_mean": _m(0), "dual_v_td_mean": _m(1),
                    "dual_corr": _m(2),
                    "dual_std_primary": _m(4), "dual_std_td": _m(5),
                    "dual_td_pick_frac": _m(3) if self.combine != "mean" else float("nan")})
        if self.combine != "mean":
            out.update({"dual_td_pick_frac_lo": _m(6), "dual_td_pick_frac_mid": _m(7),
                        "dual_td_pick_frac_hi": _m(8)})
        return out

    def scale_state_dict(self) -> dict[str, torch.Tensor]:
        return {"scale_primary": self.scale_primary.detach().clone(),
                "scale_td": self.scale_td.detach().clone(),
                "scale_init": self.scale_init.detach().clone(),

                "scale_updates": torch.tensor(self.scale_updates),
                "scale_missed": torch.tensor(self.scale_missed)}

    def load_scale_state(self, sd: dict) -> None:
        with torch.no_grad():
            for key in ("scale_primary", "scale_td", "scale_init"):
                getattr(self, key).copy_(torch.as_tensor(sd[key]))
        self.scale_updates = int(sd.get("scale_updates", self.scale_updates))
        self.scale_missed = int(sd.get("scale_missed", self.scale_missed))


class DuelingCritic:
    """Q(s, a) = V_frozen(s) + A_theta(s, a) -- only the advantage is learned online."""

    def __init__(self, critics, value_fn: "ValueEnsemble"):
        self.inner = critics
        self.value_fn = value_fn

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:


        return self.inner(state, action) + self.value_fn.frozen(state)

    def conservative(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        return self.inner.conservative(state, action) + self.value_fn.frozen(state)

    @property
    def q(self):
        return self.inner.q

    def parameters(self):
        return self.inner.parameters()

    def state_dict(self):
        return self.inner.state_dict()


class RunningStats:
    """Streaming mean/sd/min/max for the distribution-shift check."""

    def __init__(self) -> None:
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.low = float("inf")
        self.high = float("-inf")

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().flatten().float()
        for value in values.tolist():
            self.count += 1
            delta = value - self._mean
            self._mean += delta / self.count
            self._m2 += delta * (value - self._mean)
        if values.numel():
            self.low = min(self.low, float(values.min()))
            self.high = max(self.high, float(values.max()))

    def summary(self, prefix: str) -> dict[str, float]:
        if not self.count:
            return {}
        variance = self._m2 / max(self.count - 1, 1)
        return {
            f"{prefix}_mean": self._mean,
            f"{prefix}_sd": float(np.sqrt(variance)),
            f"{prefix}_min": self.low,
            f"{prefix}_max": self.high,
            f"{prefix}_count": float(self.count),
        }
