"""DINO-WM world model for pixel DVD:"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

HORIZON = 8
FEATURE_DIM = 384
N_PATCHES = 196
N_VIEWS = 2
HISTORY = 2
ACTION_DIM = 7
STATE_DIM = 9
IMAGE_SIZE = 224

DINO_ID = "facebook/dinov2-small"


_TRUNK_BY_DIM = {384: "facebook/dinov2-small", 768: "facebook/dinov2-base"}


def _adopt_feature_dim(d: int) -> None:
    """Point the module at the trunk matching a checkpoint's width."""
    global FEATURE_DIM, DINO_ID
    if d not in _TRUNK_BY_DIM:
        raise RuntimeError(
            f"dino-wm checkpoint has width {d}; known trunks are {sorted(_TRUNK_BY_DIM)}. "
            f"Add it to _TRUNK_BY_DIM and make sure the matching facebook/dinov2-* is in HF_HOME "
            f"(or allow network access).")
    FEATURE_DIM, DINO_ID = d, _TRUNK_BY_DIM[d]
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


DEFAULT_CKPT = None


FLOW_SOLVER = "heun"


FLOW_CHECKPOINT = True
FLOW_STEPS = 8


FLOW_INVERSE_ITERS = 5


FLOW_ROUND_TRIP_STEPS = 20
FLOW_ROUND_TRIP = {

    (False, 2): (5.70, 5.02, 6.38, "PASS"),
    (False, 3): (6.70, 4.64, 9.04, "PASS"),
    (False, 5): (4.76, 2.70, 7.16, "PASS"),
    (True, 2): (15.19, 13.84, 16.63, "FAIL -- the inverse does not converge, do NOT run this"),
    (True, 3): (8.75, 7.54, 10.07, "STRADDLES the gate -- UNDER-RESOLVED, do NOT run this"),
    (True, 5): (4.54, 3.26, 6.04, "PASS"),
}


def flow_round_trip_note(residual: bool, steps: int, iters: int) -> str:
    """One line stating the MEASURED round trip for this (residual, steps, iters)."""
    if int(steps) != FLOW_ROUND_TRIP_STEPS:
        return (f"round trip NOT MEASURED at steps={steps} -- the table is at "
                f"steps={FLOW_ROUND_TRIP_STEPS} only, and the inverse shares the forward grid, so "
                f"re-run the round-trip gate before trusting this arm")
    hit = FLOW_ROUND_TRIP.get((bool(residual), int(iters)))
    if hit is None:
        measured = sorted(i for r, i in FLOW_ROUND_TRIP if r == bool(residual))
        return (f"round trip NOT MEASURED at iters={iters} for residual={bool(residual)} "
                f"(measured: {measured}); re-run the round-trip gate before trusting this arm")
    mean, lo, hi, verdict = hit
    return (f"round trip {mean:.2f}% [{lo:.2f},{hi:.2f}] of signal vs the 10% gate -- {verdict}")


def timestep_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding of a continuous flow time t in [0,1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float().reshape(-1, 1) * freqs[None] * 1000.0
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class ConvPredictor(nn.Module):
    """IMPALA-style residual conv stack over the DINOv2 patch grid:"""

    def __init__(self, dim: int, side: int, n_views: int, history: int, width: int = 384,
                 blocks: int = 4, delta: bool = False, delta_ln: bool = True):
        super().__init__()
        self.dim, self.side, self.n_views, self.history = dim, side, n_views, history
        self.width, self.blocks = width, blocks


        self.delta = bool(delta)
        self.delta_ln = bool(delta_ln)
        if self.delta and self.delta_ln:
            self.delta_scale = nn.Parameter(torch.tensor(0.1))
        if self.delta:
            self.register_buffer("is_delta", torch.ones(()))
        self.stem = nn.Conv2d((history + 1) * dim, width, 3, padding=1)
        self.film = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(blocks):
            self.film.append(nn.Sequential(nn.Linear(2 * dim + n_views * width, width), nn.GELU(),
                                           nn.Linear(width, 2 * width)))
            self.convs.append(nn.ModuleList([nn.Conv2d(width, width, 3, padding=1),
                                             nn.Conv2d(width, width, 3, padding=1)]))
            self.norms.append(nn.ModuleList([nn.GroupNorm(8, width), nn.GroupNorm(8, width)]))
        self.head_norm = nn.GroupNorm(8, width)
        self.head = nn.Conv2d(width, dim, 1)
        self.pool_proj = nn.Linear(width, dim)


        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.mul_(0.0)

    def forward(self, ctx: torch.Tensor, q: torch.Tensor, act: torch.Tensor, st: torch.Tensor):
        """ctx (B,R,V,P,d), q (B,V,P,d), act (B,d), st (B,d) -> (pred (B,V,P,d), pooled (B,d))."""
        b, r, v, p, d = ctx.shape
        s = self.side
        x = torch.cat([ctx.permute(0, 2, 1, 4, 3).reshape(b * v, r * d, s, s),
                       q.permute(0, 1, 3, 2).reshape(b * v, d, s, s)], dim=1)
        cond = torch.cat([act, st], dim=-1)
        h = self.stem(x)
        for i in range(self.blocks):
            pooled = h.mean((2, 3)).reshape(b, v * self.width)
            gb = self.film[i](torch.cat([cond, pooled], dim=-1))
            gamma, beta = gb.chunk(2, dim=-1)
            gamma = gamma.repeat_interleave(v, 0)[:, :, None, None]
            beta = beta.repeat_interleave(v, 0)[:, :, None, None]
            y = h * (1.0 + gamma) + beta
            y = self.convs[i][0](F.gelu(self.norms[i][0](y)))
            y = self.convs[i][1](F.gelu(self.norms[i][1](y)))
            h = h + y
        pooled = h.mean((2, 3)).reshape(b, v, self.width).mean(1)
        out = self.head(F.gelu(self.head_norm(h)))
        out = out.reshape(b, v, d, p).permute(0, 1, 3, 2)
        return out, self.pool_proj(pooled)

    def newest_frame(self, ctx: torch.Tensor) -> torch.Tensor:
        """(B,R,V,P,d) -> the frame the delta head adds to:"""
        return ctx[:, -1]


class DinoDynamics(nn.Module):
    """Frozen DINOv2 encoder + ViT predictor over [history x views] patch tokens."""

    def __init__(
        self,
        dino,
        vae,
        n_views,
        action_dim,
        state_dim,
        history=2,
        depth=6,
        heads=6,
        mlp_dim=2048,
        horizon=HORIZON,
        dino_size=196,
        image_size=224,
        use_action=True,
        use_state=True,
        flow=False,
        residual=False,
        state_head=False,
        action_encoder="linear",
        predictor="transformer",
        conv_width=384,
        conv_blocks=4,
        conv_delta=False,
        conv_delta_ln=True,
    ):
        super().__init__()
        self.dino = dino.eval().requires_grad_(False)
        self.vae = vae.eval().requires_grad_(False) if vae is not None else None
        d = dino.config.hidden_size
        self.dim, self.n_views, self.history = d, n_views, history


        self.use_action, self.use_state = use_action, use_state
        self.patch = dino.config.patch_size
        self.dino_size = dino_size
        self.side = dino_size // self.patch
        self.n_patches = self.side**2

        self.view_emb = nn.Parameter(torch.zeros(n_views, d))
        self.frame_emb = nn.Parameter(torch.zeros(history + 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(self.n_patches, d))


        self.query = nn.Parameter(torch.randn(self.n_patches, d) * 0.02)
        self.flow = bool(flow)


        self.residual = bool(residual)


        self.mse_residual = self.residual and not self.flow and predictor == "transformer"
        if self.mse_residual:
            self.delta_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
        if self.flow:
            self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))


            self.register_buffer("feat_mean", torch.zeros(d))
            self.register_buffer("feat_std", torch.ones(d))


        self.act_step_dim = action_dim
        self.action_encoder = str(action_encoder)
        if self.action_encoder == "conv":
            self.act_conv = nn.Sequential(
                nn.Conv1d(action_dim, 128, 3, padding=1), nn.GELU(),
                nn.Conv1d(128, 256, 3, padding=1), nn.GELU(),
                nn.Conv1d(256, 256, 3, padding=1), nn.GELU())
            self.act_tok = nn.Linear(256 * horizon, d)
        elif self.action_encoder == "linear":
            self.act_tok = nn.Linear(horizon * action_dim, d)
        else:
            raise ValueError(f"action_encoder {action_encoder!r} is not 'linear' or 'conv'")
        self.state_tok = nn.Linear(state_dim, d)


        self.predictor_arch = str(predictor)
        if self.predictor_arch == "conv":
            if flow:
                raise ValueError("predictor='conv' is implemented for the MSE head only (flow=False)")
            self.conv_pred = ConvPredictor(d, self.side, n_views, history,
                                           width=int(conv_width), blocks=int(conv_blocks),
                                           delta=bool(conv_delta), delta_ln=bool(conv_delta_ln))
            self.predictor = None
        elif self.predictor_arch == "transformer":
            layer = nn.TransformerEncoderLayer(
                d, heads, mlp_dim, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
            )
            self.predictor = nn.TransformerEncoder(layer, depth)
        else:
            raise ValueError(f"predictor {predictor!r} is not 'transformer' or 'conv'")
        self.norm = nn.LayerNorm(d)


        self.has_state_head = bool(state_head)
        if self.has_state_head:
            if self.flow:
                raise ValueError("state_head is defined for the MSE head only: under flow=True the "
                                 "predictor returns a velocity at each ODE step, not a prediction")
            self.state_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                            nn.Linear(d, state_dim))
            nn.init.zeros_(self.state_head[-1].weight)
            nn.init.zeros_(self.state_head[-1].bias)
            self.register_buffer("state_delta_scale", torch.ones(state_dim))

        self.last_state_delta = None
        self.last_state_pred = None


        if vae is not None:
            c = vae.config.latent_channels

            lat_side = image_size // 2 ** (len(vae.config.block_out_channels) - 1)
            if self.side * 2 != lat_side:
                raise ValueError(
                    f"patch grid {self.side}x2={self.side * 2} != VAE latent grid {lat_side} "
                    f"(image_size={image_size}, dino_size={dino_size}); pick dino_size = "
                    f"{lat_side // 2 * self.patch}"
                )
            self.to_latent = nn.Sequential(
                nn.Conv2d(d, 4 * c, 3, padding=1), nn.GELU(), nn.PixelShuffle(2),
                nn.Conv2d(c, c, 3, padding=1),
            )


    def set_feature_stats(self, mean, std):
        self.feat_mean.copy_(mean.to(self.feat_mean))
        self.feat_std.copy_(std.clamp_min(1e-3).to(self.feat_std))

    def norm_feat(self, z):
        return (z - self.feat_mean) / self.feat_std

    def denorm_feat(self, z):
        return z * self.feat_std + self.feat_mean


    def encode(self, imgs):
        """(N,3,H,W) in [0,1] -> (N, n_patches, d) DINOv2 patch tokens."""
        if getattr(self, "finetune_dino_blocks", 0) > 0:
            return self._encode_impl(imgs)
        with torch.no_grad():
            return self._encode_impl(imgs)

    def _encode_impl(self, imgs):
        x = F.interpolate(imgs, self.dino_size, mode="bilinear", antialias=True, align_corners=False)
        mean = torch.tensor(DINO_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(DINO_STD, device=x.device).view(1, 3, 1, 1)
        out = self.dino(pixel_values=((x - mean) / std).to(self.dino.dtype)).last_hidden_state
        return out[:, 1:].float()

    def encode_views(self, imgs):
        """(B,R,V,3,H,W) -> (B,R,V,P,d)."""
        b, r, v = imgs.shape[:3]
        z = self.encode(imgs.flatten(0, 2))
        return z.reshape(b, r, v, self.n_patches, self.dim)


    def forward(self, batch, flow_steps=FLOW_STEPS):
        """-> predicted target features (B,V,P,d)."""
        z = self.encode_views(batch["context"])
        if not self.flow:
            return self.predict_from_features(z, batch["action"], batch["state"])
        return self.sample(z, batch["action"], batch["state"], flow_steps)

    def predict_from_features(self, z, action, state, z_noisy=None, t=None):
        """Context features (B,R,V,P,d) -> predicted target features (B,V,P,d)."""
        b, r, v = z.shape[:3]
        z_raw = z
        z = z + self.view_emb[None, None, :, None] + self.pos_emb[None, None, None]
        z = z + self.frame_emb[:r][None, :, None, None]
        tokens = z.reshape(b, r * v * self.n_patches, self.dim)

        if self.flow:

            q = z_noisy + self.view_emb[None, :, None] + self.pos_emb[None, None]
            q = q + self.frame_emb[-1][None, None, None]
            q = q.reshape(b, v * self.n_patches, self.dim)
            q = q + self.t_mlp(timestep_embedding(t, self.dim).to(q.dtype))[:, None]
        else:
            q = self.query[None, None] + self.view_emb[None, :, None] + self.pos_emb[None, None]
            q = q + self.frame_emb[-1][None, None, None]
            q = q.expand(b, v, self.n_patches, self.dim).reshape(b, v * self.n_patches, self.dim)


        if getattr(self, "action_encoder", "linear") == "conv":
            h = self.act_conv(action.reshape(b, -1, self.act_step_dim).transpose(1, 2))
            act = self.act_tok(h.flatten(1)) * float(self.use_action)
        else:
            act = self.act_tok(action.flatten(1)) * float(self.use_action)
        st = self.state_tok(state) * float(self.use_state)
        if getattr(self, "predictor_arch", "transformer") == "conv":


            pred, pooled = self.conv_pred(z, q.reshape(b, v, self.n_patches, self.dim), act, st)
            if self.conv_pred.delta:


                pred = z_raw[:, -1] + (self.conv_pred.delta_scale * self.norm(pred)
                                       if self.conv_pred.delta_ln else pred)
            if getattr(self, "has_state_head", False):
                delta = self.state_head(pooled).float()
                self.last_state_delta = delta
                self.last_state_pred = state[:, :delta.shape[-1]].float() + delta * self.state_delta_scale
            if self.conv_pred.delta:
                return pred
            return self.norm(pred.reshape(b, v * self.n_patches, self.dim)).reshape(
                b, v, self.n_patches, self.dim)
        cond = torch.stack([act, st], 1)
        x = torch.cat([tokens, cond, q], 1)
        if getattr(self, "mse_checkpoint", False) and torch.is_grad_enabled():


            from torch.utils.checkpoint import checkpoint as _ckpt
            for _layer in self.predictor.layers:
                x = _ckpt(_layer, x, use_reentrant=False)
            out = x
        else:
            out = self.predictor(x)
        if getattr(self, "has_state_head", False):


            delta = self.state_head(out[:, -v * self.n_patches - 1]).float()
            self.last_state_delta = delta
            self.last_state_pred = state[:, :delta.shape[-1]].float() + delta * self.state_delta_scale
        q_out = self.norm(out[:, -v * self.n_patches :]).reshape(b, v, self.n_patches, self.dim)
        if getattr(self, "mse_residual", False):
            return z_raw[:, -1] + self.delta_head(q_out)
        return q_out


    def sample(self, z_ctx, action, state, steps=FLOW_STEPS, xi=None):
        """Euler-integrate the ODE from noise (t=1) to data (t=0)."""
        b, v = z_ctx.shape[0], z_ctx.shape[2]
        anchor = z_ctx[:, -1]
        z_ctx = self.norm_feat(z_ctx)
        x = torch.randn(b, v, self.n_patches, self.dim,
                        device=z_ctx.device, dtype=z_ctx.dtype) if xi is None else xi
        dt = 1.0 / steps


        if FLOW_SOLVER != "euler":
            x = self._rk_flow(z_ctx, action, state, x, steps, FLOW_SOLVER, reverse=False)
            x = self.denorm_feat(x)
            return anchor + x if self.residual else x
        use_ckpt = torch.is_grad_enabled()
        for i in range(steps):
            t = torch.full((b,), 1.0 - i * dt, device=x.device, dtype=torch.float32)
            if use_ckpt:
                def _step(xx, tt, zc=z_ctx, a=action, st=state):
                    return xx - self.predict_from_features(zc, a, st, xx, tt) * dt
                x = torch.utils.checkpoint.checkpoint(_step, x, t, use_reentrant=False)
            else:
                x = x - self.predict_from_features(z_ctx, action, state, x, t) * dt
        x = self.denorm_feat(x)
        return anchor + x if self.residual else x

    _MIN_INVERSE_STEPS = 8


    _TABLEAU = {
        "euler": (1, [0.0],      [[]],           [1.0]),
        "heun":  (2, [0.0, 1.0], [[], [1.0]],    [0.5, 0.5]),
        "rk4":   (4, [0.0, .5, .5, 1.0], [[], [.5], [0.0, .5], [0.0, 0.0, 1.0]],
                                                 [1/6, 2/6, 2/6, 1/6]),
    }

    def _rk_flow(self, zc, action, state, x, steps, solver="heun", reverse=False):
        """One explicit RK pass along the flow ODE."""
        ev, c, A, b = self._TABLEAU[solver]
        h = (-1.0 if reverse else 1.0) / steps


        ckpt = torch.is_grad_enabled() and FLOW_CHECKPOINT
        amp = torch.autocast("cuda", dtype=torch.bfloat16,
                     enabled=torch.is_grad_enabled() and x.is_cuda)

        def stage(x, tau):
            ks = []
            for j in range(ev):
                xj = x
                for l, coef in enumerate(A[j]):
                    if coef:
                        xj = xj - h * coef * ks[l]
                t = torch.full((x.shape[0],), float(1.0 - (tau + c[j] * h)),
                               device=x.device, dtype=torch.float32)
                ks.append(self.predict_from_features(zc, action, state, xj, t))
            return x - h * sum(w * k for w, k in zip(b, ks) if w)

        with amp:
            for i in range(steps):
                tau = (1.0 + i * h) if reverse else (i * h)


                x = torch.utils.checkpoint.checkpoint(stage, x, tau, use_reentrant=False) if ckpt \
                    else stage(x, tau)


        return x.float()

    def infer_noise(self, z_ctx, action, state, z_next, steps=FLOW_STEPS,
                    iters=FLOW_INVERSE_ITERS):
        """xi | (context, a_data, z_next) -- run the SAME map backwards."""


        if FLOW_SOLVER == "euler" and steps < self._MIN_INVERSE_STEPS:
            raise ValueError(
                f"infer_noise needs steps >= {self._MIN_INVERSE_STEPS} for the fixed-point inverse "
                f"to contract (got {steps}; dt={1.0/steps:.3f}). Below that more iters makes it "
                f"WORSE, and the failure is silent. Raise --flow-steps or use a different solver.")
        b = z_next.shape[0]
        x = z_next - z_ctx[:, -1] if self.residual else z_next
        x = self.norm_feat(x)
        z_ctx = self.norm_feat(z_ctx)
        if FLOW_SOLVER != "euler":


            return self._rk_flow(z_ctx, action, state, x, steps, FLOW_SOLVER, reverse=True)
        dt = 1.0 / steps
        for i in reversed(range(steps)):
            t = torch.full((b,), 1.0 - i * dt, device=x.device, dtype=torch.float32)
            x_next = x
            for _ in range(iters):
                x = x_next + self.predict_from_features(z_ctx, action, state, x, t) * dt
        return x

    def sample_anchored(self, z_ctx, a_theta, a_data, z_next, state,
                        steps=FLOW_STEPS, iters=FLOW_INVERSE_ITERS):
        """f^(context, a_theta, xi) reconstructed so a_theta == a_data returns z_next EXACTLY."""
        with torch.no_grad():
            xi = self.infer_noise(z_ctx, a_data, state, z_next, steps, iters)
            offset = z_next - self.sample(z_ctx, a_data, state, steps, xi)
        return self.sample(z_ctx, a_theta, state, steps, xi) + offset

    def loss(self, batch):
        """MSE to the target, or rectified-flow velocity matching when flow=True."""
        with torch.no_grad():
            tgt = self.encode_views(batch["future"][:, None])[:, 0]
        if not self.flow:
            return F.mse_loss(self(batch), tgt), tgt
        z_ctx = self.encode_views(batch["context"])
        return self.flow_loss(z_ctx, batch["action"], batch["state"], tgt), tgt

    def flow_loss(self, z_ctx, action, state, tgt):
        """Rectified-flow velocity matching, on already-encoded features."""
        if self.residual:
            tgt = tgt - z_ctx[:, -1]
        tgt = self.norm_feat(tgt)
        z_ctx = self.norm_feat(z_ctx)
        t = torch.sigmoid(torch.randn(len(tgt), device=tgt.device))
        noise = torch.randn_like(tgt)
        z_t = (1 - t).view(-1, 1, 1, 1) * tgt + t.view(-1, 1, 1, 1) * noise
        v_pred = self.predict_from_features(z_ctx, action, state, z_t, t)
        return F.mse_loss(v_pred, noise - tgt)


    def features_to_image(self, feats):
        """(B,V,P,d) -> (B,V,3,H,W) in [0,1], through the frozen SD3 VAE."""
        b, v = feats.shape[:2]
        x = feats.reshape(b * v, self.side, self.side, self.dim).permute(0, 3, 1, 2)
        lat = self.to_latent(x)
        z = lat.to(self.vae.dtype) / self.vae.config.scaling_factor + self.vae.config.shift_factor
        img = (self.vae.decode(z).sample / 2 + 0.5).clamp(0, 1).float()
        return img.reshape(b, v, *img.shape[1:])

    def decoder_loss(self, batch, tgt_feats):
        """Train the feature->latent head against the true frames."""
        with torch.no_grad():
            imgs = batch["future"].flatten(0, 1)
            z = self.vae.encode(imgs.to(self.vae.dtype) * 2 - 1).latent_dist.mode()
            z = ((z - self.vae.config.shift_factor) * self.vae.config.scaling_factor).float()
        b, v = batch["future"].shape[:2]
        x = tgt_feats.detach().reshape(b * v, self.side, self.side, self.dim).permute(0, 3, 1, 2)
        return F.mse_loss(self.to_latent(x), z)


def _conv_delta_flag(payload) -> bool:
    """A conv checkpoint's delta head without a scale:"""
    return bool(payload.get("conv_pred.is_delta", torch.zeros(())).item()) if "conv_pred.is_delta" in payload else False


def load_dino_wm(ckpt: str | Path = DEFAULT_CKPT, device="cpu",
                 dtype: torch.dtype = torch.float32,
                 depth: int = 6, heads: int = 6, mlp_dim: int = 2048,
                 heads_explicit: bool = False,
                 residual: bool | None = None,
                 state_head: bool = False,
                 action_dim: int | None = None,
                 state_dim: int | None = None) -> DinoDynamics:
    """Build DinoDynamics with vae=None and load the trained predictor."""
    from transformers import AutoModel

    if ckpt is None:
        raise ValueError("a DINO-WM checkpoint is required (--dino-wm-ckpt)")
    payload = torch.load(Path(ckpt), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]


    _pe = payload.get("pos_emb")
    if _pe is not None:
        _adopt_feature_dim(int(_pe.shape[-1]))
    dino = AutoModel.from_pretrained(DINO_ID, dtype=dtype)
    if _pe is not None and dino.config.hidden_size != int(_pe.shape[-1]):
        raise RuntimeError(
            f"trunk {DINO_ID} is {dino.config.hidden_size}-wide but the checkpoint is "
            f"{int(_pe.shape[-1])}-wide")


    d_depth = 1 + max((int(k.split(".")[2]) for k in payload
                       if k.startswith("predictor.layers.")), default=depth - 1)
    lin1 = payload.get("predictor.layers.0.linear1.weight")
    d_mlp = int(lin1.shape[0]) if lin1 is not None else mlp_dim
    if (d_depth, d_mlp) != (depth, mlp_dim):
        if heads_explicit:
            raise RuntimeError(
                f"dino-wm checkpoint {ckpt} is depth={d_depth} mlp_dim={d_mlp}, but "
                f"depth={depth} mlp_dim={mlp_dim} was requested")
        raise RuntimeError(
            f"dino-wm checkpoint {ckpt} is depth={d_depth} mlp_dim={d_mlp} (not the default "
            f"{depth}/{mlp_dim}). Pass --dino-wm-heads too: head count CANNOT be inferred from "
            f"the weights and a wrong value fails silently.")


    flow = any(k.startswith("t_mlp.") for k in payload)
    if flow:
        fm, fs = payload.get("feat_mean"), payload.get("feat_std")
        if fm is None or fs is None:
            raise RuntimeError(
                f"{ckpt} has t_mlp.* (a flow head) but no feat_mean/feat_std. The flow lives in "
                f"NORMALISED feature space; with the registered defaults (0, 1) norm_feat and "
                f"denorm_feat become the identity and the ODE transports a unit Gaussian onto a "
                f"cloud with per-dim means to +13.3 and stds to 8.07. That does not raise -- it "
                f"emits garbage of the right shape with a finite DVD gradient.")
        if float(fs.std()) <= 1e-3:
            raise RuntimeError(f"{ckpt} feat_std is constant ({float(fs.std()):.2e}); "
                               f"the feature stats did not travel in the checkpoint")


        looks_absolute = float(fm.abs().max()) > 5.0
        inferred = residual is None
        if inferred:
            residual = not looks_absolute
        elif bool(residual) == looks_absolute:
            raise RuntimeError(
                f"residual={residual} was requested for {ckpt}, but its feat_mean spans "
                f"[{float(fm.min()):+.4f}, {float(fm.max()):+.4f}] which is the "
                f"{'ABSOLUTE' if looks_absolute else 'RESIDUAL'} feature regime. One of the two is "
                f"wrong and the failure would be silent, so this refuses to guess.")
        print(f"[dino-wm] FLOW head: {ckpt}\n"
              f"          residual={residual} ({'inferred from feat_mean' if inferred else 'requested'}) "
              f"feat_mean [{float(fm.min()):+.4f}, {float(fm.max()):+.4f}] "
              f"feat_std [{float(fs.min()):.4f}, {float(fs.max()):.4f}]\n"
              f"          sampling is Euler, inverse is the implicit step on the SAME grid; the "
              f"RESOLVED (steps, iters) are PER ARM and are printed by DinoWmAdapter below "
              f"(module defaults {FLOW_STEPS}/{FLOW_INVERSE_ITERS})", flush=True)
    elif any(k.startswith("delta_head.") for k in payload):

        if residual is False:
            raise RuntimeError(f"residual=False requested for {ckpt}, but it has delta_head.* (an MSE feature-residual WM)")
        residual = True
        print(f"[dino-wm] MSE head: {ckpt}\n          residual=True (delta_head.* in the weights: x_t+8 = x_t + delta)", flush=True)
    elif payload.get("feat_mean") is not None:


        fm = payload["feat_mean"]
        looks_absolute = float(fm.abs().max()) > 5.0
        if residual is None:
            residual = not looks_absolute
        elif bool(residual) == looks_absolute:
            raise RuntimeError(
                f"residual={residual} requested for {ckpt}, but feat_mean spans "
                f"[{float(fm.min()):+.4f}, {float(fm.max()):+.4f}] -- the "
                f"{'ABSOLUTE' if looks_absolute else 'RESIDUAL'} regime. Refusing to guess.")
        print(f"[dino-wm] MSE head: {ckpt}\n"
              f"          residual={residual} (inferred from feat_mean) "
              f"feat_mean [{float(fm.min()):+.4f}, {float(fm.max()):+.4f}]", flush=True)


    head_in_ckpt = any(k.startswith("state_head.") for k in payload)

    action_encoder = "conv" if any(k.startswith("act_conv.") for k in payload) else "linear"
    if action_encoder != "linear":
        print(f"[dino-wm] action encoder: {action_encoder} (from the weights)", flush=True)


    _fe = payload.get("frame_emb")
    history = int(_fe.shape[0]) - 1 if _fe is not None else HISTORY
    if history != HISTORY:
        print(f"[dino-wm] context history: {history} frame(s) (from frame_emb {tuple(_fe.shape)})", flush=True)


    _conv_keys = [k for k in payload if k.startswith("conv_pred.")]
    predictor_arch = "conv" if _conv_keys else "transformer"
    conv_width = int(payload["conv_pred.stem.weight"].shape[0]) if _conv_keys else 384
    conv_blocks = (1 + max(int(k.split(".")[2]) for k in _conv_keys if k.startswith("conv_pred.convs."))
                   if _conv_keys else 4)


    conv_delta_ln = "conv_pred.delta_scale" in payload
    conv_delta = conv_delta_ln or bool(_conv_keys) and predictor_arch == "conv" and _conv_delta_flag(payload)
    if _conv_keys:
        print(f"[dino-wm] predictor: IMPALA conv backbone, width {conv_width}, {conv_blocks} blocks "
              f"(from the weights)", flush=True)


    model = DinoDynamics(dino, None, N_VIEWS, ACTION_DIM if action_dim is None else int(action_dim),
                         STATE_DIM if state_dim is None else int(state_dim), history=history,
                         depth=depth, heads=heads, mlp_dim=mlp_dim, image_size=IMAGE_SIZE,
                         flow=flow, residual=bool(residual),
                         state_head=bool(state_head) or head_in_ckpt,
                         action_encoder=action_encoder,
                         predictor=predictor_arch, conv_width=conv_width, conv_blocks=conv_blocks,
                         conv_delta=conv_delta, conv_delta_ln=conv_delta_ln)
    missing, unexpected = model.load_state_dict(payload, strict=False)

    missing = [k for k in missing if not k.startswith("dino.")]
    if model.has_state_head and not head_in_ckpt:
        missing = [k for k in missing if not (k.startswith("state_head.") or k == "state_delta_scale")]
        print(f"[dino-wm] proprio head: FRESH (zero-init, predicts s_t until trained; "
              f"state_delta_scale must be set from data)", flush=True)
    elif head_in_ckpt:
        print(f"[dino-wm] proprio head: loaded from {ckpt}; state_delta_scale "
              f"{[round(float(x), 5) for x in model.state_delta_scale]}", flush=True)
    unexpected = [k for k in unexpected if not k.startswith("to_latent.")]
    if not flow:


        unexpected = [k for k in unexpected if k not in ("feat_mean", "feat_std")]
    if missing or unexpected:
        raise RuntimeError(
            f"dino-wm checkpoint {ckpt} does not match this architecture: "
            f"missing={missing} unexpected={unexpected}"
        )


    if flow:
        assert float(model.feat_std.std()) > 1e-3, "feature stats missing from checkpoint"


    model.ckpt_path = str(ckpt)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model


def images_to_context(imgs_uint8) -> torch.Tensor:
    """(B,R,V,224,224,3) uint8 HWC  ->  (B,R,V,3,224,224) float32 in [0,1], NCHW."""
    x = torch.as_tensor(np.asarray(imgs_uint8) if not torch.is_tensor(imgs_uint8) else imgs_uint8)
    if x.dtype != torch.uint8:
        raise TypeError(f"images_to_context expects uint8 HWC frames, got {x.dtype}")
    if x.ndim != 6 or x.shape[-1] != 3 or x.shape[-3:-1] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected (B,R,V,{IMAGE_SIZE},{IMAGE_SIZE},3) uint8, got {tuple(x.shape)}")
    return x.permute(0, 1, 2, 5, 3, 4).contiguous().float().div_(255.0)


def encode_context(model: DinoDynamics, ctx_float: torch.Tensor) -> torch.Tensor:
    """(B,R,V,3,224,224) float in [0,1] -> (B,R,V,196,384) DINOv2 patch features."""
    with torch.no_grad():
        return model.encode_views(ctx_float)


def pool_features(feats: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """(B,V,196,384) patch features -> a flat vector for the value model."""
    if feats.ndim != 4:
        raise ValueError(f"pool_features expects (B,V,P,d), got {tuple(feats.shape)}")
    if mode == "flat":


        return feats.flatten(1)
    if mode == "mean":
        pooled = feats.mean(dim=2)
    elif mode == "meanmax":
        pooled = torch.cat([feats.mean(dim=2), feats.amax(dim=2)], dim=-1)
    elif mode.startswith("grid"):


        g = int(mode[4:])
        b = feats.shape[0]
        z = (feats.reshape(b, N_VIEWS, 14, 14, FEATURE_DIM)
                  .permute(0, 1, 4, 2, 3)
                  .reshape(b * N_VIEWS, FEATURE_DIM, 14, 14))
        pooled = (torch.nn.functional.adaptive_avg_pool2d(z, g)
                       .reshape(b, N_VIEWS, FEATURE_DIM * g * g))
    else:
        raise ValueError(f"unknown pooling mode {mode!r}; this function does 'mean', 'meanmax' "
                         f"and 'grid<N>' only. 'attnpool', 'sargmax', 'xattn' and 'mqattn' are "
                         f"MODULE pools (dino_wm.AttnPool / SoftArgmaxPool / CrossAttnPool, "
                         f"pool_mqattn.MultiQueryAttnPool): they are loaded beside the value "
                         f"model and CALLED INSTEAD OF this function, never through it")
    return pooled.flatten(1)


class AttnPool(nn.Module):
    """Learned attention pooling over the 392 patches of BOTH views."""

    def __init__(self, dim: int = FEATURE_DIM):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.dim = dim

    def forward(self, feats: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        if feats.ndim != 4:
            raise ValueError(f"AttnPool expects (B,V,P,d), got {tuple(feats.shape)}")
        t = feats.flatten(1, 2)
        w = torch.softmax((t @ self.q.T) / self.dim ** 0.5, dim=1)
        return torch.einsum("bpd,bpk->bkd", t, w).flatten(1)


XATTN_QUERIES = 8
XATTN_HEADS = 6
XATTN_OUT_DIM = N_VIEWS * FEATURE_DIM


class CrossAttnPool(nn.Module):
    """K learnable queries cross-attending over the 392 patches of BOTH views."""

    def __init__(self, dim: int = FEATURE_DIM, queries: int = XATTN_QUERIES,
                 heads: int = XATTN_HEADS, out_dim: int = XATTN_OUT_DIM,
                 query_norm: bool = True):
        super().__init__()
        dim, queries, heads, out_dim = int(dim), int(queries), int(heads), int(out_dim)
        if queries < 1 or heads < 1:
            raise ValueError(f"CrossAttnPool needs queries>=1 and heads>=1, got {queries}/{heads}")
        if dim % heads:
            raise ValueError(f"CrossAttnPool: dim {dim} is not divisible by heads {heads}")
        if out_dim % queries:
            raise ValueError(
                f"CrossAttnPool: out_dim {out_dim} is not divisible by queries {queries}; each "
                f"query owns out_dim//K dims and they are concatenated, so K must divide the "
                f"total width (768 takes K in 4, 8, 16, ...)")
        self.dim, self.k, self.heads, self.out_dim = dim, queries, heads, out_dim
        self.head_dim = dim // heads
        self.per_query = out_dim // queries
        self.query_norm = bool(query_norm)

        self.q = nn.Parameter(torch.randn(queries, dim) * 0.02)


        self.q_ln = nn.LayerNorm(dim) if self.query_norm else nn.Identity()
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_ln = nn.LayerNorm(dim)


        self.out_w = nn.Parameter(torch.randn(queries, dim, self.per_query) * dim ** -0.5)
        self.out_b = nn.Parameter(torch.zeros(queries, self.per_query))
        self.register_buffer("meta", torch.tensor(
            [dim, queries, heads, out_dim, int(self.query_norm)], dtype=torch.long))

    def forward(self, feats: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        if feats.ndim != 4:
            raise ValueError(f"CrossAttnPool expects (B,V,P,d), got {tuple(feats.shape)}")
        if feats.shape[-1] != self.dim:
            raise ValueError(f"CrossAttnPool was built for d={self.dim}, got {feats.shape[-1]}")
        b = feats.shape[0]
        t = feats.flatten(1, 2)
        k = self.k_proj(t).reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(t).reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        q = self.q_ln(self.q).reshape(self.k, self.heads, self.head_dim).permute(1, 0, 2)
        w = torch.softmax(torch.einsum("hkd,bhpd->bhkp", q.to(k.dtype), k)
                          * self.head_dim ** -0.5, dim=-1)
        z = torch.einsum("bhkp,bhpd->bhkd", w, v)
        z = self.out_ln(z.permute(0, 2, 1, 3).reshape(b, self.k, self.dim))
        return (torch.einsum("bkd,kdo->bko", z, self.out_w.to(z.dtype))
                + self.out_b.to(z.dtype)).flatten(1)


    @staticmethod
    def spec_from_state_dict(state_dict: dict) -> dict:
        """K / heads / dim / out_dim as recorded IN the state dict, so loading needs no flags."""
        meta = state_dict.get("meta")
        if meta is None:
            raise KeyError("state dict has no 'meta' buffer; it was not written by CrossAttnPool")
        dim, queries, heads, out_dim, qn = (int(x) for x in torch.as_tensor(meta).tolist())
        return dict(dim=dim, queries=queries, heads=heads, out_dim=out_dim, query_norm=bool(qn))

    @classmethod
    def from_blob(cls, blob, device=None) -> "CrossAttnPool":
        """Rebuild from what blob() wrote (or from a bare state dict)."""
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        try:
            spec = cls.spec_from_state_dict(sd)
        except KeyError:

            spec = dict(dim=int(blob.get("dim", FEATURE_DIM)),
                        queries=int(blob.get("queries", XATTN_QUERIES)),
                        heads=int(blob.get("heads", XATTN_HEADS)),
                        out_dim=int(blob.get("out_dim", XATTN_OUT_DIM)),
                        query_norm=bool(blob.get("query_norm", True)))
        module = cls(**spec)
        module.load_state_dict(sd)
        return module if device is None else module.to(device)

    def blob(self, **extra) -> dict:
        """The dict the offline fit saves as xattn_pool.pt."""
        out = {"state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
               "dim": self.dim, "queries": self.k, "heads": self.heads,
               "out_dim": self.out_dim, "query_norm": self.query_norm,
               "n_views": N_VIEWS, "pooled_dim": self.out_dim}
        out.update(extra)
        return out

SARGMAX_K = 16


class SoftArgmaxPool(nn.Module):
    """Mean pooling CONCATENATED WITH K soft-argmax keypoints per view."""

    def __init__(self, dim: int = FEATURE_DIM, k: int = SARGMAX_K, hid: int = 128,
                 n_views: int = N_VIEWS):
        super().__init__()
        self.dim, self.k, self.hid, self.n_views = dim, k, hid, n_views
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid))
        self.conv = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(hid, k, 1))
        self.temp = nn.Parameter(torch.tensor(1.0))
        self._grid: dict = {}

    def _coords(self, side: int, device, dtype):
        key = (side, str(device), str(dtype))
        if key not in self._grid:
            lin = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(lin, lin, indexing="ij")
            self._grid[key] = (gx.reshape(-1), gy.reshape(-1))
        return self._grid[key]

    def forward(self, feats: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        if mode not in (None, "sargmax"):
            return pool_features(feats, mode)
        if feats.ndim != 4:
            raise ValueError(f"SoftArgmaxPool expects (B,V,P,d), got {tuple(feats.shape)}")
        b, v, p, d = feats.shape
        side = int(round(p ** 0.5))
        if side * side != p:
            raise ValueError(f"SoftArgmaxPool needs a square patch grid, got P={p}")
        h = (self.proj(feats).permute(0, 1, 3, 2)
                 .reshape(b * v, self.hid, side, side))
        m = self.conv(h).reshape(b * v, self.k, -1)
        gx, gy = self._coords(side, feats.device, m.dtype)


        w = torch.softmax(m * self.temp.clamp(0.1, 50.0), dim=-1)
        kp = torch.stack([(w * gx).sum(-1), (w * gy).sum(-1), (w * m).sum(-1)], dim=-1)
        return torch.cat([feats.mean(dim=2).flatten(1),
                          kp.reshape(b, v * self.k * 3)], dim=-1)


MQATTN_OUT_DIM = N_VIEWS * FEATURE_DIM


def pooled_dim(mode: str = "mean") -> int:
    """Width of pool_features' output, for sizing the value model's input layer."""
    if mode == "attnpool":
        return FEATURE_DIM
    if mode == "xattn":


        return XATTN_OUT_DIM
    if mode == "mqattn":


        return MQATTN_OUT_DIM
    if mode == "sargmax":

        return N_VIEWS * FEATURE_DIM + N_VIEWS * SARGMAX_K * 3
    if mode == "flat":
        return N_VIEWS * FEATURE_DIM * 196
    if mode.startswith("grid"):
        g = int(mode[4:])
        return N_VIEWS * FEATURE_DIM * g * g
    return N_VIEWS * FEATURE_DIM * (2 if mode == "meanmax" else 1)


class DinoWmAdapter(nn.Module):
    """Duck-types the state-space Dynamics interface DvdTerm was written against, so DvdTerm can hold it."""

    def __init__(self, model: DinoDynamics, action_dim: int = ACTION_DIM,
                 horizon: int = HORIZON, flow_steps: int | None = None,
                 flow_inverse_iters: int | None = None, finetune_dino_blocks: int = 0,
                 dvd_checkpoint: bool = False):
        """flow_steps / flow_inverse_iters are the PER-ARM solver knob; None = the module default."""
        super().__init__()
        self.wm = model
        self.action_dim, self.horizon = action_dim, horizon


        self.finetune_dino_blocks = int(finetune_dino_blocks)
        self.wm.mse_checkpoint = bool(dvd_checkpoint)


        self.ctx_feats: torch.Tensor | None = None


        explicit = (flow_steps, flow_inverse_iters) != (None, None)
        self.flow_steps = FLOW_STEPS if flow_steps is None else int(flow_steps)
        self.flow_inverse_iters = (FLOW_INVERSE_ITERS if flow_inverse_iters is None
                                   else int(flow_inverse_iters))
        if self.flow_steps < 1 or self.flow_inverse_iters < 1:
            raise ValueError(
                f"flow_steps={self.flow_steps} flow_inverse_iters={self.flow_inverse_iters}: both "
                f"are loop counts and must be >= 1. iters=0 would skip the fixed point entirely "
                f"and return the EXPLICIT reverse step, which is not the inverse of anything.")


        self.flow = bool(getattr(self.wm, "flow", False))
        if explicit and not self.flow:


            raise RuntimeError(
                f"flow_steps={flow_steps} flow_inverse_iters={flow_inverse_iters} were set, but "
                f"{getattr(self.wm, 'ckpt_path', 'this checkpoint')} has NO flow head (no t_mlp.*), "
                f"so there is no ODE for them to configure. They would be inert and this arm would "
                f"silently duplicate its own MSE control. Point --dino-wm-ckpt at a flow "
                f"checkpoint, or drop --flow-steps/--flow-inverse-iters.")
        if self.flow:
            self.sample_anchored = self._flow_sample_anchored
            self.infer_noise = self._flow_infer_noise
            print(f"[dino-wm] adapter exposes sample_anchored + infer_noise -> DvdTerm will print "
                  f"anchored=True feature_space=True (no arm has ever printed both; if it prints "
                  f"anchored=False the run is integrating from FRESH NOISE and must be killed).\n"
                  f"[dino-wm] FLOW SOLVER RESOLVED "
                  f"({'per-arm' if explicit else 'module defaults, no --flow-* flag given'}):\n"
                  f"          ckpt={getattr(self.wm, 'ckpt_path', '<unknown>')}\n"
                  f"          residual={self.wm.residual} flow_steps={self.flow_steps} "
                  f"flow_inverse_iters={self.flow_inverse_iters}\n"
                  f"          {flow_round_trip_note(self.wm.residual, self.flow_steps, self.flow_inverse_iters)}\n"
                  f"          cost {self.flow_steps * (self.flow_inverse_iters + 2)} predictor "
                  f"forwards per anchored actor update, {self.flow_steps} of them retaining "
                  f"activations", flush=True)
        self._freeze_dino()


    def _trainable_dino_params(self):
        """The last `finetune_dino_blocks` encoder blocks, or nothing when the flag is off."""
        n = getattr(self, "finetune_dino_blocks", 0)
        if n <= 0:
            return []
        enc = getattr(getattr(self.wm.dino, "encoder", None), "layer", None)
        if enc is None:
            raise RuntimeError("finetune_dino_blocks set but dino.encoder.layer was not found; "
                               "the encoder would stay frozen while the run claimed otherwise")
        out = []
        for blk in list(enc)[-n:]:
            out += list(blk.parameters())
        return out

    def _freeze_dino(self):
        self.wm.dino.requires_grad_(False).eval()


        for prm in self._trainable_dino_params():
            prm.requires_grad_(True)

    @torch.no_grad()
    def feature_spread(self, feats):
        """Mean per-dim std across the batch."""
        return float(feats.reshape(-1, feats.shape[-1]).std(0).mean())

    def _dino_param_ids(self) -> set[int]:
        keep = {id(p) for p in self._trainable_dino_params()}
        return {id(p) for p in self.wm.dino.parameters() if id(p) not in keep}

    def parameters(self, recurse: bool = True):
        """Every parameter except the FROZEN part of the encoder."""
        frozen = self._dino_param_ids()
        for p in super().parameters(recurse=recurse):
            if id(p) not in frozen:
                yield p

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        self._freeze_dino()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_dino()
        return self


    def _context(self, raw_state: torch.Tensor) -> torch.Tensor:
        if self.ctx_feats is None:
            raise RuntimeError(
                "DinoWmAdapter.ctx_feats is None: set it to encode_context(model, frames) for this "
                "batch before calling the model"
            )
        if self.ctx_feats.shape[0] != raw_state.shape[0]:
            raise RuntimeError(
                f"ctx_feats batch {self.ctx_feats.shape[0]} != state batch {raw_state.shape[0]}; "
                "the context slot is stale"
            )


        return self.ctx_feats[:, -int(getattr(self.wm, "history", 2)):]

    def forward(self, raw_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(raw_state (B,>=9), action (B,56) or (B,8,7)) -> predicted features (B,V,196,384)."""
        z = self._context(raw_state)
        b = raw_state.shape[0]
        a = action.view(b, self.horizon, self.action_dim)
        if self.flow:
            return self.wm.sample(z, a, raw_state[:, :STATE_DIM], self.flow_steps)
        p = self.wm.predict_from_features(z, a, raw_state[:, :STATE_DIM])


        if self.wm.residual and getattr(self, "mse_anchor", False):
            return z[:, -1] + p
        return p

    def loss(self, raw_state: torch.Tensor, action: torch.Tensor,
             target_feats: torch.Tensor) -> torch.Tensor:
        """MSE against the TRUE next-frame features (B,V,196,384), in feature space."""
        b = raw_state.shape[0]
        a = action.view(b, self.horizon, self.action_dim)
        if self.flow:
            return self.wm.flow_loss(self._context(raw_state), a,
                                     raw_state[:, :STATE_DIM], target_feats)
        return F.mse_loss(self(raw_state, action), target_feats)

    @property
    def last_state_pred(self) -> torch.Tensor | None:
        """The proprio head's s_t + delta from the LAST forward, (B, 9) raw units; None without a head."""
        return self.wm.last_state_pred

    def state_residual_loss(self, raw_state: torch.Tensor, next_proprio: torch.Tensor) -> torch.Tensor:
        """MSE of the proprio head's NORMALISED RESIDUAL against (s_{t+8} - s_t)[:9] / state_delta_scale."""
        delta = self.wm.last_state_delta
        if delta is None or delta.shape[0] != raw_state.shape[0]:
            raise RuntimeError("state_residual_loss: no proprio-head output for this batch -- call "
                               "forward/loss on the same rows first")
        k = delta.shape[-1]
        target = (next_proprio[:, :k] - raw_state[:, :k]) / self.wm.state_delta_scale
        return F.mse_loss(delta, target)


    def _flow_sample_anchored(self, raw_state: torch.Tensor, a_theta: torch.Tensor,
                              a_data: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """DvdTerm._rollout's 4-positional-argument contract:"""
        z = self._context(raw_state)
        b = raw_state.shape[0]
        out = self.wm.sample_anchored(
            z, a_theta.view(b, self.horizon, self.action_dim),
            a_data.view(b, self.horizon, self.action_dim), z_next,
            raw_state[:, :STATE_DIM], self.flow_steps, self.flow_inverse_iters)
        if torch.is_grad_enabled() and a_theta.requires_grad and not out.requires_grad:


            raise RuntimeError(
                "sample_anchored returned a tensor with no grad_fn while a_theta requires grad: "
                "the flow sampler is running under no_grad, so the DVD term would contribute "
                "exactly zero actor gradient behind healthy-looking metrics.")
        return out

    def _flow_infer_noise(self, raw_state: torch.Tensor, action: torch.Tensor,
                          z_next: torch.Tensor) -> torch.Tensor:
        """xi | (context, action, z_next), in NORMALISED feature units."""
        b = raw_state.shape[0]
        return self.wm.infer_noise(
            self._context(raw_state), action.view(b, self.horizon, self.action_dim),
            raw_state[:, :STATE_DIM], z_next, self.flow_steps, self.flow_inverse_iters)
