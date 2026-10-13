"""IMPALA-encoder actor and critic that see what the POLICY sees, and nothing more."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import os as _os


IMG = int(_os.environ.get("PIXEL_IMG", 84))
PROPRIO = 9


CAMERAS = tuple(c for c in _os.environ.get(
    "PIXEL_CAMERAS", "agentview,eye_in_hand").split(",") if c)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        y = self.c1(F.relu(x))
        y = self.c2(F.relu(y))
        return x + y


class ImpalaTower(nn.Module):
    """One camera -> `out_dim` features."""

    def __init__(self, in_channels: int = 3, out_dim: int = 128,
                 channels: tuple[int, ...] = (16, 32, 32)):
        super().__init__()
        layers, c_in = [], in_channels
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1),
                       nn.MaxPool2d(3, stride=2, padding=1),
                       _ResidualBlock(c), _ResidualBlock(c)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_channels, IMG, IMG)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, out_dim), nn.LayerNorm(out_dim))
        self.out_dim = out_dim

    def forward(self, x):
        return F.relu(self.head(F.relu(self.conv(x))))


class PixelEncoder(nn.Module):
    """(images uint8 [B,2,84,84,3], proprio [B,9]) -> features [B, 2*out_dim + 9]."""

    def __init__(self, out_dim: int = 128, proprio_dim: int = PROPRIO,
                 shift_pad: int = 4):
        super().__init__()
        self.towers = nn.ModuleList([ImpalaTower(3, out_dim) for _ in CAMERAS])


        self.shift_pad = shift_pad
        self.proprio_dim = proprio_dim
        self.out_dim = out_dim * len(CAMERAS) + proprio_dim

    def forward(self, images: torch.Tensor, proprio: torch.Tensor,
                augment: bool = False) -> torch.Tensor:
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)

        x = images.permute(0, 1, 4, 2, 3)
        if augment:
            x = random_shift(x.reshape(-1, *x.shape[2:]),
                             pad=self.shift_pad).reshape(x.shape)
        feats = [t(x[:, i]) for i, t in enumerate(self.towers)]
        return torch.cat(feats + [proprio], dim=-1)


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ random shift:"""
    n, c, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode="replicate")

    ox = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (torch.arange(h, device=x.device)[None] + oy[:, None])
    cols = (torch.arange(w, device=x.device)[None] + ox[:, None])
    idx_r = rows[:, None, :, None].expand(n, c, h, w + 2 * pad)
    x = torch.gather(x, 2, idx_r[..., : w + 2 * pad])
    idx_c = cols[:, None, None, :].expand(n, c, h, w)
    return torch.gather(x, 3, idx_c)


def preprocess(images: dict[str, np.ndarray]) -> np.ndarray:
    """Env observation `pixels` dict -> uint8 array [2, 84, 84, 3], nearest-neighbour downsample."""
    idx = np.linspace(0, 223, IMG).astype(np.int64)
    out = np.empty((len(CAMERAS), IMG, IMG, 3), dtype=np.uint8)
    for i, cam in enumerate(CAMERAS):
        out[i] = images[cam][np.ix_(idx, idx)]
    return out


class PixelCriticEnsemble(nn.Module):
    """CriticEnsemble's interface, but the first argument is a FEATURE vector from PixelEncoder."""

    def __init__(self, feature_dim: int, chunk_dim: int, ensemble_size: int,
                 hidden: tuple[int, ...] = (256, 256, 256), layernorm: bool = True):
        super().__init__()
        nets = []
        for _ in range(ensemble_size):
            layers, c = [], feature_dim + chunk_dim
            for h in hidden:
                layers += [nn.Linear(c, h)] + ([nn.LayerNorm(h)] if layernorm else []) + [nn.Mish()]
                c = h
            layers.append(nn.Linear(c, 1))
            nets.append(nn.Sequential(*layers))
        self.nets = nn.ModuleList(nets)

    def forward(self, feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((feature, action), dim=-1)
        return torch.stack([n(x) for n in self.nets], dim=0)

    def conservative(self, feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self(feature, action).min(dim=0).values


    def reduce(self, feature: torch.Tensor, action: torch.Tensor, mode: str) -> torch.Tensor:
        """Ensemble reduction chosen at the CALL SITE, for best-of-N ranking only."""
        values = self(feature, action)
        if mode.startswith("lcb"):


            lam = float(mode[3:]) if len(mode) > 3 else 2.0
            return values.mean(dim=0) - lam * values.std(dim=0)
        if mode == "max":
            return values.max(dim=0).values
        if mode == "mean":
            return values.mean(dim=0)
        if mode == "min":
            return values.min(dim=0).values
        raise ValueError(f"reduction must be min, max, mean or lcb[lam]; got {mode!r}")


class ReplayBuffer:


    FIELDS = (
        "state", "base", "action", "next_state", "next_base", "reward", "done", "discount",
        "mc_return", "data_source", "base_samples", "next_base_samples",
        "next_state_1step",
    )


    FIELD_FALLBACK = {"next_state_1step": "next_state"}


    META_DEFAULTS = {
        "is_critical": 0.0,
        "is_catastrophic": 0.0,
        "critical_type": 0.0,
        "steps_to_catastrophe": float("nan"),
        "trajectory_id": -1.0,
        "timestep": -1.0,
        "catastrophic_timestep": float("nan"),


        "behaviour_logp": float("nan"),


        "episode_len": 0.0,
    }

