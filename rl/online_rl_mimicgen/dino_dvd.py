"""Wiring that makes `--dvd-dynamics dino_wm` a drop-in f for DvdTerm."""
from __future__ import annotations

import numpy as np
import torch


if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

from .dino_wm import (
    DEFAULT_CKPT,
    FEATURE_DIM,
    HORIZON,
    IMAGE_SIZE,
    N_PATCHES,
    N_VIEWS,
    STATE_DIM,
    DinoWmAdapter,
    encode_context,
    images_to_context,
    load_dino_wm,
    pooled_dim,
)
from .pixel import CAMERAS, IMG as PIXEL_IMG, preprocess as pixel_preprocess


FRAME_FEAT_SHAPE = (N_VIEWS, N_PATCHES, FEATURE_DIM)


def assert_frame_size() -> None:
    """PIXEL_IMG is read from the environment at import time and sizes the replay's image column."""
    if PIXEL_IMG != IMAGE_SIZE:
        raise RuntimeError(
            f"--dvd-dynamics dino_wm needs PIXEL_IMG={IMAGE_SIZE} (got {PIXEL_IMG}); the world "
            "model was trained on unresized 224x224 frames and pixel.preprocess is only the "
            "identity at 224")
    if tuple(CAMERAS) != ("agentview", "eye_in_hand"):
        raise RuntimeError(f"dino-wm expects views ('agentview', 'eye_in_hand'), got {CAMERAS}")


class Bf16Adapter(DinoWmAdapter):
    """DinoWmAdapter with the predictor run under bf16 autocast, output cast back to fp32."""

    def __init__(self, model, action_dim: int = 7, bf16: bool = True, **kw):


        super().__init__(model, action_dim=action_dim, **kw)
        self.bf16 = bool(bf16)

    def autocast(self, device_type: str):
        return torch.autocast(device_type, dtype=torch.bfloat16,
                              enabled=self.bf16 and device_type == "cuda")

    def forward(self, raw_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with self.autocast(raw_state.device.type):
            return super().forward(raw_state, action).float()


def build(ckpt: str | None = None, device="cuda", bf16: bool = True,
          action_dim: int = 7, arch: str = "", flow_steps: int | None = None,
          flow_inverse_iters: int | None = None,
          finetune_dino_blocks: int = 0, dvd_checkpoint: bool = False,
          state_head: bool = False) -> Bf16Adapter:
    """`arch` is "depth,heads,mlp_dim"; empty = the published 6,6,2048 baseline."""
    assert_frame_size()
    kw = {}
    if arch:
        d, h, m = (int(x) for x in arch.replace(" ", "").split(","))
        kw = dict(depth=d, heads=h, mlp_dim=m, heads_explicit=True)
    model = load_dino_wm(ckpt or DEFAULT_CKPT, device=device, state_head=state_head, **kw)
    adapter = Bf16Adapter(model, action_dim=action_dim, bf16=bf16,
                          flow_steps=flow_steps, flow_inverse_iters=flow_inverse_iters,
                          finetune_dino_blocks=finetune_dino_blocks,
                          dvd_checkpoint=dvd_checkpoint)
    if finetune_dino_blocks:
        ntr = sum(p.numel() for p in adapter._trainable_dino_params()) / 1e6
        print(f'[dino-wm] ENCODER FINETUNING ON: last {finetune_dino_blocks} block(s), '
              f'{ntr:.1f}M encoder params trainable. This moves the actor observation, the '
              f'world-model input and V input together -- V was distilled on frozen '
              f'features.', flush=True)
    print(f"[dino-wm] {ckpt or DEFAULT_CKPT} on {device}  bf16={bf16}  "
          f"predictor={sum(p.numel() for p in adapter.parameters()) / 1e6:.1f}M trainable / "
          f"dino={sum(p.numel() for p in adapter.wm.dino.parameters()) / 1e6:.1f}M frozen",
          flush=True)
    return adapter


def adapter_device(adapter) -> torch.device:
    return next(adapter.parameters()).device


@torch.no_grad()
def encode_frames(adapter, frames) -> torch.Tensor:
    """(B, V, 224, 224, 3) uint8 -> (B, V, 196, 384) fp32 DINOv2 patch features."""
    device = adapter_device(adapter)
    x = images_to_context(frames[:, None]).to(device)
    with adapter.autocast(device.type):
        return encode_context(adapter.wm, x)[:, 0].float()


def set_context(adapter, prev_feat: torch.Tensor, cur_feat: torch.Tensor) -> None:
    """Point the adapter at [t-8, t]."""
    adapter.ctx_feats = torch.stack([prev_feat.float(), cur_feat.float()], dim=1)


def select_rows(total: int, rows: int, device, group: int = 1) -> torch.Tensor | None:
    """Indices of a uniform subsample, or None for 'take everything'."""
    if rows <= 0 or rows >= total:
        return None
    if group > 1:
        n_states = max(rows // group, 1)
        states = torch.randperm(total // group, device=device)[:n_states]
        return (states.unsqueeze(1) * group
                + torch.arange(group, device=device)).flatten()
    return torch.randperm(total, device=device)[:rows]


def context_for_rows(adapter, batch: dict, rows: torch.Tensor | None) -> None:
    """adapter.ctx_feats <- (stored t-8 features, a fresh encode of frame t) for `rows`."""
    prev = batch["prev_feat"]
    image = batch["image"]
    if rows is not None:
        prev, image = prev[rows], image[rows]
    set_context(adapter, prev, encode_frames(adapter, image))


def fit_batch(adapter, batch: dict, rows: int) -> dict:
    """A (possibly subsampled) copy of `batch` for DvdTerm.fit_step on the feature path."""
    index = select_rows(batch["state"].shape[0], rows, batch["state"].device)
    out = {key: (value if index is None else value[index]) for key, value in batch.items()}
    context_for_rows(adapter, out, None)
    out["target_feat"] = encode_frames(adapter, out["next_image"])
    return out


class FrameHistory:
    """The [t-8, t] context for ONE live env stream (the trainer's rollout, or one eval loop)."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.prev: torch.Tensor | None = None
        self.cur: torch.Tensor | None = None

    def reset(self) -> None:
        self.prev = self.cur = None

    def push(self, observation) -> torch.Tensor:
        """Encode this chunk's frames and roll the pair."""
        cur = encode_frames(self.adapter, pixel_preprocess(observation["pixels"])[None])
        self.prev = self.cur if self.cur is not None else cur
        self.cur = cur
        return self.prev

    def apply(self, n: int) -> None:
        """Point the adapter at this stream's context, tiled to `n` candidate rows."""
        if self.cur is None:
            raise RuntimeError("FrameHistory.apply before push: no frames have been observed")
        set_context(self.adapter, self.prev.expand(n, -1, -1, -1),
                    self.cur.expand(n, -1, -1, -1))

    def prev_feat(self) -> np.ndarray:
        """The t-8 features as the replay stores them:"""
        return self.prev[0].to(torch.float16).cpu().numpy()


def resolve_proprio_dim(value_fn, pool: str) -> int:
    """How many raw proprio dims V's input carries after the pooled features, 0 or 9."""
    width = int(value_fn.input_dim)
    pooled = pooled_dim(pool)
    if width == pooled:
        return 0
    if width == pooled + STATE_DIM:
        return STATE_DIM
    raise ValueError(
        f"--value-model has input_dim={width}, which is neither pooled_dim({pool!r})={pooled} nor "
        f"{pooled + STATE_DIM} (pooled + 9-dim raw proprio). Refit it with value/fit_v.sh "
        f"(value/fit_v_mqattn.py), or pass the --dino-pool that produced it.")
