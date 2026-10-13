"""Load a from-scratch DINO-WM checkpoint on a non-shipped trunk (DINOv3-S/16, DINOv2-B/14, DINOv3-B/16)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from . import dino_dvd
from . import dino_wm as DW

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def is_sweep_checkpoint(ckpt: str | Path) -> bool:
    """True for a wm_sweep checkpoint (a dict carrying `trunk_hf_id` metadata beside `state_dict`)."""
    payload = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
    return isinstance(payload, dict) and "state_dict" in payload and "trunk_hf_id" in payload


class DinoDynamicsAny(DW.DinoDynamics):
    """DinoDynamics with the prefix-token count taken from the trunk (CLS + registers)."""

    def __init__(self, dino, *args, **kw):
        super().__init__(dino, *args, **kw)
        self.n_prefix = 1 + int(getattr(dino.config, "num_register_tokens", 0) or 0)

    def _encode_impl(self, imgs):
        x = F.interpolate(imgs, self.dino_size, mode="bilinear", antialias=True, align_corners=False)
        mean = torch.tensor(IMNET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMNET_STD, device=x.device).view(1, 3, 1, 1)
        out = self.dino(pixel_values=((x - mean) / std).to(self.dino.dtype)).last_hidden_state
        assert out.shape[1] == self.n_prefix + self.n_patches, (out.shape, self.n_prefix, self.n_patches)
        return out[:, self.n_prefix:].float()


def load_any(ckpt: str | Path, device="cuda", bf16: bool = True, dtype=torch.float32) -> dino_dvd.Bf16Adapter:
    from transformers import AutoModel
    payload = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
    assert isinstance(payload, dict) and "state_dict" in payload and "trunk_hf_id" in payload, \
        "not a wm_sweep checkpoint (use dino_dvd.build for the shipped DINOv2-S ones)"
    sd = payload["state_dict"]
    dino = AutoModel.from_pretrained(payload["trunk_hf_id"], dtype=dtype)
    d = int(dino.config.hidden_size)
    assert d == payload["feature_dim"] == sd["pos_emb"].shape[-1]
    model = DinoDynamicsAny(dino, None, DW.N_VIEWS, DW.ACTION_DIM, DW.STATE_DIM, history=payload["history"],
                            depth=payload["depth"], heads=payload["heads"], mlp_dim=payload["mlp_dim"],
                            dino_size=payload["dino_size"], image_size=DW.IMAGE_SIZE, flow=False, residual=False)
    assert model.n_prefix == payload["n_prefix_dropped"] and model.n_patches == sd["pos_emb"].shape[0]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("dino.")]
    if missing or unexpected:
        raise RuntimeError(f"{ckpt}: missing={missing} unexpected={unexpected}")
    model.ckpt_path = str(ckpt)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    DW.FEATURE_DIM, DW.N_PATCHES = d, model.n_patches
    DW.DINO_ID = payload["trunk_hf_id"]
    dino_dvd.FEATURE_DIM, dino_dvd.N_PATCHES = d, model.n_patches
    dino_dvd.FRAME_FEAT_SHAPE = (DW.N_VIEWS, model.n_patches, d)
    adapter = dino_dvd.Bf16Adapter(model, action_dim=7, bf16=bf16)
    print(f"[dino-wm-any] {ckpt}: trunk {payload['trunk_hf_id']} {payload['dino_size']}px grid "
          f"{payload['grid']}x{payload['grid']} d={d} prefix {model.n_prefix} heads {payload['heads']} "
          f"mlp {payload['mlp_dim']}; FRAME_FEAT_SHAPE={dino_dvd.FRAME_FEAT_SHAPE}", flush=True)
    return adapter
