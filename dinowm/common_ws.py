"""Shared data layer of the offline pipeline: the frame cache (build_cache.py), the DINO-WM trainer (train_wm_ph.py),
RoboMeter scoring (robometer/score_rollouts.py) and the offline V fit (value/fit_v_mqattn.py).

WS_TASK. The task is chosen by the environment variable WS_TASK, read ONCE at import: it names the cache files
($CACHE_ROOT/wm_frames/<task>_frames_u8.npy, <task>_meta.npz) and every output. It has NO default. The research code
silently meant threading_d0 when it was unset (and called that cache flow55k_*), so a square or hammer command without
it trained on threading data; here every entry point calls _need_task(), which refuses an unset or unknown value. Set
it per command (WS_TASK=square_d0 python ...), never export it in a shell that also launches RL runs. To reuse a cache
built by the research code for threading, link flow55k_{frames_u8.npy,meta.npz} as threading_d0_*.

THE DATA. D_off = the frozen prior's own rollouts, 100 successes + 100 failures per task (ROLLOUTS), decoded once to
uint8 (N, 2 cameras, 224, 224, 3). Train = episodes 0-79 of BOTH splits, held out = episodes 80-99 (TRAIN_EPISODES).

THE TWO-FRAME CONTEXT (windows()). One world-model sample is (prev, cur, nxt) = frames (t-8, t, t+8) of one episode
with the 8 actions t..t+7: context [t-8, t] -> target t+8, H = 8 = the RL action chunk. Only t = 0 and t >= 8 are
used: at t = 0 the context is frame 0 twice (what the RL code feeds at the first decision of an episode), and frames
1..7 are dropped because no real t-8 exists for them. The same rule gives the V fit its predicted rows.

Augmentation: the DrQ random shift (pad 4) is PAIRED, i.e. one offset per sample and camera shared by all its frames
(shift_fixed with explicit offsets), applied before the resize to the trunk's 196 px.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

assert os.environ.get("PIXEL_IMG") == "224", "source scripts/env.sh first (PIXEL_IMG=224)"

# the prior-rollout dataset of each task (directory names below $DATA_ROOT are <name>_success and <name>_failure)
ROLLOUTS = {
    "square_d0": "mimicgen_square_d0_224x224_mtdit_flow_40k",
    "square_d1": "mimicgen_square_d1_224x224_mtdit_flow_45k",
    "threading_d0": "mimicgen_threading_d0_224x224_mtdit_flow_55k",
    "hammer_cleanup_d1": "mimicgen_hammer_cleanup_d1_224x224_mtdit_flow_35k",
}
TASKS = tuple(ROLLOUTS)

WS_TASK = os.environ.get("WS_TASK", "")
_ASSETS = Path(os.environ.get("ASSET_ROOT", Path(__file__).resolve().parents[1] / "assets"))
DATA = Path(os.environ.get("DATA_ROOT", _ASSETS / "datasets"))
CKPT = Path(os.environ.get("CKPT_ROOT", _ASSETS / "checkpoints"))
CACHE = Path(os.environ.get("CACHE_ROOT", _ASSETS / "cache")) / "wm_frames"
WM_OUT = CKPT / "wm"
OUT = Path(os.environ.get("RUNS_ROOT", _ASSETS / "runs")) / "wm_curves"
FRAMES = CACHE / f"{WS_TASK}_frames_u8.npy"
META = CACHE / f"{WS_TASK}_meta.npz"
CAMS = ("agentview", "eye_in_hand")
H = 8
TRAIN_EPISODES = 80
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
SHAPE1 = (2, 224, 224, 3)
FRAME_BYTES = int(np.prod(SHAPE1))


SPECS = {
    "dinov2s14": dict(hf="facebook/dinov2-small", px=196, patch=14, dim=384, n_prefix=1,
                      pad=4, shift="pre", heads=6, mlp=2048),
}
DEPTH = 6


if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)


def _need_task(expect: str | None = None) -> str:
    """Refuse to run without an explicit, known WS_TASK (see the module docstring); `expect` is a --task argument that
    must name the same task. Returns the task."""
    if WS_TASK not in TASKS:
        raise SystemExit(f"WS_TASK={WS_TASK!r}: set WS_TASK to one of {', '.join(TASKS)} for this command. There is no "
                         f"default task")
    if expect is not None and expect != WS_TASK:
        raise SystemExit(f"WS_TASK={WS_TASK!r} but --task {expect!r}: the cache and every output are named by WS_TASK; "
                         f"make them agree")
    return WS_TASK


def wm_path(task: str, tag: str) -> Path:
    """Where train_wm_ph.py writes a world model and where every consumer looks for it. The directory name is
    load-bearing: train_online refuses a --dino-wm-ckpt whose path does not contain mimicgen_<task>_224x224_dinowm."""
    return WM_OUT / f"mimicgen_{task}_224x224_dinowm_{tag}" / f"wm_dinov2s14_{tag}.pt"


def shift_fixed(x: torch.Tensor, off: torch.Tensor, pad: int) -> torch.Tensor:
    """pixel.random_shift with the offsets given:"""
    n, c, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode="replicate")
    ox, oy = off[:, 0].to(x.device).long(), off[:, 1].to(x.device).long()
    rows = torch.arange(h, device=x.device)[None] + oy[:, None]
    cols = torch.arange(w, device=x.device)[None] + ox[:, None]
    idx_r = rows[:, None, :, None].expand(n, c, h, w + 2 * pad)
    x = torch.gather(x, 2, idx_r[..., : w + 2 * pad])
    idx_c = cols[:, None, None, :].expand(n, c, h, w)
    return torch.gather(x, 3, idx_c)


def frac_to_off(u: np.ndarray, pad: int) -> np.ndarray:
    """uniform fractions in [0,1) -> integer offsets in [0, 2 pad] (same relative shift for every pad)."""
    return np.minimum((np.asarray(u) * (2 * pad + 1)).astype(np.int64), 2 * pad)


class Backbone(torch.nn.Module):
    """Frozen HF DINO trunk."""

    def __init__(self, name: str, device, dtype=torch.float32):
        super().__init__()
        from transformers import AutoModel
        s = SPECS[name]
        self.name, self.spec = name, s
        self.model = AutoModel.from_pretrained(s["hf"], dtype=dtype).to(device).eval().requires_grad_(False)
        cfg = self.model.config
        n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        assert cfg.hidden_size == s["dim"] and cfg.patch_size == s["patch"], (cfg.hidden_size, cfg.patch_size)
        assert 1 + n_reg == s["n_prefix"], (name, n_reg)
        assert s["px"] % s["patch"] == 0
        self.px, self.pad, self.n_prefix, self.dim = s["px"], s["pad"], s["n_prefix"], s["dim"]
        self.side = s["px"] // s["patch"]
        self.n_patches = self.side ** 2
        self.register_buffer("mean", torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1))

    def prep(self, x01: torch.Tensor, off: torch.Tensor | None = None) -> torch.Tensor:
        x = x01
        if off is not None and self.spec["shift"] == "pre":
            x = shift_fixed(x, off, self.pad)
        x = F.interpolate(x, self.px, mode="bilinear", antialias=True, align_corners=False)
        if off is not None and self.spec["shift"] == "post":
            x = shift_fixed(x, off, self.pad)
        return (x - self.mean) / self.std

    def forward(self, x01: torch.Tensor, off: torch.Tensor | None = None) -> torch.Tensor:
        x = self.prep(x01, off)
        out = self.model(pixel_values=x.to(self.model.dtype)).last_hidden_state
        assert out.shape[1] == self.n_prefix + self.n_patches, (out.shape, self.n_prefix, self.n_patches)
        return out[:, self.n_prefix:].float()


def encode_u8(bb: Backbone, u8: torch.Tensor, off: torch.Tensor | None = None, bf16: bool = True,
              chunk: int = 384) -> torch.Tensor:
    """(B, ..., V, 224, 224, 3) uint8 on device -> (B, ..., V, P, d) fp32; off broadcastable (B, ..., V, 2)."""
    lead = u8.shape[:-3]
    x = u8.reshape(-1, 224, 224, 3)
    o = None if off is None else off.reshape(-1, 2)
    outs = []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    with torch.no_grad(), ctx:
        for i in range(0, x.shape[0], chunk):

            xi = x[i:i + chunk].permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            outs.append(bb(xi, None if o is None else o[i:i + chunk]))
    return torch.cat(outs).reshape(*lead, bb.n_patches, bb.dim)


def pool_mean(z: torch.Tensor) -> torch.Tensor:
    """(B, V, P, d) -> (B, V*d) = dino_wm.pool_features(z, 'mean') (views concatenated, never summed)."""
    return z.mean(dim=2).flatten(1)


def predictor_params(wm):
    return [p for n, p in wm.named_parameters() if not n.startswith("dino.")]


def predictor_state(wm) -> dict:
    return {k: v.detach().to("cpu", torch.float32).clone() for k, v in wm.state_dict().items()
            if not k.startswith("dino.")}


def load_meta() -> dict:
    _need_task()
    z = np.load(META, allow_pickle=False)
    return {k: z[k] for k in z.files}


def windows(M, which: str) -> dict:
    """Index arrays of the (prev, cur, nxt) = (t-8, t, t+8) windows of the train or held-out episodes: cur has frame
    0 or >= 8, prev is cur itself at frame 0, nxt stays inside the same episode and split."""
    n = len(M["ep"])
    i = np.arange(n - H)
    ok = (M["ep"][i + H] == M["ep"][i]) & (M["split"][i + H] == M["split"][i])
    fr = M["fr"][i]
    ok &= (fr >= H) | (fr == 0)
    ok &= (M["ep"][i] < TRAIN_EPISODES) if which == "train" else (M["ep"][i] >= TRAIN_EPISODES)
    cur = i[ok]
    prev = np.where(M["fr"][cur] >= H, cur - H, cur)
    assert np.all(M["ep"][prev] == M["ep"][cur]) and np.all(M["split"][prev] == M["split"][cur])
    return dict(prev=prev, cur=cur, nxt=cur + H, split=M["split"][cur], ep=M["ep"][cur], fr=M["fr"][cur],
                uid=np.char.add(np.char.add(M["split"][cur].astype(str), ":"), M["ep"][cur].astype(str)))


def chunk(M, cur) -> np.ndarray:
    """The 8 actions executed from frame cur (t..t+7), flattened to (N, 56)."""
    cur = np.asarray(cur)
    return M["action"][cur[:, None] + np.arange(H)[None]].reshape(len(cur), -1).astype(np.float32)


class FrameReader:
    """pread random access to the uint8 cache (never mmap, so RSS stays small)."""

    def __init__(self, path=None, threads: int = 8):
        if path is None:
            _need_task()
            path = FRAMES
        with open(path, "rb") as fh:
            ver = np.lib.format.read_magic(fh)
            rd = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}[ver]
            shape, fortran, dtype = rd(fh)
            self.offset = fh.tell()
        assert dtype == np.uint8 and not fortran and tuple(shape[1:]) == SHAPE1, (shape, dtype)
        self.n = int(shape[0])
        self.fd = os.open(path, os.O_RDONLY)
        self.pool = ThreadPoolExecutor(threads)

    def _one(self, out, j, r):
        got = os.preadv(self.fd, [memoryview(out[j].reshape(-1))], self.offset + int(r) * FRAME_BYTES)
        assert got == FRAME_BYTES, (r, got)

    def read(self, rows) -> np.ndarray:
        rows = np.asarray(rows, np.int64).reshape(-1)
        out = np.empty((len(rows), *SHAPE1), np.uint8)
        list(self.pool.map(lambda jr: self._one(out, *jr), enumerate(rows)))
        return out

    def close(self):
        self.pool.shutdown()
        os.close(self.fd)


def rss_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def lr_lambda(total: int, warmup: int = 500, floor: float = 0.1):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return f
