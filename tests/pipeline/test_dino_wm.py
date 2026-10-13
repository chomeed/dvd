'''CPU self-check of online_rl_mimicgen/dino_wm.py on a production world model. No GPU, no dataset, B = 2.

Every test prints the number it measured; the asserts are there so a silent regression cannot pass.
Needs one production world model below $CKPT_ROOT/wm (or DINO_WM_CKPT=<file>) and facebook/dinov2-small in the
Hugging Face cache; prints SKIP otherwise.

    python -W ignore tests/pipeline/test_dino_wm.py
'''
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import _pipeline_util as U

U.use_repo()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from online_rl_mimicgen import dino_wm  # noqa: E402
from online_rl_mimicgen.dino_wm import (  # noqa: E402
    FEATURE_DIM, HISTORY, HORIZON, N_PATCHES, N_VIEWS, DinoWmAdapter,
    encode_context, images_to_context, load_dino_wm, pool_features, pooled_dim,
)

B = 2
_CACHE: dict = {}


def _ckpt() -> Path:
    if os.environ.get("DINO_WM_CKPT"):
        return Path(os.environ["DINO_WM_CKPT"])
    found = [U.wm_file(task) for task in U.TASKS if U.wm_file(task).exists()]
    if not found:
        U.skip("no production world model below $CKPT_ROOT/wm (and DINO_WM_CKPT is not set)")
    return found[0]


def _model():
    if "model" not in _CACHE:
        ckpt = _ckpt()
        t0 = time.time()
        _CACHE["model"] = load_dino_wm(ckpt, device="cpu")
        _CACHE["seconds"], _CACHE["ckpt"] = time.time() - t0, ckpt
    return _CACHE["model"]


def test_load():
    model = _model()
    ckpt = _CACHE["ckpt"]
    payload = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload, strict=False)
    non_dino = [k for k in missing if not k.startswith("dino.")]
    n_pred = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("dino."))
    print(f"[1] load_dino_wm({ckpt.parent.name}/{ckpt.name}) ok in {_CACHE['seconds']:.1f}s  missing={len(missing)} "
          f"(non-dino {len(non_dino)})  unexpected={len(unexpected)}  predictor {n_pred / 1e6:.2f}M  "
          f"state head {model.has_state_head}  action encoder {model.action_encoder}")
    assert not non_dino, non_dino
    assert not unexpected, unexpected
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training
    # the production recipe: proprio head + conv action encoder, 2-frame context, 8-step lead
    assert model.has_state_head and model.action_encoder == "conv"
    assert (HISTORY, HORIZON, N_VIEWS, N_PATCHES, FEATURE_DIM) == (2, 8, 2, 196, 384)
    assert not bool((model.state_delta_scale == 1.0).all()), "state_delta_scale was never set by the trainer"


def test_images_to_context():
    imgs = np.random.randint(0, 256, (B, HISTORY, N_VIEWS, 224, 224, 3), dtype=np.uint8)
    imgs[1, 0, 1, 7, 11, 2] = 200          # plant a known value at a known (h, w, c) and find it at (c, h, w)
    before = imgs.copy()
    ctx = images_to_context(imgs)
    planted = ctx[1, 0, 1, 2, 7, 11].item()
    print(f"[2] shape={tuple(ctx.shape)} dtype={ctx.dtype} range=[{ctx.min():.3f},{ctx.max():.3f}] "
          f"planted 200@(h7,w11,c2) -> {planted:.6f} (200/255={200 / 255:.6f})  "
          f"input unmutated={np.array_equal(imgs, before)}")
    assert ctx.shape == (B, HISTORY, N_VIEWS, 3, 224, 224)
    assert ctx.dtype == torch.float32
    assert 0.0 <= ctx.min().item() and ctx.max().item() <= 1.0
    assert abs(planted - 200 / 255) < 1e-6
    assert np.array_equal(imgs, before), "images_to_context mutated its input"
    assert torch.equal(ctx, images_to_context(torch.from_numpy(imgs)))          # torch input, same answer
    _CACHE["ctx"] = ctx


def test_grad():
    adapter = DinoWmAdapter(_model())
    # random stand-in features: this test is about the gradient path, not about the encoder
    adapter.ctx_feats = torch.randn(B, HISTORY, N_VIEWS, N_PATCHES, FEATURE_DIM)
    state = torch.randn(B, 37)
    action = torch.randn(B, HORIZON * 7, requires_grad=True)
    t0 = time.time()
    pred = adapter(state, action)
    g = torch.autograd.grad(pred.mean(), action)[0]
    dt = time.time() - t0
    print(f"[3] pred={tuple(pred.shape)} grad={tuple(g.view(B, HORIZON, 7).shape)} "
          f"|grad|={g.norm().item():.3e} nonzero={int((g != 0).sum())}/{g.numel()} fwd+bwd {dt:.1f}s")
    assert pred.shape == (B, N_VIEWS, N_PATCHES, FEATURE_DIM)
    assert g.shape == (B, HORIZON * 7)
    assert torch.isfinite(g).all() and g.abs().sum() > 0
    # the guards fire loudly rather than predicting from a stale context
    for bad, why in ((None, "ctx_feats=None"),
                     (torch.randn(1, HISTORY, N_VIEWS, N_PATCHES, FEATURE_DIM), "batch mismatch")):
        adapter.ctx_feats = bad
        try:
            adapter(state, action)
            raise AssertionError(f"no error for {why}")
        except RuntimeError as e:
            print(f"[3] guard {why}: {str(e)[:60]}...")


def test_freeze():
    adapter = DinoWmAdapter(_model())
    dino_n = sum(p.numel() for p in adapter.wm.dino.parameters())
    adapter.requires_grad_(True)                       # what DvdTerm.__init__ does
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    total_exposed = sum(p.numel() for p in adapter.parameters())
    dino_grad = [n for n, p in adapter.named_parameters() if "dino." in n and p.requires_grad]
    exposed_ids = {id(p) for p in adapter.parameters()}
    leaked = [n for n, p in adapter.named_parameters() if "dino." in n and id(p) in exposed_ids]
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-4)   # what DvdTerm.__init__ does next
    opt_n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print(f"[4] after requires_grad_(True): dino={dino_n / 1e6:.1f}M frozen, trainable={trainable / 1e6:.2f}M, "
          f"exposed by .parameters()={total_exposed / 1e6:.2f}M, Adam sees {opt_n / 1e6:.2f}M; dino params with "
          f"requires_grad={len(dino_grad)}, dino params leaked into .parameters()={len(leaked)}")
    assert not dino_grad, dino_grad
    assert not leaked, leaked
    assert opt_n == trainable == total_exposed
    assert trainable < dino_n, "trainable set is not smaller than the encoder -- it leaked"
    adapter.requires_grad_(False).train(True)          # must not re-enable the trunk either
    assert not any(p.requires_grad for p in adapter.wm.dino.parameters())
    assert not adapter.wm.dino.training, "dino must stay in eval mode"
    print(f"[4] after requires_grad_(False).train(True): dino trainable=0, dino.training={adapter.wm.dino.training}")


def test_pool():
    feats = torch.randn(B, N_VIEWS, N_PATCHES, FEATURE_DIM)
    m, mm = pool_features(feats, "mean"), pool_features(feats, "meanmax")
    print(f"[5] mean -> {tuple(m.shape)} (pooled_dim={pooled_dim('mean')})   "
          f"meanmax -> {tuple(mm.shape)} (pooled_dim={pooled_dim('meanmax')})")
    assert m.shape == (B, N_VIEWS * FEATURE_DIM) == (B, pooled_dim("mean"))
    assert mm.shape == (B, N_VIEWS * 2 * FEATURE_DIM) == (B, pooled_dim("meanmax"))
    assert torch.allclose(mm[:, :FEATURE_DIM], feats[:, 0].mean(1))          # layout [v0_mean | v0_max | v1_mean | v1_max]
    assert torch.allclose(mm[:, FEATURE_DIM:2 * FEATURE_DIM], feats[:, 0].amax(1))
    assert torch.allclose(m[:, FEATURE_DIM:], feats[:, 1].mean(1))
    try:
        pool_features(feats, "sum")
        raise AssertionError("unknown mode accepted")
    except ValueError as e:
        print(f"[5] guard: {e}")


def test_import_cost():
    code = ("import time,sys;t=time.time();import online_rl_mimicgen.dino_wm as m;"
            "print(time.time()-t, 'lerobot' in sys.modules, 'diffusers' in sys.modules, m.__file__)")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=str(U.RL), capture_output=True, text=True,
                         check=True).stdout.split()
    secs, lero, diff = float(out[0]), out[1] == "True", out[2] == "True"
    print(f"[6] import online_rl_mimicgen.dino_wm: {secs:.2f}s  lerobot={lero} diffusers={diff}")
    assert secs < 30.0, secs
    assert not (lero or diff), "dino_wm must stay importable without the policy stack (the offline pipeline relies on it)"
    assert U.RL.resolve() in Path(out[3]).resolve().parents


def test_end_to_end():
    """One real encode -> predict -> pool pass, the shapes the DVD loop sees."""
    model = _model()
    if "ctx" not in _CACHE:
        test_images_to_context()
    adapter = DinoWmAdapter(model)
    t0 = time.time()
    z = encode_context(model, _CACHE["ctx"][:B])
    dt = time.time() - t0
    adapter.ctx_feats = z
    state = torch.randn(B, 37)
    with torch.no_grad():
        pred = adapter(state, torch.rand(B, HORIZON * 7) * 2 - 1)
        pred2 = adapter(state, torch.rand(B, HORIZON * 7) * 2 - 1)
    v_in = pool_features(pred, "mean")
    d = (pool_features(pred2, "mean") - v_in).norm().item()          # the prediction must depend on the action
    print(f"[7] encode_context {tuple(z.shape)} in {dt:.1f}s -> pred {tuple(pred.shape)} "
          f"-> pooled {tuple(v_in.shape)}; ||pooled(a) - pooled(a')|| = {d:.4f}")
    assert z.shape == (B, HISTORY, N_VIEWS, N_PATCHES, FEATURE_DIM)
    assert v_in.shape == (B, pooled_dim("mean"))
    assert d > 0, "pooled prediction ignores the action"


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"module={dino_wm.__file__}")
    U.main(globals())
