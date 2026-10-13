"""CPU unit test of online_rl_mimicgen.pool_mqattn.MultiQueryAttnPool, the "pv4" readout of V.

Checks what a readout that the trainer loads has to get right:
  * EXACT REDUCTION to dino_wm.AttnPool at queries=1, per_view=False, out_dim=None, with AttnPool's
    own query copied in -- reported as a max abs difference, and required to be 0;
  * widths (per-view K=1 and joint K=2 are both 768 = the mean-pool width; pv4 = per-view K=4 projected to 768),
    and the optional flat output projection;
  * the per-view softmax really is restricted to one camera, and the concat layout is [v0_q0 .. v0_qK | v1_q0 ..];
  * convexity: every query's output is a convex combination of RAW patch vectors (weights sum to 1
    and are non-negative), which is what "raw, no key/value projections" means;
  * the shape guards, fp16 input against fp32 parameters, and a blob() -> from_blob() ROUND TRIP
    that must reproduce the outputs bit for bit.
Run: python tests/pipeline/test_pool_mqattn.py
"""
import _pipeline_util as U

U.use_repo()

import torch  # noqa: E402

from online_rl_mimicgen.dino_wm import AttnPool  # noqa: E402
from online_rl_mimicgen.pool_mqattn import MultiQueryAttnPool, concat_dim, pooled_dim  # noqa: E402

B, V, P, D = 5, 2, 196, 384


def _x():
    torch.manual_seed(0)
    return torch.randn(B, V, P, D)


def test_exact_reduction_to_attnpool():
    x = _x()
    ref = AttnPool(dim=D).eval()
    mq = MultiQueryAttnPool.from_attnpool(ref).eval()
    assert mq.q.shape == ref.q.shape == (1, D)
    with torch.no_grad():
        a, b = ref(x, None), mq(x, None)
    d_fp32 = float((a - b).abs().max())
    assert a.shape == b.shape == (B, D), (a.shape, b.shape)
    assert d_fp32 == 0.0, f"K=1 joint is NOT AttnPool: max |d| = {d_fp32}"
    with torch.no_grad():          # with the mode argument AttnPool ignores, and on a different batch shape
        x9 = torch.randn(9, V, P, D)
        assert float((ref(x9, "attnpool") - mq(x9, "attnpool")).abs().max()) == 0.0
        # fp16 patch grids (what the fit caches) -- AttnPool has no upcast, so compare after .float()
        d_fp16 = float((ref(x.half().float(), None) - mq(x.half(), None)).abs().max())
    print(f"reduction to AttnPool: max abs diff fp32 {d_fp32:.3g}, fp16-input {d_fp16:.3g}, out {tuple(b.shape)}, "
          f"params {mq.q.numel()}")


def test_widths():
    x = _x()
    assert pooled_dim(1, False) == 384 and concat_dim(1, False) == 384            # AttnPool
    assert pooled_dim(2, False) == 768 and pooled_dim(1, True) == 768             # mean-pool width
    assert pooled_dim(4, True) == 3072 and pooled_dim(4, True, 768) == 768        # pv4: projected
    for queries, per_view, out_dim, want in [(1, False, None, 384), (2, False, None, 768), (1, True, None, 768),
                                             (4, True, None, 3072), (4, True, 768, 768), (8, False, 256, 256),
                                             (2, False, 256, 256)]:
        pool = MultiQueryAttnPool(dim=D, queries=queries, per_view=per_view, out_dim=out_dim).eval()
        y = pool(x, None)
        assert y.shape == (B, want), (queries, per_view, out_dim, y.shape)
        assert pool.pooled_dim == pool.out_dim == want
        assert pool.concat_dim == (V * queries * D if per_view else queries * D)
        nq = (V if per_view else 1) * queries * D
        npar = nq + ((pool.concat_dim * want + want) if out_dim else 0)
        assert sum(p.numel() for p in pool.parameters()) == npar, (queries, per_view, out_dim)
        assert pool(x).allclose(y) and pool(x, "mean").allclose(y), "mode must be accepted and IGNORED"
        print(f"ok queries={queries} per_view={per_view} out_dim={out_dim}: out {tuple(y.shape)} "
              f"concat={pool.concat_dim} params={npar:,}")


def test_scopes():
    x = _x()
    pv = MultiQueryAttnPool(dim=D, queries=3, per_view=True).eval()
    w = pv.weights(x)
    assert w.shape == (B, V, P, 3)
    assert torch.allclose(w.sum(2), torch.ones(B, V, 3), atol=1e-6) and bool((w >= 0).all()), \
        "per-view weights are not a convex combination over that camera's patches"
    x2 = x.clone(); x2[:, 1] = torch.randn(B, P, D)
    y, y2 = pv(x, None), pv(x2, None)
    half = 3 * D
    assert torch.equal(y[:, :half], y2[:, :half]), "view 1 changed view 0's output -> softmax leaked"
    assert not torch.allclose(y[:, half:], y2[:, half:])
    same = torch.cat([x[:, :1], x[:, :1]], 1)          # separate queries per camera really are separate
    assert not torch.allclose(pv(same, None)[:, :half], pv(same, None)[:, half:])
    sh = MultiQueryAttnPool(dim=D, queries=3, per_view=True, share_queries=True).eval()
    zs = sh(same, None)
    assert torch.allclose(zs[:, :half], zs[:, half:]), "shared queries disagree with themselves"
    assert sh.q.shape == (1, 3, D) and pv.q.shape == (V, 3, D)
    jt = MultiQueryAttnPool(dim=D, queries=2, per_view=False).eval()      # JOINT: one softmax over all V*P patches
    wj = jt.weights(x)
    assert wj.shape == (B, V * P, 2)
    assert torch.allclose(wj.sum(1), torch.ones(B, 2), atol=1e-6) and bool((wj >= 0).all())
    with torch.no_grad():          # the joint output IS the weighted mean of the RAW patch vectors
        manual = torch.einsum("bpd,bpk->bkd", x.flatten(1, 2), wj).flatten(1)
        assert float((manual - jt(x, None)).abs().max()) < 1e-6
    print(f"scopes ok: per-view weights (B,V,P,K)={tuple(w.shape)} sum-to-1 per camera; joint weights "
          f"{tuple(wj.shape)} sum-to-1 over {V * P} patches; output = raw weighted mean")


def test_guards_and_dtype():
    x = _x()
    g = MultiQueryAttnPool(dim=D, queries=2, per_view=True, patches=P).eval()
    for bad, msg in ((torch.randn(B, V, 100, D), "P="), (torch.randn(B, V, P, 768), "d="),
                     (torch.randn(B, 3, P, D), "views"), (torch.randn(B, P, D), "(B,V,P,d)")):
        try:
            g(bad); raise AssertionError(f"no guard for {msg}")
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
    for kw, msg in ((dict(queries=0), "queries"), (dict(out_dim=0), "out_dim"),
                    (dict(per_view=False, share_queries=True), "share_queries")):
        try:
            MultiQueryAttnPool(dim=D, **kw); raise AssertionError(f"no guard for {kw}")
        except ValueError as e:
            assert msg in str(e), (kw, str(e))
    assert g(x.half(), None).shape == (B, 2 * V * D)           # fp16 cache vs fp32 parameters


def test_round_trip():
    x = _x()
    for spec in (dict(queries=1, per_view=False), dict(queries=2, per_view=False), dict(queries=1, per_view=True),
                 dict(queries=4, per_view=True, out_dim=768),
                 dict(queries=4, per_view=True, share_queries=True, query_norm=True),
                 dict(queries=8, per_view=False, out_dim=256, patches=P)):
        pool = MultiQueryAttnPool(dim=D, **spec).eval()
        y = pool(x, None)
        blob = pool.blob(fit="unit test")
        got = MultiQueryAttnPool.spec_from_state_dict(blob["state_dict"])
        want = dict(dim=D, queries=spec["queries"], per_view=spec["per_view"], n_views=V,
                    out_dim=spec.get("out_dim"), query_norm=spec.get("query_norm", False),
                    share_queries=spec.get("share_queries", False), patches=spec.get("patches"))
        assert got == want, (got, want)
        assert torch.equal(MultiQueryAttnPool.from_blob(blob).eval()(x, None), y), "round trip changed it"
        assert torch.equal(MultiQueryAttnPool.from_blob(blob["state_dict"]).eval()(x, None), y), \
            "bare state dict round trip changed it"
        print(f"round trip ok {spec}: out {tuple(y.shape)} pooled_dim={blob['pooled_dim']} params={blob['params']:,}")


def test_released_readouts():
    """Every released V directory holds a pv4 readout that reloads to the shape the RL code expects (777 = 768 + 9)."""
    dirs = sorted((U.root("CKPT_ROOT") / "value").glob("v_dino_*_mqattn_pv4*"))
    if not dirs:
        U.skip("no V directory below $CKPT_ROOT/value")
    x = _x()
    for d in dirs:
        blob = torch.load(d / "mqattn_pool.pt", map_location="cpu", weights_only=False)
        pool = MultiQueryAttnPool.from_blob(blob).eval()
        spec = MultiQueryAttnPool.spec_from_state_dict(blob["state_dict"])
        assert (spec["queries"], spec["per_view"], spec["out_dim"]) == (4, True, 768), (d.name, spec)
        assert pool(x, None).shape == (B, 768)
        print(f"released readout ok: {d.name} {spec}")


if __name__ == "__main__":
    U.main(globals())
