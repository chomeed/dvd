"""Rounds 2 + 3 of the critic study at another checkpoint / task of a matched Q vs DVD pair (s44, 600k config).
User 2026-10-01: "regarding the OOD action ablation, could you also check later ckpts of Q and DVD of the same config?" and
"after this could you queue the same process for square d0?".
usage: [QOOD_TASK=hammer_cleanup_d1|square_d0|square_d1|threading_d0] QOOD_STEP=600000 python late.py r2 <ref|screen|select|cands|gt> <arm> [--worker i --workers n]
       ... python late.py r2 pickseeds          (non-hammer tasks: choose the 5 eval seeds from the 175k reference episodes)
       ... python late.py r3 <cands|gt> <arm> [--dim d --worker i --workers n]
       ... python late.py ready                 (exit 0 once both checkpoints of that step exist and are complete)
round2.py / round3.py are imported unchanged; only checkpoints, eval seeds and output dirs are swapped: same screening rule,
x grid + 4 prior draws, 16 CRN rollouts continuing with the arm's own actor AT THAT STEP, both critics from that step.
Outputs: hammer r2_<k>k/, r3_<k>k/ (175k = round2/round3's r2/, r3/); square_d0 r2_sq_<k>k/; square_d1 r2_sd1_<k>k/;
threading_d0 r2_thr_<k>k/ (and r3_* alike). User 2026-10-01: "and also queue the ablation study on the other two tasks too".
Round-3 decisions = the round-2 decisions whose 13 candidates' success counts span >= 5/16 (the rule used at 175k).
Seeds of the non-hammer tasks: at 175k a reference episode is run for seeds 10000-10009 with each policy; up to 2 seeds where a policy fails
(both-fail first), then the lowest seeds where both succeed, 5 in total (hammer: 2 failing + 3 succeeding); reused at 400k/600k."""
import argparse, glob, json, os, sys, time
from pathlib import Path
import round2 as R2, round3 as R3

TASK = os.environ.get("QOOD_TASK", "hammer_cleanup_d1")
STEP = int(os.environ["QOOD_STEP"]); K = f"{STEP // 1000}k"
HERE = Path(__file__).resolve().parent
RUNS = {"hammer_cleanup_d1": ("dsrl-na-q_hammer_cleanup_d1_*_600k_s44", "dsrl-na-svg1.0_hammer_cleanup_d1_*_dpinitvtd_*_nodistill_600k_s44"),
        "square_d0": ("dsrl-na-q_square_d0_*_600k_s44", "dsrl-na-svg1.0_square_d0_*_dpinitvtd_*_nodistill_600k_s44"),
        "square_d1": ("dsrl-na-q_square_d1_*_600k_s44", "dsrl-na-svg1.0_square_d1_*_dpinitvtd_*_nodistill_600k_s44"),
        "threading_d0": ("dsrl-na-q_threading_d0_*_600k_s44", "dsrl-na-svg1.0_threading_d0_*_dpinitvtd_*_nodistill_600k_s44")}[TASK]
TAG = {"hammer_cleanup_d1": "", "square_d0": "_sq", "square_d1": "_sd1", "threading_d0": "_thr"}[TASK]
R2.OUT = HERE / f"r2{TAG}_{K}"
R3.OUT = HERE / f"r3{TAG}_{K}"
SEEDFILE = HERE / f"r2{TAG}_175k" / "seeds.json"
CANDIDATE_SEEDS = list(range(10000, 10010))

def nearest(run_glob, strict=True):
    zs = [z for z in R2.P.glob(f"{run_glob}/checkpoints/0*.zip")]
    if not zs:
        if strict: raise SystemExit(f"no checkpoints for {run_glob}")
        return None
    z = min(zs, key=lambda p: abs(int(p.stem) - STEP))
    if abs(int(z.stem) - STEP) >= 1000:
        if strict: raise SystemExit(f"no checkpoint within 1000 steps of {STEP} for {run_glob} (nearest {z.name})")
        return None
    return z

def ready():
    for g in RUNS:
        z = nearest(g, strict=False)
        if z is None:
            return False
        for f in (z, z.with_name(z.stem + "_svg.pt")):
            if not f.exists() or time.time() - f.stat().st_mtime < 120:      # still being written
                return False
    return True

def decisions():
    out = []
    for arm in ("q", "dvd"):
        for seed, ts in json.loads((R2.OUT / arm / "selected.json").read_text()).items():
            for t in ts:
                fs = sorted(glob.glob(str(R2.OUT / arm / f"ep{seed}" / "gt" / f"t{t:03d}_c*.json")))
                succ = [sum(k >= 0 for k in json.load(open(f))["k"]) for f in fs]
                if len(fs) == 13 and max(succ) - min(succ) >= 5:
                    out.append((arm, int(seed), t))
    return sorted(out)
R3.decisions = decisions

def pickseeds(a):
    import pickle
    res = {}
    for s in CANDIDATE_SEEDS:
        res[s] = tuple(pickle.load(open(R2.OUT / arm / f"ep{s}" / "ref.pkl", "rb"))["meta"]["success"] for arm in ("q", "dvd"))
    both_fail = [s for s in CANDIDATE_SEEDS if not any(res[s])]
    one_fail = [s for s in CANDIDATE_SEEDS if any(res[s]) and not all(res[s])]
    both_ok = [s for s in CANDIDATE_SEEDS if all(res[s])]
    seeds = sorted((both_fail + one_fail)[:2] + both_ok[:5 - len((both_fail + one_fail)[:2])])
    SEEDFILE.write_text(json.dumps(dict(seeds=seeds, outcomes={s: dict(q=res[s][0], dvd=res[s][1]) for s in CANDIDATE_SEEDS})))
    print(f"[late {TASK} {K}] outcomes (q, dvd success): {res}\n  -> seeds {seeds}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("round", choices=["r2", "r3", "ready"])
    ap.add_argument("stage", nargs="?", default="")
    ap.add_argument("arm", nargs="?", default="q", choices=["q", "dvd"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--dim", type=int, default=1)
    a = ap.parse_args()
    if a.round == "ready":
        sys.exit(0 if ready() else 1)
    R2.CKPT = {"q": nearest(RUNS[0]), "dvd": nearest(RUNS[1])}
    if TASK != "hammer_cleanup_d1":
        if a.round == "r2" and a.stage == "ref" and STEP == 175000:
            R2.SEEDS = CANDIDATE_SEEDS
        elif SEEDFILE.exists():
            R2.SEEDS = json.loads(SEEDFILE.read_text())["seeds"]
        elif a.stage != "pickseeds":
            raise SystemExit(f"{SEEDFILE} missing: run the 175k reference episodes and pickseeds first")
    print(f"[late {TASK} {K}] q={R2.CKPT['q'].name} dvd={R2.CKPT['dvd'].name} seeds={R2.SEEDS} -> {R2.OUT.name}, {R3.OUT.name}", flush=True)
    {"r2": {"ref": R2.stage_ref, "screen": R2.stage_screen, "select": R2.stage_select, "cands": R2.stage_cands, "gt": R2.stage_gt,
            "pickseeds": pickseeds},
     "r3": {"cands": R3.stage_cands, "gt": R3.stage_gt}}[a.round][a.stage](a)
