"""Round 3: the round-2 nearby-action test along the other action dimensions (y, z, gripper, rotations).
At the 17 action-sensitive round-2 decisions: 8 offsets m in {-0.4..-0.1, 0.1..0.4} added to ONE action dim of the executed chunk
(all 8 steps, clipped to [-1, 1]); the executed chunk (m = 0) reuses its round-2 ground truth. 16 rollouts per candidate with the
round-2 seeds (common random numbers across dims and with round 2). Both critics score every candidate; support = k-NN distance
to 256 prior draws (1 = typical draw).
stages: cands (both critics + support, all dims) | gt <arm> --dim d --worker i --workers n
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np, torch
import qood as Q, round2 as R2

DIMS = {1: "y", 2: "z", 6: "gripper", 5: "rot_z", 3: "rot_x", 4: "rot_y"}
OFFS = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
OUT = Path(__file__).resolve().parent / "r3"

def decisions():
    rows = pickle.load(open(Path(__file__).resolve().parent / "figs_r2" / "r2_results.pkl", "rb"))["rows"]
    return sorted([(r["arm"], r["seed"], r["t"]) for r in rows if r["succ"].max() - r["succ"].min() >= 5])

def stage_cands(a):
    critics = {k: Q.build(R2.CKPT[k], R2.SEEDS[0], with_dvd=True)[0] for k in ("q", "dvd")}
    for arm in ("q", "dvd"):
        f = OUT / arm / "cands.pkl"
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        E = R2.Episodes(arm); flow, dev = E.flow, E.model.device
        out = {}
        for (ar, seed, t) in decisions():
            if ar != arm:
                continue
            rec = E.ref(seed)["recs"][t]
            obs_vec = Q.restore(E.env, rec["snap"])
            g = torch.Generator(device=dev); g.manual_seed(5_000_000 + t)
            with torch.no_grad():
                draws = [E.model.diffusion_policy(obs_vec[None], torch.randn(1, E.model.diffusion_act_chunk, E.model.diffusion_act_dim,
                                                                             device=dev, generator=g), return_numpy=True)[0] for _ in range(256)]
            X = lambda A: torch.as_tensor(np.stack([x.reshape(flow.horizon, flow.action_dim)[flow.executed].reshape(-1) for x in A]), dtype=torch.float64)
            Dd = X(draws); dd = torch.cdist(Dd, Dd); dd.fill_diagonal_(float("inf"))
            refk = dd.topk(5, largest=False).values.mean(1).median()
            for dim in DIMS:
                acts = [Q.perturb(rec["action"], flow, m, dim) for m in OFFS]
                vals = {k: Q.critic_values(c, flow, obs_vec, E.env.cur_frame, E.env.prev_frame, acts).mean(1) for k, c in critics.items()}
                knn = (torch.cdist(X(acts), Dd).topk(5, largest=False).values.mean(1) / refk).numpy()
                # the executed chunk (m = 0) scored the same way, for the 9-point curve
                v0 = {k: float(Q.critic_values(c, flow, obs_vec, E.env.cur_frame, E.env.prev_frame, [rec["action"]]).mean()) for k, c in critics.items()}
                out[(seed, t, dim)] = dict(actions=acts, offsets=OFFS, values=vals, knn=knn, v0=v0,
                                           knn0=float((torch.cdist(X([rec["action"]]), Dd).topk(5, largest=False).values.mean(1) / refk)[0]))
            print(f"{arm} ep{seed} t={t}: candidates for {len(DIMS)} dims", flush=True)
        pickle.dump(out, open(f, "wb"))

def stage_gt(a):
    C = pickle.load(open(OUT / a.arm / "cands.pkl", "rb"))
    jobs = [(seed, t, ci) for (seed, t, dim) in sorted(C) if dim == a.dim for ci in range(len(OFFS))][a.worker::a.workers]
    E = R2.Episodes(a.arm)
    for seed, t, ci in jobs:
        f = OUT / a.arm / f"ep{seed}" / f"dim{a.dim}" / f"t{t:03d}_c{ci:02d}.json"
        if f.exists():
            continue
        R2.mem_guard(); f.parent.mkdir(parents=True, exist_ok=True)
        rec = E.ref(seed)["recs"][t]
        G, K, end = [], [], None
        for i in range(a.n):
            Q.restore(E.env, rec["snap"]); Q.seed_all(9_000_000 + 1000 * t + i)        # the round-2 seeds
            obs_vec, r, term, trunc, info = E.env.step(C[(seed, t, a.dim)]["actions"][ci])
            if end is None:
                end = np.asarray(E.env.cur_pos, dtype=np.float64).tolist()              # full proprio after the chunk
            k = 0
            while not (term or trunc):
                obs_vec, r, term, trunc, info = E.env.step(Q.act(E.model, obs_vec)); k += 1
            G.append(Q.GAMMA8 ** k if term else 0.0); K.append(k if term else -1)
        f.write_text(json.dumps(dict(seed=seed, t=t, dim=a.dim, c=ci, m=OFFS[ci], G=G, k=K, end_state=end)))
        print(f"[gt {a.arm} dim{a.dim} w{a.worker}] ep{seed} t={t} m={OFFS[ci]:+.1f}: {sum(k >= 0 for k in K)}/{a.n}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cands", "gt", "exec_end"])
    ap.add_argument("arm", nargs="?", default="q", choices=["q", "dvd"])
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    {"cands": stage_cands, "gt": stage_gt}[a.stage](a)
