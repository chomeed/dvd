"""Round 2 of the critic study: several episodes, decision-critical states, matched-spacing candidates, more rollouts.

stages (each resumable: finished outputs are skipped)
  ref     <arm>: one eval episode per seed with the arm's own actor (snapshots + critic values on the grid)   -> r2/<arm>/ep<seed>/
  screen  <arm> --worker i --workers n: every 2nd decision, candidates m in {-0.4, 0, +0.4}, 3 rollouts each
  select  <arm>: up to 3 decisions per episode where the outcome is uncertain or depends on the action
  cands   <arm>: at each selected decision, 9 grid offsets (-0.4..0.4 step 0.1) + 4 prior draws; both critics' values;
                 support (k-NN and Mahalanobis vs 256 prior draws)
  gt      <arm> --worker i --workers n: 16 common-random-number rollouts per candidate, continuation = the arm's actor
arms: q = DSRL Q s44 @175k, dvd = DSRL DVD s44 @175k
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import qood as Q

P = Path("/NHNHOME/WORKSPACE/26moe002_B/chomeed/post_iclr2027")
CKPT = {"q": sorted(P.glob("dsrl-na-q_*600k_s44/checkpoints/0175020.zip"))[0],
        "dvd": sorted(P.glob("dsrl-na-svg1.0*600k_s44/checkpoints/0175012.zip"))[0]}
SEEDS = [10000, 10001, 10002, 10006, 10009]
GRID = [round(x, 2) for x in np.arange(-0.4, 0.41, 0.1)]
SCREEN = [-0.4, 0.0, 0.4]
OUT = Path(__file__).resolve().parent / "r2"


def mem_guard(limit_gib=350):
    cg = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().split("0::")[1].strip().replace("/init.scope", "")
    while True:
        s = 0
        for line in open(cg + "/memory.stat"):
            k, v = line.split()
            if k in ("anon", "shmem"):
                s += int(v)
        if s / 2 ** 30 < limit_gib:
            return
        print(f"[mem-guard] container at {s / 2 ** 30:.0f} GiB >= {limit_gib}; waiting", flush=True)
        time.sleep(120)


def ep_dir(arm, seed):
    return OUT / arm / f"ep{seed}"


def rollouts(model, env, snap, action, n, seed0):
    """n rollouts from the snapshot: execute `action`, then the actor to the end. returns returns, success chunk idx, eef disp"""
    G, K, disp = [], [], None
    for i in range(n):
        obs_vec = Q.restore(env, snap)
        Q.seed_all(seed0 + i)
        p0 = env.cur_pos[:3].copy()
        obs_vec, r, term, trunc, info = env.step(action)
        if disp is None:
            disp = (env.cur_pos[:3] - p0).tolist()
        k = 0
        while not (term or trunc):
            obs_vec, r, term, trunc, info = env.step(Q.act(model, obs_vec))
            k += 1
        G.append(Q.GAMMA8 ** k if term else 0.0); K.append(k if term else -1)
    return G, K, disp


class Episodes:
    """one actor model + env per worker; resets the env to an episode's seed only when switching episodes"""
    def __init__(self, arm):
        self.arm = arm
        self.model, self.env, self.flow, _ = Q.build(CKPT[arm], SEEDS[0])
        Q.track_raw(self.env)
        self.cur, self.refs = None, {}

    def ref(self, seed):
        if seed not in self.refs:
            self.refs[seed] = pickle.load(open(ep_dir(self.arm, seed) / "ref.pkl", "rb"))
        if self.cur != seed:
            self.env.env.reset(seed=seed); self.cur = seed
        return self.refs[seed]


def stage_ref(a):
    for seed in SEEDS:
        d = ep_dir(a.arm, seed)
        if (d / "ref.pkl").exists():
            continue
        ns = argparse.Namespace(ckpt=CKPT[a.arm], seed=seed, out=str(d), axis=0, deltas=GRID)
        Q.phase_ref(ns)


def stage_screen(a):
    jobs = []
    for seed in SEEDS:
        meta = pickle.load(open(ep_dir(a.arm, seed) / "ref.pkl", "rb"))["meta"]
        jobs += [(seed, t, mi) for t in range(0, meta["n_decisions"], 2) for mi in range(len(SCREEN))]
    jobs = jobs[a.worker::a.workers]
    E = Episodes(a.arm)
    for seed, t, mi in jobs:
        f = ep_dir(a.arm, seed) / "screen" / f"t{t:03d}_m{mi}.json"
        if f.exists():
            continue
        mem_guard()
        f.parent.mkdir(exist_ok=True)
        rec = E.ref(seed)["recs"][t]
        act = Q.perturb(rec["action"], E.flow, SCREEN[mi], 0)
        G, K, disp = rollouts(E.model, E.env, rec["snap"], act, 3, 7_000_000 + 1000 * t)
        f.write_text(json.dumps(dict(seed=seed, t=t, m=SCREEN[mi], G=G, k=K, disp=disp)))
        print(f"[screen {a.arm} w{a.worker}] ep{seed} t={t} m={SCREEN[mi]:+.1f}: {sum(k >= 0 for k in K)}/3", flush=True)


def stage_select(a):
    sel = {}
    for seed in SEEDS:
        d = ep_dir(a.arm, seed)
        S = {}
        for f in glob.glob(str(d / "screen" / "*.json")):
            r = json.load(open(f)); S.setdefault(r["t"], {})[r["m"]] = np.mean(np.array(r["k"]) >= 0)
        cand = []
        for t, v in S.items():
            if len(v) < len(SCREEN):
                continue
            p0, spread = v[0.0], max(v.values()) - min(v.values())
            if 0.1 <= p0 <= 0.9 or spread >= 0.5:
                cand.append((p0 * (1 - p0) + spread, t))
        sel[seed] = sorted(t for _, t in sorted(cand, reverse=True)[:3])
        print(f"{a.arm} ep{seed}: {len(cand)} informative decisions, selected {sel[seed]}")
    (OUT / a.arm / "selected.json").write_text(json.dumps(sel))


def stage_cands(a):
    sel = json.loads((OUT / a.arm / "selected.json").read_text())
    critics = {k: Q.build(CKPT[k], SEEDS[0], with_dvd=True)[0] for k in ("q", "dvd")}
    E = Episodes(a.arm)
    flow, dev = E.flow, E.model.device
    for seed, ts in sel.items():
        seed = int(seed)
        f = ep_dir(a.arm, seed) / "cands.pkl"
        if f.exists():
            continue
        ref = E.ref(seed)
        out = {}
        for t in ts:
            rec = ref["recs"][t]
            obs_vec = Q.restore(E.env, rec["snap"])
            g = torch.Generator(device=dev); g.manual_seed(5_000_000 + t)
            with torch.no_grad():
                draws = [E.model.diffusion_policy(obs_vec[None], torch.randn(1, E.model.diffusion_act_chunk, E.model.diffusion_act_dim,
                                                                             device=dev, generator=g), return_numpy=True)[0]
                         for _ in range(256 + 4)]
            acts = [Q.perturb(rec["action"], flow, m, 0) for m in GRID] + draws[256:]
            kinds = ["grid"] * len(GRID) + ["prior"] * 4
            vals = {k: Q.critic_values(c, flow, obs_vec, E.env.cur_frame, E.env.prev_frame, acts).mean(1)
                    for k, c in critics.items()}
            X = lambda A: torch.as_tensor(np.stack([x.reshape(flow.horizon, flow.action_dim)[flow.executed].reshape(-1)
                                                    for x in A]), dtype=torch.float64)
            D, C = X(draws[:256]), X(acts)
            dd = torch.cdist(D, D); dd.fill_diagonal_(float("inf"))
            refk = dd.topk(5, largest=False).values.mean(1).median()
            knn = (torch.cdist(C, D).topk(5, largest=False).values.mean(1) / refk).numpy()
            mu = D.mean(0); S = torch.cov(D.T) + 1e-6 * torch.eye(D.shape[1], dtype=D.dtype)
            Si = torch.linalg.inv(0.8 * S + 0.2 * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0], dtype=D.dtype))
            md = lambda Y: torch.sqrt((((Y - mu) @ Si) * (Y - mu)).sum(1))
            mahal = (md(C) / md(D).median()).numpy()
            out[t] = dict(actions=acts, kinds=kinds, offsets=GRID + [None] * 4, values=vals, knn=knn, mahal=mahal)
            print(f"{a.arm} ep{seed} t={t}: knn {np.round(knn, 2)}", flush=True)
        pickle.dump(out, open(f, "wb"))


def stage_gt(a):
    sel = json.loads((OUT / a.arm / "selected.json").read_text())
    jobs = []
    for seed, ts in sel.items():
        C = pickle.load(open(ep_dir(a.arm, int(seed)) / "cands.pkl", "rb"))
        jobs += [(int(seed), t, ci) for t in ts for ci in range(len(C[t]["actions"]))]
    jobs = jobs[a.worker::a.workers]
    E = Episodes(a.arm)
    cands = {}
    for seed, t, ci in jobs:
        f = ep_dir(a.arm, seed) / "gt" / f"t{t:03d}_c{ci:02d}.json"
        if f.exists():
            continue
        mem_guard()
        f.parent.mkdir(exist_ok=True)
        if seed not in cands:
            cands[seed] = pickle.load(open(ep_dir(a.arm, seed) / "cands.pkl", "rb"))
        rec = E.ref(seed)["recs"][t]
        G, K, disp = rollouts(E.model, E.env, rec["snap"], cands[seed][t]["actions"][ci], a.n, 9_000_000 + 1000 * t)
        f.write_text(json.dumps(dict(seed=seed, t=t, c=ci, G=G, k=K, disp=disp)))
        print(f"[gt {a.arm} w{a.worker}] ep{seed} t={t} c={ci}: {sum(k >= 0 for k in K)}/{a.n}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ref", "screen", "select", "cands", "gt"])
    ap.add_argument("arm", choices=["q", "dvd"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    {"ref": stage_ref, "screen": stage_screen, "select": stage_select, "cands": stage_cands, "gt": stage_gt}[a.stage](a)
