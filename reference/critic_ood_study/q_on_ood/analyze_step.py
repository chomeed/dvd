"""Critic study at any checkpoint: round-2 headline (x shifts + 4 prior draws) and round-3 per-axis results (9 shifted actions per
axis, m = 0 from round 2) for Q(s,a) vs gamma^8 V(f(s,a)), pooled and split by policy / episode outcome.
usage: python analyze_step.py <r2 dir> <r3 dir> [--tag 600k]     e.g. r2 r3 (175k), r2_600k r3_600k
writes figs_steps/results_<tag>.pkl"""
import argparse, glob, json, os, pickle
import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("r2"); ap.add_argument("r3"); ap.add_argument("--tag", default="")
a = ap.parse_args()
GAP = 5
NAMES = {0: "x", 1: "y", 2: "z", 3: "rot_x", 4: "rot_y", 5: "rot_z", 6: "gripper"}
OFFS9 = np.round(np.arange(-0.4, 0.41, 0.1), 2)

def outcome(arm, seed):
    import pickle as pk
    return bool(pk.load(open(f"{a.r2}/{arm}/ep{seed}/ref.pkl", "rb"))["meta"]["success"])

rows2 = []                                                     # round-2 style: 13 candidates
for arm in ("q", "dvd"):
    sel = json.load(open(f"{a.r2}/{arm}/selected.json"))
    for seed, ts in sel.items():
        if not ts: continue
        C = pickle.load(open(f"{a.r2}/{arm}/ep{seed}/cands.pkl", "rb")); ok = outcome(arm, seed)
        for t in ts:
            c = C[t]; R = [json.load(open(f"{a.r2}/{arm}/ep{seed}/gt/t{t:03d}_c{ci:02d}.json")) for ci in range(len(c["actions"]))]
            succ = np.array([sum(k >= 0 for k in r["k"]) for r in R]); n = len(R[0]["k"])
            rows2.append(dict(arm=arm, seed=int(seed), t=t, succ=succ, p=succ / n, mc=np.array([np.mean(r["G"]) for r in R]),
                              kinds=np.array(c["kinds"]), q=np.asarray(c["values"]["q"]), dvd=np.asarray(c["values"]["dvd"]),
                              knn=np.asarray(c["knn"]), fail=not ok))
inf2 = [r for r in rows2 if r["succ"].max() - r["succ"].min() >= GAP]
rows = []                                                      # per-axis 9-point rows
for r in inf2:
    g = r["kinds"] == "grid"
    rows.append(dict(arm=r["arm"], seed=r["seed"], t=r["t"], dim=0, succ=r["succ"][g], p=r["p"][g], mc=r["mc"][g], q=r["q"][g],
                     dvd=r["dvd"][g], knn=r["knn"][g], fail=r["fail"]))
C3 = {arm: pickle.load(open(f"{a.r3}/{arm}/cands.pkl", "rb")) for arm in ("q", "dvd") if os.path.exists(f"{a.r3}/{arm}/cands.pkl")}
for d in range(1, 7):
    for r in inf2:
        fs = [f"{a.r3}/{r['arm']}/ep{r['seed']}/dim{d}/t{r['t']:03d}_c{ci:02d}.json" for ci in range(8)]
        if r["arm"] not in C3 or not all(os.path.exists(f) for f in fs): continue
        R = [json.load(open(f)) for f in fs]; R.insert(4, json.load(open(f"{a.r2}/{r['arm']}/ep{r['seed']}/gt/t{r['t']:03d}_c04.json")))
        c = C3[r["arm"]][(r["seed"], r["t"], d)]
        val = {k: np.insert(np.asarray(c["values"][k], float), 4, c["v0"][k]) for k in ("q", "dvd")}
        succ = np.array([sum(k >= 0 for k in x["k"]) for x in R]); n = len(R[0]["k"])
        rows.append(dict(arm=r["arm"], seed=r["seed"], t=r["t"], dim=d, succ=succ, p=succ / n, mc=np.array([np.mean(x["G"]) for x in R]),
                         q=val["q"], dvd=val["dvd"], knn=np.insert(np.asarray(c["knn"]), 4, c["knn0"]), fail=r["fail"]))

def pairs(r):
    N = len(r["mc"]); return [(i, j) for i in range(N) for j in range(i + 1, N) if abs(int(r["succ"][i]) - int(r["succ"][j])) >= GAP]
def acc(rs, k):
    v = [np.sign(r[k][i] - r[k][j]) == np.sign(r["mc"][i] - r["mc"][j]) for r in rs for i, j in pairs(r)]
    return np.mean(v) if v else np.nan
def regret(rs, k): return np.mean([r["p"].max() - r["p"][np.argmax(r[k])] for r in rs]) if rs else np.nan
def mono(rs, k): return np.mean([abs(spearmanr(np.arange(len(r[k])), r[k]).statistic) >= 0.95 for r in rs]) if rs else np.nan
def boot(rs, f, B=2000):
    if not rs: return np.array([np.nan, np.nan])
    rng = np.random.default_rng(0)
    return np.nanpercentile([f([rs[i] for i in rng.integers(0, len(rs), len(rs))]) for _ in range(B)], [2.5, 97.5])
def informative(r): return r["succ"].max() - r["succ"].min() >= GAP

summary = {}
print(f"== {a.tag or a.r2}: {len(rows2)} selected decisions, {len(inf2)} action-sensitive along x "
      f"(Q-policy {sum(r['arm'] == 'q' for r in inf2)}, DVD-policy {sum(r['arm'] == 'dvd' for r in inf2)}; failed-episode {sum(r['fail'] for r in inf2)})")
for k in ("q", "dvd"):
    ci = boot(inf2, lambda x: acc(x, k)); cr = boot(inf2, lambda x: regret(x, k))
    summary[("headline", k)] = (acc(inf2, k), ci, regret(inf2, k), cr)
    print(f"  headline (x shifts + prior draws) {'Q   ' if k == 'q' else 'V(f)'}: acc {acc(inf2, k):.2f} [{ci[0]:.2f},{ci[1]:.2f}]  regret {regret(inf2, k):.2f} [{cr[0]:.2f},{cr[1]:.2f}]")
d_ = boot(inf2, lambda x: acc(x, "dvd") - acc(x, "q")); print(f"  V(f) - Q accuracy {acc(inf2, 'dvd') - acc(inf2, 'q'):+.2f} [{d_[0]:+.2f},{d_[1]:+.2f}]")
print(f"{'axis':8s} {'dec':>3s} {'inf':>3s} {'pairs':>5s} | {'acc Q':>5s} {'acc V':>5s} {'V-Q [95% CI]':>20s} | {'reg Q':>5s} {'reg V':>5s} | {'mono Q':>6s} {'mono V':>6s}")
for d, nm in NAMES.items():
    rd = [r for r in rows if r["dim"] == d]
    if not rd: print(f"{nm:8s} (not run)"); continue
    inf = [r for r in rd if informative(r)]
    ci = boot(inf, lambda x: acc(x, "dvd") - acc(x, "q"))
    summary[(d, "all")] = dict(acc_q=acc(inf, "q"), acc_v=acc(inf, "dvd"), d_ci=ci, reg_q=regret(inf, "q"), reg_v=regret(inf, "dvd"),
                               mono_q=mono(rd, "q"), mono_v=mono(rd, "dvd"), n=len(rd), n_inf=len(inf))
    s = summary[(d, "all")]
    print(f"{nm:8s} {len(rd):3d} {len(inf):3d} {sum(len(pairs(r)) for r in inf):5d} | {s['acc_q']:5.2f} {s['acc_v']:5.2f} "
          f"{s['acc_v'] - s['acc_q']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] | {s['reg_q']:5.2f} {s['reg_v']:5.2f} | {s['mono_q']:6.2f} {s['mono_v']:6.2f}")
allinf = [r for r in rows if informative(r)]
ci = boot(allinf, lambda x: acc(x, "dvd") - acc(x, "q"))
print(f"{'ALL axes':8s} {len(rows):3d} {len(allinf):3d} {sum(len(pairs(r)) for r in allinf):5d} | {acc(allinf, 'q'):5.2f} {acc(allinf, 'dvd'):5.2f} "
      f"{acc(allinf, 'dvd') - acc(allinf, 'q'):+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] | {regret(allinf, 'q'):5.2f} {regret(allinf, 'dvd'):5.2f} | {mono(rows, 'q'):6.2f} {mono(rows, 'dvd'):6.2f}")
for name, sel in (("success eps", lambda r: not r["fail"]), ("failed eps", lambda r: r["fail"]), ("Q policy", lambda r: r["arm"] == "q"), ("DVD policy", lambda r: r["arm"] == "dvd")):
    rs = [r for r in allinf if sel(r)]
    print(f"  all axes, {name:11s}: {len(rs):3d} inf. | acc Q {acc(rs, 'q'):.2f}  V(f) {acc(rs, 'dvd'):.2f} | regret Q {regret(rs, 'q'):.2f}  V(f) {regret(rs, 'dvd'):.2f}")
os.makedirs("figs_steps", exist_ok=True)
pickle.dump(dict(rows=rows, rows2=rows2, summary=summary), open(f"figs_steps/results_{a.tag or os.path.basename(a.r2)}.pkl", "wb"))
