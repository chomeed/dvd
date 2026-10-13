"""Figures for the 2026-10-08 weekly meeting: the OOD action ablation (critic study).

Question: when the executed action chunk is shifted to nearby, increasingly out-of-distribution actions, which critic still ranks
them by their true outcome -- a learned Q(s,a) (DSRL Q run) or DVD's gamma^8 V(f(s,a)) through the DINO world model (DSRL DVD run)?
Data: post_iclr2027/q_on_ood/figs_steps/results_<cell>.pkl (analyze_step.py). Per decision: 9 candidate actions = the executed chunk
shifted along one action axis by -0.4 ... +0.4; ground truth = 16 common-random-number rollouts per candidate with the episode's own
policy; both critics score the same candidates. Decisions come from the s44 DSRL Q and DSRL DVD runs at the same checkpoint
(175k/400k/600k), 4 tasks. A pair of candidates counts when their success counts differ by >= 5/16; accuracy = fraction of such
pairs a critic orders like the true discounted return. Regret = (best success - success of the critic's top-scored candidate) / 16.
CIs: 95% bootstrap over decisions (4000 resamples).
usage: python make_figures.py   (writes fig1..fig4 .png/.pdf, numbers.csv into this folder)"""
import csv, os, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/NHNHOME/WORKSPACE/26moe002_B/chomeed/post_iclr2027/q_on_ood/figs_steps"
GAP, B = 5, 4000
TASKS = [("", "hammer"), ("sq_", "square_d0"), ("sd1_", "square_d1"), ("thr_", "threading")]
STEPS = ("175k", "400k", "600k")
AXES = ["x", "y", "z", "rot_x", "rot_y", "rot_z"]                      # dim 0..5; gripper shifts never changed the outcome
SHIFTS = np.round(np.linspace(-0.4, 0.4, 9), 1)
# the user's pastel scheme: Q #FFC7C2, V(f) #CDF4D3, with darker outlines of the same hue
FILL = {"q": "#FFC7C2", "dvd": "#CDF4D3"}; EDGE = {"q": "#C2453B", "dvd": "#2E8B45"}
LBL = {"q": "Q(s,a)  (DSRL Q)", "dvd": "γ⁸V(f(s,a))  (DSRL + DVD)"}
INK, INK2, MUTED, GRID, SURF = "#1b1b1a", "#55534e", "#8f8d87", "#e6e5e1", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.linewidth": 0.8})


def pairs(r):
    n = len(r["mc"])
    return [(i, j) for i in range(n) for j in range(i + 1, n) if abs(int(r["succ"][i]) - int(r["succ"][j])) >= GAP]


def counts(r):
    """per decision, cached: number of clearly-ordered pairs, and how many each critic orders like the true return"""
    if "_cnt" not in r:
        pp = pairs(r)
        r["_cnt"] = {"n": len(pp), **{k: sum(np.sign(r[k][i] - r[k][j]) == np.sign(r["mc"][i] - r["mc"][j]) for i, j in pp)
                                       for k in ("q", "dvd")}}
    return r["_cnt"]


def acc(rs, k):
    n = sum(counts(r)["n"] for r in rs)
    return float(sum(counts(r)[k] for r in rs) / n) if n else np.nan


def regret(rs, k):
    return float(np.mean([(r["succ"].max() - r["succ"][int(np.argmax(r[k]))]) / 16 for r in rs])) if rs else np.nan


def boot(rs, f=None, what=None):
    """95% bootstrap CI over decisions of acc_k (what=k) or of acc_dvd - acc_q (what='diff'), vectorised with numpy"""
    if not rs:
        return np.array([np.nan, np.nan])
    n = np.array([counts(r)["n"] for r in rs], float)
    cq = np.array([counts(r)["q"] for r in rs], float); cv = np.array([counts(r)["dvd"] for r in rs], float)
    idx = np.random.default_rng(0).integers(0, len(rs), (B, len(rs)))
    N = n[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        aq, av = cq[idx].sum(1) / N, cv[idx].sum(1) / N
    v = {"q": aq, "dvd": av, "diff": av - aq}[what]
    return np.nanpercentile(v, [2.5, 97.5])


def informative(r):
    return r["succ"].max() - r["succ"].min() >= GAP


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- data
CELLS = []                                    # (task, step, informative rows)
for tag, task in TASKS:
    for st in STEPS:
        rows = pickle.load(open(f"{SRC}/results_{tag}{st}.pkl", "rb"))["rows"]
        CELLS.append((task, st, [r for r in rows if informative(r)]))
ALL = [r for _, _, rs in CELLS for r in rs]

STATS = []
for task, st, rs in CELLS:
    d = dict(task=task, step=st, decisions=len(rs), pairs=sum(len(pairs(r)) for r in rs))
    for k in ("q", "dvd"):
        d[f"acc_{k}"] = acc(rs, k); d[f"acc_{k}_ci"] = boot(rs, what=k)
        d[f"regret_{k}"] = regret(rs, k)
        for arm in ("q", "dvd"):
            sub = [r for r in rs if r["arm"] == arm]
            d[f"acc_{k}_on_{arm}run"] = acc(sub, k)
            d[f"acc_{k}_on_{arm}run_ci"] = boot(sub, what=k)
    d["diff"] = d["acc_dvd"] - d["acc_q"]; d["diff_ci"] = boot(rs, what="diff")
    STATS.append(d)
mono = {k: float(np.mean([abs(spearmanr(np.arange(9), r[k])[0]) >= 0.9 for r in ALL])) for k in ("q", "dvd", "mc")}

with open(os.path.join(HERE, "numbers.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "step", "decisions", "pairs", "acc_Q", "acc_Q_lo", "acc_Q_hi", "acc_Vf", "acc_Vf_lo", "acc_Vf_hi", "Vf_minus_Q",
                "diff_lo", "diff_hi", "regret_Q", "regret_Vf", "acc_Q_on_Qrun", "acc_Vf_on_Qrun", "acc_Q_on_DVDrun", "acc_Vf_on_DVDrun"])
    for d in STATS:
        w.writerow([d["task"], d["step"], d["decisions"], d["pairs"], *(f"{x:.3f}" for x in (
            d["acc_q"], *d["acc_q_ci"], d["acc_dvd"], *d["acc_dvd_ci"], d["diff"], *d["diff_ci"], d["regret_q"], d["regret_dvd"],
            d["acc_q_on_qrun"], d["acc_dvd_on_qrun"], d["acc_q_on_dvdrun"], d["acc_dvd_on_dvdrun"]))])
print("wrote numbers.csv")


def sgn(x):
    """signed, 2 decimals; 3 when it would otherwise round to +-0.00"""
    return f"{x:+.3f}" if abs(x) < 0.005 else f"{x:+.2f}"


def dumbbell(ax, rows, labels, groups=None, keys=("acc_q", "acc_dvd"), xlim=(0.25, 0.8), annotate=True, title=None):
    """one horizontal row per dict (acc_<k>, acc_<k>_ci, diff, diff_ci); rows grouped by `groups` under bold headers"""
    groups = groups or [None] * len(rows)
    ys, y, lastg, heads = [], 0.0, object(), []
    for g in groups:
        if g != lastg:
            if ys:
                y += 0.55
            if g not in (None, ""):                    # "" / None: a gap only, no header
                heads.append((g, y)); y += 0.75
        ys.append(y); y += 1.0; lastg = g
    ys = -np.array(ys)
    ax.axvline(0.5, color=MUTED, lw=1, zorder=1)
    for yy, d in zip(ys, rows):
        a, b = d[keys[0]], d[keys[1]]
        ax.plot([a, b], [yy, yy], color="#cfcdc7", lw=2, solid_capstyle="round", zorder=2)
        for k, key, off in (("q", keys[0], 0.17), ("dvd", keys[1], -0.17)):
            lo, hi = d[key + "_ci"]
            ax.plot([lo, hi], [yy + off, yy + off], color=EDGE[k], lw=1.4, alpha=0.9, zorder=2, solid_capstyle="round")
            ax.scatter([d[key]], [yy], s=95, color=FILL[k], edgecolors=EDGE[k], linewidths=1.6, zorder=3)
        if annotate:
            lo, hi = d["diff_ci"]
            ax.text(xlim[1] + 0.012, yy, f"{sgn(d['diff'])}  [{sgn(lo)}, {sgn(hi)}]", va="center", ha="left", fontsize=10,
                    color=INK if lo > 0 else INK2, fontweight="bold" if lo > 0 else "normal")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    for g, gy in heads:
        ax.text(-0.01, -gy, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=11.5, fontweight="bold",
                color=INK)
    top = max([ys.max() + 0.2] + [-gy + 0.5 for _, gy in heads])
    ax.set_xlim(*xlim); ax.set_ylim(ys.min() - 0.7, top + 0.35)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.text(0.5, top + 0.3, "chance", ha="center", va="top", fontsize=9, color=MUTED)
    if title:
        ax.set_title(title, loc="left", fontsize=13, color=INK, pad=10)


# ---------------------------------------------------------------- fig 1
fig = plt.figure(figsize=(20, 8.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.4, 0.36, 0.95], wspace=0.3, left=0.055, right=0.985, top=0.85, bottom=0.27)

# A. one decision (threading 175k, the Q run's own episode, z axis): true success peaks at the executed action; Q is monotone
axA = fig.add_subplot(gs[0])
ex = next(r for r in pickle.load(open(f"{SRC}/results_thr_175k.pkl", "rb"))["rows"]
          if r["arm"] == "q" and r["seed"] == 10003 and r["t"] == 14 and r["dim"] == 2)
truth = ex["succ"] / 16
axA.bar(SHIFTS, truth, width=0.055, color="#d9d7d1", zorder=1, label="true success (16 rollouts)")
for k in ("q", "dvd"):
    v = np.asarray(ex[k], float); v = (v - v.min()) / (v.max() - v.min())
    axA.plot(SHIFTS, v, color=EDGE[k], lw=2, zorder=3, solid_capstyle="round")
    axA.scatter(SHIFTS, v, s=55, color=FILL[k], edgecolors=EDGE[k], linewidths=1.4, zorder=4)
    bi = int(np.argmax(v))
    axA.annotate("top pick", (SHIFTS[bi], v[bi]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9,
                 color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
axA.axvline(0, color=MUTED, lw=1, zorder=0)
axA.text(-0.015, 1.15, "executed\naction", fontsize=9, color=INK2, va="top", ha="right")
knn = np.asarray(ex["knn"])
# knn (round3.py): mean distance to the 5 nearest of 256 prior draws, divided by the draws' own median (1 = a typical draw)
axA.text(0.0, -0.135, f"Distance to 256 prior draws (1 = typical draw): {knn[4]:.1f} at the executed action, {knn[0]:.1f} and "
         f"{knn[-1]:.1f} at the ends.", transform=axA.transAxes, ha="left", va="top", fontsize=9.5, color=INK2)
axA.set_xticks(SHIFTS); axA.set_xticklabels([f"{s:+.1f}" if s else "0" for s in SHIFTS], fontsize=9.5)
axA.set_xlabel("shift of the executed action chunk along z")
axA.set_ylabel("true success  ·  critic score (min-max per decision)")
axA.set_ylim(0, 1.18); axA.set_yticks([0, 0.5, 1])
axA.grid(axis="y", color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)
axA.set_title("A.  One decision (threading, Q run's own episode)", loc="left", fontsize=13, color=INK, pad=10)
axA.text(0.0, -0.185, f"Across all {len(ALL):,} decisions the score is monotone in the shift (|Spearman ρ| ≥ 0.9)\n"
         f"for Q in {mono['q']:.0%} and for V(f) in {mono['dvd']:.0%}, but the true outcome only in {mono['mc']:.0%}:\n"
         "Q mostly extrapolates monotonically and misses the peak.", transform=axA.transAxes, fontsize=10, color=INK, va="top")

# B. dumbbells, 12 cells
axB = fig.add_subplot(gs[1])
dumbbell(axB, STATS, [d["step"] for d in STATS], [d["task"] for d in STATS], xlim=(0.25, 0.75),
         title="B.  Pairwise ranking accuracy on nearby actions")
axB.set_xlabel("fraction of clearly-ordered action pairs ranked correctly")
import matplotlib.transforms as mtransforms
axB.text(0.75 + 0.012, 1.01, "V(f) − Q  [95% CI]", transform=mtransforms.blended_transform_factory(axB.transData, axB.transAxes),
         ha="left", va="bottom", fontsize=10, color=INK2)

# C. regret scatter
axC = fig.add_subplot(gs[3])
lim = (0.2, 0.7)
axC.fill_between(lim, lim, lim[0], color="#eef8f0", zorder=0)
axC.plot(lim, lim, color=MUTED, lw=1, zorder=1)
axC.text(0.69, 0.212, "below the line: V(f) loses less", ha="right", va="bottom", fontsize=10, color=INK)
MARK = {"hammer": "o", "square_d0": "s", "square_d1": "D", "threading": "^"}
SHADE = {"175k": "#b9b6ae", "400k": "#77746c", "600k": "#2f2d29"}
for d in STATS:
    axC.scatter(d["regret_q"], d["regret_dvd"], marker=MARK[d["task"]], s=90, color=SHADE[d["step"]], edgecolors="white",
                linewidths=1.5, zorder=3)
axC.set_xlim(*lim); axC.set_ylim(*lim); axC.set_aspect("equal")
axC.set_xlabel("Q: success lost vs the best nearby action"); axC.set_ylabel("V(f): success lost vs the best nearby action")
axC.grid(color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)
axC.set_title("C.  Cost of acting on the top pick", loc="left", fontsize=13, color=INK, pad=10)
hm = [Line2D([], [], ls="", marker=MARK[t], color=INK2, markersize=8, label=t) for t in MARK]
hs = [Line2D([], [], ls="", marker="o", color=SHADE[s], markeredgecolor="white", markersize=9, label=s) for s in SHADE]
axC.legend(handles=hm + hs, loc="upper left", frameon=False, fontsize=9.5, handletextpad=0.3, ncol=2, columnspacing=1.0)

fig.legend(handles=[Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor=FILL[k], markeredgecolor=EDGE[k],
                           markeredgewidth=1.6, label=LBL[k]) for k in ("q", "dvd")]
           + [plt.Rectangle((0, 0), 1, 1, color="#d9d7d1", label="true success (panel A)")],
           loc="upper right", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.985, 0.995))
fig.suptitle("Which critic ranks nearby, out-of-distribution actions correctly?", x=0.05, ha="left", y=0.975, fontsize=16,
             color=INK, fontweight="bold")
fig.text(0.055, 0.02, "Each decision: the executed 8-step action chunk shifted along one axis (−0.4…+0.4, 9 candidates); truth = 16 "
         "common-random-number rollouts per candidate with the episode's own policy. Critics: DSRL Q and DSRL + DVD, seed 44, same "
         "checkpoint.\nB: pairs whose success differs by ≥ 5/16; bold = CI excludes 0. C: (best success − success of the critic's "
         "top-scored candidate)/16, mean over decisions; one point per task × checkpoint. CIs: 95% bootstrap over decisions.",
         fontsize=9, color=INK2, ha="left", va="bottom")
save(fig, "fig1_ood_action_ranking")

# ---------------------------------------------------------------- fig 2: per axis, pooled over checkpoints
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for ax, (tag, task) in zip(axs.flat, TASKS):
    rs = [r for t, _, rr in CELLS if t == task for r in rr]
    rows, labels = [], []
    for name in AXES + ["all axes"]:
        sub = rs if name == "all axes" else [r for r in rs if r["dim"] == AXES.index(name)]
        d = {f"acc_{k}": acc(sub, k) for k in ("q", "dvd")}
        for k in ("q", "dvd"):
            d[f"acc_{k}_ci"] = boot(sub, what=k)
        d["diff"] = d["acc_dvd"] - d["acc_q"]; d["diff_ci"] = boot(sub, what="diff")
        rows.append(d); labels.append(f"{name} (n={len(sub)})")
    dumbbell(ax, rows, labels, [None] * 6 + [""], xlim=(0.1, 0.85), title=f"{task}  (175k + 400k + 600k)")
for ax in axs[-1]:
    ax.set_xlabel("pairwise ranking accuracy")
fig.legend(handles=[Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor=FILL[k], markeredgecolor=EDGE[k],
                           markeredgewidth=1.6, label=LBL[k]) for k in ("q", "dvd")], loc="upper right", ncol=2, frameon=False,
           fontsize=11)
fig.suptitle("Ranking accuracy per shifted action axis", x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK)
fig.tight_layout(rect=(0.02, 0, 1, 0.95), h_pad=3, w_pad=9)
save(fig, "fig2_per_axis")

# ---------------------------------------------------------------- fig 3: over training
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
xs = np.arange(3)
for ax, (tag, task) in zip(axs, TASKS):
    ds = [d for d in STATS if d["task"] == task]
    ax.axhline(0.5, color=MUTED, lw=1)
    for k in ("q", "dvd"):
        m = [d[f"acc_{k}"] for d in ds]; lo = [d[f"acc_{k}_ci"][0] for d in ds]; hi = [d[f"acc_{k}_ci"][1] for d in ds]
        ax.fill_between(xs, lo, hi, color=EDGE[k], alpha=0.10, lw=0)
        ax.plot(xs, m, color=EDGE[k], lw=2, solid_capstyle="round")
        ax.scatter(xs, m, s=70, color=FILL[k], edgecolors=EDGE[k], linewidths=1.5, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(STEPS); ax.set_title(task, loc="left", fontsize=12.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlim(-0.3, 2.3)
    ax.set_xlabel("online env steps of the critic's run")
axs[0].set_ylabel("pairwise ranking accuracy"); axs[0].set_ylim(0.25, 0.8)
axs[0].text(-0.25, 0.505, "chance", fontsize=9, color=MUTED, ha="left", va="bottom")
fig.legend(handles=[Line2D([], [], color=EDGE[k], lw=2, marker="o", markersize=8, markerfacecolor=FILL[k], label=LBL[k])
                    for k in ("q", "dvd")], loc="upper right", ncol=2, frameon=False, fontsize=11)
fig.suptitle("The gap holds over training (bands: 95% bootstrap CI)", x=0.01, ha="left", fontsize=14.5, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, "fig3_over_training")

# ---------------------------------------------------------------- fig 4: home distribution
fig, axs = plt.subplots(1, 2, figsize=(16, 7.4))
for ax, arm, ttl in ((axs[0], "q", "Decisions from the Q run's own episodes"), (axs[1], "dvd", "Decisions from the DVD run's own episodes")):
    rows = []
    for d in STATS:
        sub = [r for t, s, rr in CELLS if t == d["task"] and s == d["step"] for r in rr if r["arm"] == arm]
        e = {f"acc_{k}": d[f"acc_{k}_on_{arm}run"] for k in ("q", "dvd")}
        for k in ("q", "dvd"):
            e[f"acc_{k}_ci"] = d[f"acc_{k}_on_{arm}run_ci"]
        e["diff"] = e["acc_dvd"] - e["acc_q"]; e["diff_ci"] = boot(sub, what="diff")
        rows.append(e)
    pooled = [r for _, _, rr in CELLS for r in rr if r["arm"] == arm]
    dumbbell(ax, rows, [d["step"] for d in STATS], [d["task"] for d in STATS], xlim=(0.15, 0.8),
             title=f"{ttl}\npooled: Q {acc(pooled, 'q'):.2f}, V(f) {acc(pooled, 'dvd'):.2f}  ({len(pooled)} decisions)")
    ax.set_xlabel("pairwise ranking accuracy")
fig.legend(handles=[Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor=FILL[k], markeredgecolor=EDGE[k],
                           markeredgewidth=1.6, label=LBL[k]) for k in ("q", "dvd")], loc="upper right", ncol=2, frameon=False,
           fontsize=11)
fig.suptitle("Not a home-field effect: V(f) wins even on the states Q was trained on", x=0.02, ha="left", fontsize=15,
             fontweight="bold", color=INK)
fig.tight_layout(rect=(0.03, 0, 1, 0.92), w_pad=12)
save(fig, "fig4_home_distribution")
print("monotone in the shift:", {k: f"{v:.0%}" for k, v in mono.items()}, "of", len(ALL), "decisions")

# ---------------------------------------------------------------- simplified: the example panel's view for every task, averaged
def mm(v):
    """a critic's scores rescaled to 0-1 within one decision (only the shape matters, raw scales differ between critics)"""
    v = np.asarray(v, float)
    return (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.full_like(v, 0.5)


TRUTH = "#d9d7d1"
MARK = {"q": "o", "dvd": "s"}                     # simplified figures: circle = Q, square = DVD
fig = plt.figure(figsize=(13.33, 7.5))
gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.15, 1], height_ratios=[1, 0.9], hspace=0.1, wspace=0.24,
                      left=0.115, right=0.985, top=0.78, bottom=0.17)
for j, (name, rs) in zip([0, 1, 2, 3, 5], [(t, [r for tt, _, rr in CELLS if tt == t for r in rr]) for _, t in TASKS] + [("all tasks", ALL)]):
    ax1, ax2 = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
    ax1.bar(SHIFTS, np.mean([r["succ"] / 16 for r in rs], 0), width=0.075, color=TRUTH, zorder=2)
    for k in ("q", "dvd"):
        v = np.mean([mm(r[k]) for r in rs], 0)
        ax2.plot(SHIFTS, v, color=EDGE[k], lw=2.4, zorder=3, solid_capstyle="round")
        ax2.scatter(SHIFTS, v, s=34, marker=MARK[k], color=FILL[k], edgecolors=EDGE[k], linewidths=1.3, zorder=4)
    for ax in (ax1, ax2):
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.set_xlim(-0.5, 0.5); ax.set_xticks([-0.4, 0, 0.4])
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(labelsize=14)
    ax1.set_ylim(0, 1); ax1.set_yticks([0, 0.5, 1]); ax1.set_xticklabels([])
    ax2.set_ylim(0.3, 0.8); ax2.set_yticks([0.3, 0.8]); ax2.set_xticklabels(["−0.4", "0", "+0.4"])
    if j == 0:
        ax1.set_yticklabels(["0%", "50%", "100%"]); ax2.set_yticklabels(["low", "high"])
        ax1.set_ylabel("true success", fontsize=15, labelpad=8)
        ax2.set_ylabel("critic score\n(rescaled, avg.)", fontsize=15, labelpad=8)
    else:
        ax1.set_yticklabels([]); ax2.set_yticklabels([])
    ax1.set_title(name, fontsize=17, color=INK, fontweight="bold" if name == "all tasks" else "normal", pad=8)
fig.text(0.55, 0.085, "shift of the executed action  (0 = what the policy did)", ha="center", fontsize=15, color=INK)
fig.text(0.03, 0.02, "True success = share of 16 rollouts that succeed. Each critic's scores are rescaled to 0–1 within a decision "
         "(their raw scales differ), then averaged.", ha="left", fontsize=12.5, color=INK2)
# truth peaks at 0 on every task except square_d1 (-0.1, flat top), DVD's score at 0 on every task: hence "around"
fig.suptitle("Averaged per task: success and DVD's score peak around the executed action", x=0.03, ha="left", y=0.965, fontsize=19,
             fontweight="bold", color=INK)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TRUTH, label="true success (16 rollouts per action)")]
           + [Line2D([], [], color=EDGE[k], lw=2.4, marker=MARK[k], markersize=7, markerfacecolor=FILL[k], label=LBL[k]) for k in ("q", "dvd")],
           loc="upper left", bbox_to_anchor=(0.025, 0.905), ncol=3, frameon=False, fontsize=14.5, handlelength=1.6, columnspacing=1.6)
save(fig, "fig1_simple_by_task")

# ---------------------------------------------------------------- simplified: ranking accuracy per task (alone, and with the example)
from matplotlib.patches import Patch

GROUPS = [(t, [r for tt, _, rr in CELLS if tt == t for r in rr]) for _, t in TASKS] + [("all tasks", ALL)]
ACC = {g: {k: acc(rs, k) for k in ("q", "dvd")} for g, rs in GROUPS}
for g, rs in GROUPS:                                  # the simplified titles say "on every task": hold them to it
    assert boot(rs, what="diff")[0] > 0, g
WHO = {"q": "Q", "dvd": "DVD"}


def accuracy_bars(ax):
    """grouped bars from 0 per task + all tasks, chance line; the only value labels are the pooled pair"""
    xs, W, OFF = np.array([0, 1.2, 2.4, 3.6, 5.1]), 0.4, 0.24
    for k, off in (("q", -OFF), ("dvd", OFF)):
        ax.bar(xs + off, [ACC[g][k] for g, _ in GROUPS], width=W, color=FILL[k], edgecolor=EDGE[k], linewidth=1.8, zorder=2)
    ax.axhline(0.5, color=INK2, lw=1.6, zorder=3)
    ax.text(1.008, 0.5, "chance", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=16, color=INK2)
    for k, off in (("q", -OFF), ("dvd", OFF)):
        v = ACC["all tasks"][k]
        ax.annotate(f"{v:.0%}", (xs[-1] + off, v), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=17, fontweight="bold", color=INK)
        ax.annotate(WHO[k], (xs[-1] + off, v), xytext=(0, 28), textcoords="offset points", ha="center", va="bottom",
                    fontsize=15, color=INK2)
    ax.axvline((xs[3] + xs[4]) / 2, color=GRID, lw=1.2, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels([g for g, _ in GROUPS], fontsize=16, color=INK)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 0.75); ax.set_yticks([0, 0.25, 0.5, 0.75]); ax.set_yticklabels(["0%", "25%", "50%", "75%"], fontsize=15)
    ax.set_ylabel("picks the better of two actions", fontsize=16, color=INK, labelpad=8)
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.grid(axis="y", color=GRID, lw=1.0, zorder=0); ax.set_axisbelow(True)


def simple_frame(fig, with_example):
    X0 = 0.035
    fig.text(X0, 0.94, "DVD ranks nearby actions better than Q on every task", fontsize=26, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(X0, 0.878, "How often each critic picks the better of two nearby, out-of-distribution actions", fontsize=17,
             color=INK2, ha="left", va="center")
    handles = [Patch(facecolor=FILL[k], edgecolor=EDGE[k], linewidth=1.8, label=LBL[k]) for k in ("q", "dvd")]
    if with_example:
        handles.append(Patch(facecolor=TRUTH, edgecolor="none", label="true success (example)"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(X0 - 0.008, 0.816), ncol=3, frameon=False, fontsize=16,
               handlelength=1.4, handleheight=1.0, columnspacing=2.0, handletextpad=0.6)
    fig.text(X0, 0.03, "Pooled over checkpoints · one training seed per critic · pairs scored only when true success clearly differs",
             fontsize=14, color=INK2, ha="left", va="bottom")


fig = plt.figure(figsize=(13.333, 7.5))                 # the ranking accuracy chart alone
simple_frame(fig, with_example=False)
accuracy_bars(fig.add_axes([0.1, 0.15, 0.8, 0.6]))
save(fig, "fig1_simple_accuracy")

fig = plt.figure(figsize=(13.333, 7.5))                 # the chart plus one example decision (threading)
simple_frame(fig, with_example=True)
accuracy_bars(fig.add_axes([0.1, 0.15, 0.52, 0.6]))
axE = fig.add_axes([0.728, 0.2, 0.22, 0.37])
axE.bar(SHIFTS, ex["succ"] / 16, width=0.075, color=TRUTH, zorder=1)
def in_words(n):
    """rollout outcome of a top pick without a number (the slide keeps numbers to the pooled pair)"""
    words = {16: "all succeed", 15: "almost all succeed", 14: "almost all succeed", 0: "all fail", 1: "almost all fail"}
    assert n in words, n
    return words[n]


for k in ("q", "dvd"):
    v = mm(ex[k])
    axE.plot(SHIFTS, v, color=EDGE[k], lw=2.2, zorder=3, solid_capstyle="round", solid_joinstyle="round")
    axE.scatter(SHIFTS, v, s=30, marker=MARK[k], color=FILL[k], edgecolors=EDGE[k], linewidths=1.2, zorder=4)
    bi = int(np.argmax(v))
    axE.scatter([SHIFTS[bi]], [v[bi]], s=170, marker=MARK[k], color=FILL[k], edgecolors=EDGE[k], linewidths=2.2, zorder=5)
    kw = dict(xytext=(-12, 30), ha="right") if k == "dvd" else dict(xytext=(-6, 30), ha="center")
    axE.annotate(f"{WHO[k]}'s top pick:\n{in_words(int(ex['succ'][bi]))}", (SHIFTS[bi], v[bi]), textcoords="offset points", va="bottom",
                 fontsize=14, color=INK, linespacing=1.15, annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.1, shrinkA=2, shrinkB=9), **kw)
axE.set_xlim(-0.48, 0.48); axE.set_ylim(0, 1.08)
axE.set_xticks([-0.4, 0, 0.4]); axE.set_xticklabels(["−0.4", "executed", "+0.4"], fontsize=14, color=INK2)
axE.set_xlabel("shift of the executed action", fontsize=14, color=INK2)
axE.set_yticks([]); axE.set_ylabel("bars: true success\nlines: score (rescaled)", fontsize=14, color=INK2, labelpad=6)
axE.spines["left"].set_visible(False)
fig.text(0.705, 0.71, "One decision (threading)", fontsize=16, color=INK2, ha="left", va="bottom")
save(fig, "fig1_simple_ood_action_ranking")
