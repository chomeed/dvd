# OOD action ablation: which critic ranks nearby actions? (weekly meeting, 2026-10-08)

**Question.** If we shift the executed action to nearby, increasingly out-of-distribution actions, which critic still orders them by their true outcome? The two critics are a learned Q(s,a) (DSRL Q) and DVD's γ⁸V(f(s,a)) through the DINO world model (DSRL + DVD).

## Setup
- **Runs.** The s44 DSRL Q run (10-critic ensemble) and the s44 DSRL + DVD run (dpinit V, TD-only). Both are evaluated at the same checkpoints, 175k / 400k / 600k, on hammer, square_d0, square_d1 and threading.
- **States.** 274 states, taken from 5 reference episodes per run and checkpoint. Each episode contributes up to 3 states where the outcome is uncertain or depends on the action.
- **Candidates.** At each state, the executed 8-step chunk is shifted along one action axis by −0.4…+0.4. The shift is applied to all 8 steps, in normalized action units, clipped to [−1, 1], giving 9 candidates.
  - The axes are x, y, z, rot_x, rot_y and rot_z. Gripper shifts never changed the outcome, so that axis is dropped.
- **How out-of-distribution the shifts are.** Distance is measured to 256 prior draws, with 1 = a typical draw. The median distance is 0.87 at the executed chunk and 4.4 at ±0.4.
- **Ground truth.** 16 common-random-number rollouts per candidate. Each rollout continues with the run's own actor.
- **Decisions.** A decision is one (state, axis) sweep. 1,477 decisions are kept: those where the 9 candidates' successes differ by ≥ 5/16.
- **Accuracy.** The fraction of candidate pairs (successes differing by ≥ 5/16) that a critic orders the same way as the true discounted return. Chance is 0.5.
- **Regret.** (best success − success of the critic's top-scored candidate) / 16, averaged over decisions.
- **Confidence intervals.** 95% bootstrap over decisions.

## Figures

**`fig1_ood_action_ranking`** (main slide)
- **A. One hand-picked decision** (threading 175k, z axis, from an episode of the Q run).
  - True success is 16/16 at shifts −0.1 and 0, then 15/16 at +0.1, and 0/16 at +0.3 and +0.4.
  - V(f) peaks at +0.1, where success is 15/16. Q rises steadily and picks +0.4, where success is 0.
  - The pattern is general: across all decisions, the score is monotone in the shift (|Spearman ρ| ≥ 0.9) for Q in 77%, for V(f) in 29%, and for the truth in only 5%.
- **B. Ranking accuracy.**
  - Q is at chance (0.37–0.55). It is significantly below chance in hammer 600k.
  - V(f) is higher in all 12 cells: 0.54–0.66, a gap of +0.05 to +0.20.
  - The gap's CI excludes 0 in 9 cells. In 3 of them it does so barely, with a lower bound of +0.003 to +0.005 (square_d0 600k, square_d1 400k, threading 175k).
  - Pooled over checkpoints, the CI excludes 0 on every task: hammer +0.18, square_d0 +0.09, square_d1 +0.06, threading +0.11.
  - All decisions pooled: Q 0.51 vs V(f) 0.61, a gap of +0.10 [+0.08, +0.12].
- **C. Cost of acting on the top pick.**
  - All 12 points lie below the diagonal: V(f)'s top pick loses less success in every cell. Regret is 0.26–0.37 for V(f) vs 0.39–0.65 for Q; pooled, 0.31 vs 0.49.
  - The gap is largest on threading.

**`fig2_per_axis`** (backup)
- V(f)'s point estimate is ahead on 23 of the 24 task × axis cells. square_d1 x is a tie (+0.001); square_d1 rot_z is the only reversal (−0.02, CI includes 0). The CI excludes 0 on 9 of the 24.
- Largest gaps: hammer x +0.42, hammer y +0.23, square_d1 y +0.22, threading z +0.21 and threading rot_z +0.19.

**`fig3_over_training`** (backup)
- V(f) is above Q at every checkpoint on every task.
- Both critics are lowest at 600k on hammer. On square_d1, both fall from 400k to 600k.

**`fig4_home_distribution`** (backup)
- This answers "Q only lost because it was scored on unfamiliar states".
- On decisions from the Q run's own episodes, Q scores 0.50 and V(f) 0.61: a gap of +0.11 [+0.08, +0.14] over 757 decisions.
- On the DVD run's episodes, it is 0.51 vs 0.60: a gap of +0.09 [+0.06, +0.13] over 720 decisions.

**Simplified versions** (slide-ready, few numbers)
- `fig1_simple_accuracy`: ranking accuracy per task and over all tasks, with a chance line. The only numbers on it are the pooled pair (Q 51%, DVD 61%).
- `fig1_simple_ood_action_ranking`: the same chart plus the threading example decision.
- `fig1_simple_by_task`: the example's view for every task, averaged over all of that task's decisions.
  - Top row: true success at each shift. Bottom row: each critic's score, rescaled to 0–1 within each decision, then averaged.
  - DVD's average peaks at the executed action on every task.
  - Q's average is flat or peaks at the extremes on three tasks, and peaks weakly at 0 on threading.
  - Averaging hides Q's per-decision behaviour: its top pick is ±0.4 in 82% of decisions, but the direction varies.

`numbers.csv` holds the per-cell numbers.

`make_figures.py` regenerates every figure and the CSV in a few seconds from `post_iclr2027/q_on_ood/figs_steps/results_*.pkl`.

## Policy success of the same checkpoints

One noise draw per action chunk (no best-of-N), 100 episodes, env seeds 10000–10099. Numbers are success %, Q / DVD.

| task | 175k | 400k | 600k |
|---|---|---|---|
| hammer | 41 / 38 | 44 / 58 | 45 / 59 |
| square_d0 | 69 / 83 | 74 / 87 | 67 / 92 |
| square_d1 | 41 / 46 | 35 / 53 | 30 / 59 |
| threading | 91 / 78 | 94 / 91 | 89 / 97 |

The DVD policy is not yet ahead on hammer at 175k or on threading at 175k and 400k, although V(f) already ranks better there. By 600k it leads on all four tasks.

## Caveats
- The study measures critic quality, not policy success. It explains why DVD keeps improving where Q stalls; the success table above remains the primary result.
- One seed per arm (s44), at three checkpoints. Both critics come from DSRL runs; DICE-RL's critics were not tested.
- The effects are moderate, and V(f) is far from perfect (accuracy 0.54–0.66). square_d1 is the weakest task.
- "Nearby" means single-axis shifts of up to ±0.4 around the executed chunk. This is local ranking, which is what the actor gradient and best-of-N selection depend on.
