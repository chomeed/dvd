# configs/ : the campaign as TASK x ARM

One file per arm, parametrised by task, seed and budget:

```bash
bash scripts/run.sh configs/<method>/<arm>.sh TASK SEED [STEPS]        # STEPS defaults to 600000
TRAIN_ONLINE_DRYRUN=1 bash scripts/run.sh ...                          # print the resolved configuration, start nothing
bash scripts/run_campaign.sh configs/campaign/<method>.txt GPU [--list]  # a list of runs, one after another
```

Before the first run on a machine (`environment.md`, section 5):

```bash
bash scripts/fetch_assets.sh --tier train --task TASK   # published assets to the paths of tasks.sh (--dry-run lists them)
bash scripts/build_pixel_cache.sh TASK                  # the demo pixel cache, 46-86 GB
python tests/configs/test_configs.py                    # test_installed_assets: what a campaign cell still misses
bash scripts/smoke_train.sh <method> <arm> TASK         # the cell on a 2000-step budget: collection, updates, checkpoint, eval
```

`TASK` is `square_d0`, `square_d1`, `threading_d0` or `hammer_cleanup_d1`. The campaign is every cell with seeds 43
and 44 at 600k env steps and seed 45 at 1M: 36 DICE-RL runs (`campaign/dice_rl.txt`) and 48 DSRL runs
(`campaign/dsrl.txt`). All 48 DSRL runs are finished; the 12 DICE-RL runs with seed 45 had not been launched on
2026-10-13.

| file | arm | V online | world model / V loaded | reference launcher `ARM=` | ran on |
|---|---|---|---|---|---|
| `dice_rl/dvd.sh` | DICE-RL + DVD | `DISTILL=tdonly`: TD | continuation WM (hammer: base WM), `_pv4` V | `dvd` (square_d1, threading_d0, hammer), `dvdtd` (square_d0) | all 4 tasks |
| `dice_rl/dvd.sh` | DICE-RL + DVD | `DISTILL=tddistill`: TD + RoboMeter distillation | same | `dvd` (square_d0), `dvdd` (the other three) | all 4 tasks |
| `dice_rl/q.sh` | DICE-RL + pixel Q ensemble | - | none | `q` | all 4 tasks |
| `dsrl/ours.sh` | DSRL + DVD | TD | continuation WM; `_pv4` V (square_d0, hammer) or `_pv4_seg` V (square_d1, threading_d0) | `ours` | all 4 tasks |
| `dsrl/oursdist.sh` | DSRL + DVD | TD + RoboMeter distillation | as `ours` | `oursdist` | square_d0, hammer |
| `dsrl/oursmc.sh` | DSRL + DVD, Monte Carlo initialised V | TD | continuation WM; `v_dino_mc_return_*` V | `oursmc` | square_d0, square_d1, hammer |
| `dsrl/randv.sh` | DSRL + DVD, random V (same V for every seed) | TD | continuation WM; architecture and scaler of the `ours` V | `randv` | square_d0, square_d1, hammer |
| `dsrl/q.sh` | DSRL with 10 + 10 Q critics | - | none | `q` | all 4 tasks |

Without `DISTILL`, `dice_rl/dvd.sh` runs the paper recipe of the task: TD + distill on square_d0, TD only on the
other three. The other value is the distill A/B twin.

- `tasks.sh` is the per-task asset table (prior, world-model tags, V directories, spool, demo dataset, pixel
  cache); every arm file sources it, and every path in it comes from the roots of `scripts/env.sh`.
  `scripts/fetch_assets.py` downloads to exactly these paths; the test checks the two against each other.
- `dsrl/_common.sh` holds what the five DSRL arms share.
- The argv of each cell is the argv of the reference launchers (`reference/launchers/`), token for token, with the
  method's flags renamed `svg` -> `dvd` and the paths taken from the roots. Run names and wandb groups are the
  reference ones with the same rename. `tests/configs/test_configs.py` checks all of it against the reference
  launchers' own output, and against the `config.json` of the reference runs when those are at hand.
- Evaluation protocols differ by method and must be labelled: DICE-RL checkpoints (every 50k steps) are scored
  afterwards with the best of 16 prior draws (`scripts/eval_loop.sh`); DSRL checkpoints (every 25k steps) are scored
  during the run with one noise draw per chunk. Both use 100 episodes on env seeds 10000-10099.
- TD + distill cells need the task's RoboMeter teacher running (`scripts/robometer_teacher.sh start TASK`);
  `scripts/run_campaign.sh` skips them while no teacher answers.
- Two behaviours of the frozen prior are part of every reference number and are not configurable here: it is
  conditioned on the task id string (not the instruction sentence), and on a duplicated current frame (its
  observation history is reset before every chunk).
