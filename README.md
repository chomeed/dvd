# DVD: world-model value gradients for online policy refinement

Code to reproduce one experiment campaign: online reinforcement learning that improves a frozen flow-matching robot
policy (the *prior*) on four simulated MimicGen tasks (`square_d0`, `square_d1`, `threading_d0`,
`hammer_cleanup_d1`). The method, **DVD**, trains the actor by ascending `gamma^8 * V(f(s, a))`: a value function V,
pretrained on RoboMeter progress labels, applied to the prediction of a DINO world model f. DVD is run inside two RL
algorithms, **DICE-RL** (a residual actor plus best-of-16 selection over prior samples) and **DSRL** (a SAC actor in
the prior's noise space), and compared with a Q-critic baseline of each.

The repository is the minimal production path of a larger research code base: the two trainers, the evaluators,
the exact configuration of all 84 campaign runs, the offline pipeline that made the world models and value
functions, a fetcher for the published assets, and tests that tie this code to the code that produced the results.
It contains no weights or data.

## Quick start

```bash
git clone <this repository> dvd && cd dvd
conda create -n dvd -c conda-forge python=3.12.14 -y && conda activate dvd && pip install --no-deps -r requirements.txt
# four source checkouts at pinned commits under third_party/, and ffmpeg 6.x: environment.md, sections 2 and 3
bash tests/run_all.sh                                             # CPU, no assets needed; must end in "0 fail"
hf auth login                                                     # several asset repositories are private: their owner must grant you read access
bash scripts/fetch_assets.sh --tier train --task square_d0 --skip teacher     # 2.5 GB; --dry-run lists it first
bash scripts/build_pixel_cache.sh square_d0                       # decoded demonstration frames, 46 GB
python tests/configs/test_configs.py                              # test_installed_assets: what a run still misses
G=0 bash scripts/smoke_train.sh dsrl q square_d0                  # a 2000-step version of one run
G=0 bash scripts/run.sh configs/dsrl/ours.sh square_d0 43         # one real run: DSRL + DVD, seed 43, 600k steps
```

How far these lines were checked on the machine that produced the results (the *reference machine*): the two test
commands were run as printed; the fetch, the pixel-cache build, the smoke run and the real run were executed only
with their dry-run switches (`--dry-run`, `--print`, `TRAIN_ONLINE_DRYRUN=1`), which resolve every argument and
start nothing; the clone, the installation and the login were not executed on the reference machine, where
environment and assets were already in place. In particular **no training step of this repository has been run
there yet**: `AGENTS.md`, section 6, is the ladder of checks to climb on a new machine, and says what was and was
not verified.

One run needs 65 to 131 GiB of host RAM by its end, depending on the cell, and a DICE-RL run 13 to 37 GiB of GPU
memory; a 600k-step run takes 8 to 34 hours (`environment.md`, section 1).

## Where to read on

| document | content |
|---|---|
| `AGENTS.md` | the full guide: scope and provenance, the method, a map of every file, the pipeline with exact commands, the verification ladder, reference results, behaviours kept on purpose, known traps, tests, open gaps |
| `environment.md` | installation, exact versions, third-party checkouts, the RoboMeter environment, where every asset comes from and goes, sizes, what was not verified |
| `configs/README.md` | the campaign as a TASK x ARM matrix, and how to run one cell or the whole list |
| `tests/README.md` | what each test proves, the two ways to run the suite, what SKIP means |
| `reference/README.md` | verbatim snapshots of the reference machine (launchers, results table, the commit of the research code) |

Reference results: `reference/results_2026-10-12.md`. Two evaluation protocols are in use (best of 16 for DICE-RL,
a single noise draw for DSRL); name the protocol whenever a number is quoted.
