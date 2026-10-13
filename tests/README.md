# tests/

Ten plain Python scripts. Each one prints what it measured and asserts; each check is a `test_*` function and ends
in one line `PASS <name>`, `SKIP <name>: <reason>` or `FAIL <name>`. They run on CPU, write only into temporary
directories and start no training. One of them, `pipeline/test_bon16_parity.py`, needs a GPU and input files and is
described separately below. Test scripts are named relative to `tests/` in this file; every other path is relative
to the repository root.

```bash
bash tests/run_all.sh                    # every script, one summary line each, then the totals
bash tests/run_all.sh rl test_recipes    # only the scripts whose path contains one of these words
python tests/rl/test_legacy.py           # one script on its own (source scripts/env.sh first)
```

`run_all.sh` sources `scripts/env.sh`, sets `PYTHONDONTWRITEBYTECODE=1`, keeps one log per script in a temporary
directory (`TEST_LOG_DIR` to choose it) and exits 0 only if no script failed. The scripts are not driven by pytest
(pytest is not in `requirements.txt`, and collection by it has not been tried).

## What PASS, SKIP and FAIL mean

- **PASS**: the check ran and held.
- **SKIP**: an input the check needs is not on this machine, and the line names it. Nothing was verified. A SKIP
  is expected on a fresh clone and is a to-do on a machine that is about to train: `SKIP test_installed_assets`
  means no asset was found under the roots and nothing was checked, not that the assets are fine.
- **FAIL**: the check ran and did not hold; the traceback is in the log. The script exits non-zero.

Two modes are worth running. Both must end with `0 fail`; they differ in how many checks SKIP.

## Mode 1: fresh clone

Nothing but the main Python environment (`environment.md`, section 2). No assets, no third-party checkouts, no
network.

```bash
bash tests/run_all.sh
```

What runs: the argv of all 32 cells x 2 budgets against the golden files, the DICE-RL configuration resolved key
by key, the legacy rename tables and converter, the guards on the trainers' checkpoint loaders, the readout and
data-rule unit tests on synthetic inputs, the recipe wrappers, the refusal of a missing or wrong `WS_TASK`, and the
hygiene scan. What SKIPs: everything that reads a weight file, a dataset, a reference run, the research tree or the
DSRL fork. Last measured: 42 pass, 28 skip, 0 fail, in about a minute (the totals grow as tests are added;
what must hold is `0 fail`).

## Mode 2: reference machine (or any machine with the assets in place)

Set what the machine has before calling `run_all.sh`; every variable is optional and each one turns SKIPs into
checks.

```bash
export ASSET_ROOT=<assets>                 # or DATA_ROOT / CKPT_ROOT / CACHE_ROOT one by one (scripts/env.sh layout)
export THIRD_PARTY=<checkouts>             # or DSRL_ROOT / MIMICGEN_ROOT / TASK_ZOO_ROOT one by one
export HF_HOME=<hugging face cache> HF_OFFLINE=1
export LIVE_TREE=<research tree>           # the directory holding online_rl_mimicgen/ and 0811/
export LEGACY_RUNS=<dir of DICE-RL runs>:<dir of DSRL runs>
export HYGIENE_EXTRA_WORDS=<root directory name>,<unix user name>
bash tests/run_all.sh
```

Last measured on the reference machine, with `LEGACY_RUNS` naming the two directories of the 69 reference runs:
69 pass, 1 skip (the GPU test), 0 fail, in about four minutes on CPU.

A machine that has the published assets but neither the research tree nor the reference runs sets the first three
lines only. That is the pre-flight to run before the first training run on a new machine.

## Environment variables

| variable | read by | meaning |
|---|---|---|
| `ASSET_ROOT`, `DATA_ROOT`, `CKPT_ROOT`, `CACHE_ROOT` | all | the asset roots of `scripts/env.sh`. Tests look for priors, world models, V directories, score tables, frame-cache metadata and demo datasets below them and SKIP when they are absent |
| `THIRD_PARTY`, `DSRL_ROOT` | `configs/test_configs.py`, `rl/test_legacy.py`, `pipeline/test_eval_tools.py` | the DSRL checkout with its stable-baselines3 fork: without it nothing that imports the DSRL modules can run |
| `MIMICGEN_ROOT`, `TASK_ZOO_ROOT`, `ROBOSUITE_ROOT` | the same three, through the imports of the DSRL modules | the simulator checkouts (`environment.md`, section 3) |
| `HF_HOME`, `HF_OFFLINE` | tests that build a world model | the cache must hold `facebook/dinov2-small` |
| `LIVE_TREE` | `rl/test_source_parity.py`, `rl/test_legacy.py`, `rl/test_cpu_parity.py` | root of the pre-rename research tree this code was cut from (commit in `reference/LIVE_COMMIT`). Only the reference machine has it |
| `LEGACY_RUNS` | `configs/test_configs.py`, `rl/test_legacy.py`, `pipeline/test_eval_tools.py` | `:`-separated directories that hold run directories written by the research code (`<dir>/<run>/config.json`, `checkpoints/`); a run directory itself is accepted |
| `HYGIENE_EXTRA_WORDS` | `pipeline/test_recipes.py` | comma-separated words that must not appear anywhere in the repository outside `reference/`, e.g. the reference machine's root directory name and Unix user name. The test file itself spells no such word |
| `DINO_WM_CKPT` | `pipeline/test_dino_wm.py` | one world-model file to test instead of the first one below `$CKPT_ROOT/wm` |
| `TEST_LOG_DIR`, `PYTHON` | `run_all.sh` | log directory; interpreter |

## What each script covers

| script | always (fresh clone) | with assets / checkouts | with `LIVE_TREE` / `LEGACY_RUNS` |
|---|---|---|---|
| `configs/test_configs.py` | `scripts/run.sh` produces the reference argv token for token for 32 cells x 2 budgets, with the reference run name, wandb group and environment; default `DISTILL` is the paper recipe; bad inputs are refused; the campaign lists are complete; the DICE-RL trainer resolves the argv to the reference configuration key by key, and the dry run prints the same; `scripts/env.sh` keeps to its own checkout; the table of `scripts/fetch_assets.py` targets the paths the configs name; `scripts/smoke_train.sh` launches the campaign cell with only its budget changed | DSRL configuration key by key (`DSRL_ROOT`); `test_installed_assets`: every prior resolves to the reference checkpoint, every world model and V directory a campaign cell names exists, every task has its state scaler; it also lists what is not built yet (pixel caches, dataset videos) | `LEGACY_RUNS`: the `config.json` of every reference run equals what this repository resolves for the same task, arm, seed and budget |
| `rl/test_source_parity.py` | the package holds exactly the ported files; no file but `legacy.py` spells the method's old name; every allow-listed hunk has a reason | | `LIVE_TREE`: the normalised diff of every ported file against the research tree equals the allow-list, hunk for hunk |
| `rl/test_legacy.py` | rename tables cover this tree; `upgrade_config`, `upgrade_checkpoint`, `sidecar`, `remap_asset_path` (a missing explicit override is an error) and the converter CLI on synthetic inputs; `refuse_unconverted_*` and that `train_online --resume`, `--dvd-dynamics-init` and `dsrl_mimicgen --resume` call them before loading anything | | `LIVE_TREE`: the tables equal a mechanical derivation from both trees. `LEGACY_RUNS`: every real `config.json` upgrades and is accepted unchanged in value; stored asset paths resolve under the roots to the same files; one real DICE-RL checkpoint and real DSRL weights load, and the trainer's guard refuses the unconverted DICE-RL checkpoint |
| `rl/test_cpu_parity.py` | | | `LIVE_TREE` + `CKPT_ROOT`: readout, V ensemble and world model give bit-identical fp32 outputs on CPU under both trees, for every V directory and world model found |
| `pipeline/test_pool_mqattn.py` | the pv4 readout: exact reduction to the single-query pool, widths, per-view scopes, guards, blob round trip | every released readout below `$CKPT_ROOT/value` reloads with the shape the RL code expects (4 queries per view, 768 wide) | |
| `pipeline/test_dino_wm.py` | frame-to-context conversion, the mean / meanmax pooling layouts, `dino_wm` imports without the policy stack | load, gradient path, freezing and one encode-predict-pool pass on a production world model (`CKPT_ROOT` or `DINO_WM_CKPT`, and `facebook/dinov2-small` in the cache) | |
| `pipeline/test_recipes.py` | `dinowm/train_wm.sh`, `value/fit_v.sh` and `robometer/serve.sh` print the released recipes; no offline entry point accepts a missing, unknown or contradicted `WS_TASK`; `python -m online_rl_mimicgen.pixel_cache` has a working `--help` and a one-line usage error; hygiene scan of the whole repository outside `reference/` | the printed recipes equal the metadata stored with the released world models and V directories | |
| `pipeline/test_offline_data.py` | context windows, scored rows and their order, the Monte Carlo target, on synthetic data | the same rules on the released frame-cache metadata, score tables and V directories | |
| `pipeline/test_eval_tools.py` | `eval/compare_evals.py`; the prior's text table and as-run defaults (task id string, duplicated frame); `eval/eval_bon16.py` upgrades an old checkpoint, resolves its asset paths, refuses missing or unused overrides, unknown config keys and a depth-2 best-of-N checkpoint | how `eval_dsrl_ckpt` finds `config.json` for a linked checkpoint and with `--config` (needs the DSRL fork and the simulator imports) | `LEGACY_RUNS`: the best-of-16 scorer of one real DVD checkpoint and one real Q checkpoint is built on CPU |
| `pipeline/test_bon16_parity.py` | prints SKIP (see below) | | |

## The GPU checks, which `run_all.sh` does not run

`pipeline/test_bon16_parity.py` proves that `eval/eval_bon16.py` assembles the trainer's scorer correctly, on
decisions recorded by the evaluator itself. It needs one GPU (the recorded scores are bf16 GPU values), a few GiB
of memory, the task's prior, world model and V directory, and a DICE-RL checkpoint:

```bash
G=0 bash -c 'source scripts/env.sh
  python eval/eval_bon16.py --ckpt <dvd run>/checkpoints/<step>.pt --out /tmp/e.json --episodes 2 --dump /tmp/d.pt
  python tests/pipeline/test_bon16_parity.py --dump /tmp/d.pt [--q-ckpt <q run of the same task>/checkpoints/<step>.pt]'
```

The end-to-end check of an evaluator is a replay: evaluate a checkpoint whose result file exists and compare episode
by episode. Episodes are seeded individually, so the first N episodes of a replay must equal the first N of the
stored 100.

```bash
G=0 bash -c 'source scripts/env.sh
  python eval/eval_bon16.py --ckpt <run>/checkpoints/<step>.pt --out /tmp/new.json --episodes 10
  python eval/compare_evals.py <stored result>.json /tmp/new.json --require-identical'
```

`compare_evals.py` reads best-of-16, DSRL and prior result files and pairs them on the env seeds they share.

## What the suite does not cover

- No training step is executed by any test: the update loops of `train_online` and `dsrl_dino`, `torch.compile`,
  the in-run evaluation subprocess of `dsrl_dino` and the RoboMeter spool exchange are covered by source parity,
  resolved-configuration equality and CPU parity of the building blocks only.
- CPU parity is fp32 on one thread. The campaign runs the world model under bf16 autocast on a GPU; that path is
  exercised by the GPU checks above, for inference only.
- The guards on `--resume` and `--dvd-dynamics-init` are tested as functions and by reading the source for the call
  sites; the loaders themselves sit inside `train()`, which needs a GPU, a simulator and the demo cache.

## Keeping the gates honest

- **Any edit to a file under `rl/online_rl_mimicgen/`** shows up in the test script `rl/test_source_parity.py` as
  an unexplained hunk. Regenerate the allow-list on a machine with the research tree and write the reason of every new entry:

  ```bash
  LIVE_TREE=<research tree> python tests/rl/test_source_parity.py --show     # read the hunks
  LIVE_TREE=<research tree> python tests/rl/test_source_parity.py --update   # new entries get "TODO: explain this edit"
  ```

  `--update` keeps the reasons of unchanged hunks; `test_allowlist_reasons` fails while a `TODO` is left. Comments
  and docstrings are not compared, argparse help strings are. A hunk that touches an entry already in the list
  replaces it, so its reason has to be written again.
- **The method's old name** may be spelled in `rl/online_rl_mimicgen/legacy.py` only. A trainer that has to
  recognise an old artifact calls a function of that module.
- **The golden files** under `tests/configs/golden/` are the output of `tests/configs/make_golden.py` on the
  reference machine and are not edited by hand.
- **Hygiene.** `pipeline/test_recipes.py` scans every text file outside `reference/`, `.git` and the asset and
  output directories for home-directory paths, e-mail addresses, token shapes and for any absolute or `~/` path
  that exists on the machine running the test, plus the words of `HYGIENE_EXTRA_WORDS`. It scans itself.
