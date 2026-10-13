# AGENTS.md

A guide to this repository for a coding agent or engineer who has never seen the project: what the code does, where
everything is, how to reproduce the results on another machine, and which traps are already known.

The repository holds the code of one experiment campaign: online reinforcement learning (RL) that improves a frozen
robot policy on four simulated MimicGen manipulation tasks. The method under test is called **DVD**: the actor is
trained by ascending `gamma^8 * V(f(s, a))`, a value function V applied to what a world model f predicts the camera
images will look like after the next 8 actions. DVD is run inside two RL algorithms, **DICE-RL** and **DSRL**, and
compared with a Q-critic baseline of each. What is here is the minimal code that produced those results, the exact
configuration of every run, the tools that fetch the inputs, and the tests that tie this code to the code that ran.

Other documents, each authoritative for its subject: `environment.md` (installation, versions, asset layout, sizes),
`configs/README.md` (the run matrix), `tests/README.md` (the tests), `reference/README.md` (provenance snapshots),
`robometer/finetune/NOTES.md` (the RoboMeter teacher). Every script outside `rl/` starts with a header that is its
full manual; for the entry points under `rl/` use `--help` and the files under `configs/`.

## Contents

1. [Start here](#1-start-here): the short version, and the conventions of this file
2. [Scope and provenance](#2-scope-and-provenance)
3. [The method in one page](#3-the-method-in-one-page)
4. [Repository map](#4-repository-map)
5. [End-to-end pipeline](#5-end-to-end-pipeline)
6. [Verification ladder for a new machine](#6-verification-ladder-for-a-new-machine)
7. [Reference results](#7-reference-results)
8. [Behaviours kept on purpose](#8-behaviours-kept-on-purpose)
9. [Traps](#9-traps)
10. [Tests and gates](#10-tests-and-gates)
11. [Known gaps and next steps](#11-known-gaps-and-next-steps)
12. [Glossary](#12-glossary)

## 1. Start here

**The short version**, for a reader with one minute:

- Everything takes its paths from `scripts/env.sh`. Nothing else in the repository holds a path.
- One run is `bash scripts/run.sh configs/<method>/<arm>.sh TASK SEED [STEPS]`. That command trains for real. Put
  `TRAIN_ONLINE_DRYRUN=1` in front to see the resolved configuration and start nothing.
- No training step of this repository has been executed on the machine that produced the results. The code is tied
  to the code that ran by three gates (source, configuration, CPU numerics; section 10) and by replayed evaluations.
  On a new machine climb the ladder of section 6 before trusting a number.
- The two methods are evaluated with different protocols (best of 16 against a single draw). Never mix them in one
  comparison, and compare means over the 350k-600k window, not single evaluations (section 7).
- Several things that look like mistakes are kept on purpose because every reference number depends on them
  (section 8).
- Only the flags used under `configs/` are supported; the trainers' parsers are much larger (section 2.5).
- A change under `rl/` must be recorded in the source-parity allow-list, with its reason (section 10).

**Words.** The *reference machine* is the one machine on which every result quoted here was produced. The *research
tree* is the larger, older code base the campaign was launched from; this repository is cut from it. A *method* is
one of the two RL algorithms; an *arm* is one variant of a method (DVD, DVD with a randomly initialised V, the Q
baseline, ...), and each arm is one file under `configs/`. A *cell* is one combination of method, arm and task (for
example DSRL + DVD on square_d0); a *run* is a cell with a seed and a step budget. Where this document counts **32
cells**, the two ways DICE-RL DVD trains V online (TD only and TD + distill, the `DISTILL` switch of
`configs/dice_rl/dvd.sh`) are counted separately on each task: 12 DICE-RL + 20 DSRL. *The paper* is the ICLR 2027
submission that reports the method. The *campaign* is the set of 84
runs listed in `configs/campaign/`. Section 12 is a glossary of the short names that appear in file and run names
(pv4, dpinit, bon16, ...).

**Where commands run.** Every command is typed in the repository root, in a shell where the main Python environment
is active (`python` is taken from `PATH`). The shell wrappers (`scripts/*.sh`, `dinowm/train_wm.sh`,
`value/fit_v.sh`, `robometer/serve.sh`, `tests/run_all.sh`) source `scripts/env.sh` themselves. Before a bare
`python ...` command, run `source scripts/env.sh` once in that shell; to pin a GPU, `export G=<index>` before that.
Text in `<angle brackets>` is a placeholder.

**How far each command was checked.** Each command below carries one of three tags. They describe the state on the
reference machine on 2026-10-13, when its memory and disk were committed to the running campaign.

| tag | meaning |
|---|---|
| `[run]` | executed as printed (with real paths in place of placeholders); CPU only |
| `[dry]` | executed only in a form that changes nothing: with `TRAIN_ONLINE_DRYRUN=1` or `RUN_PRINT_ARGV=1` in front, or with `--dry-run`, `--print`, `--list` or `--help` added. Arguments, paths and the configuration were resolved; nothing was downloaded, built, trained or evaluated |
| `[not executed]` | not executed on the reference machine. The first run on another machine is the test |

## 2. Scope and provenance

### 2.1 What is included

The minimal production path: what a campaign run, its evaluation, or the regeneration of its inputs executes.

| part | where |
|---|---|
| the two RL trainers and their shared modules | `rl/online_rl_mimicgen/` |
| the simulator wrapper | `envs/lerobot_env_mimicgen/` |
| evaluators | `eval/`, `rl/online_rl_mimicgen/baselines/eval_dsrl_ckpt.py` |
| the campaign as configuration: 7 arm files, one per-task asset table, 2 run lists | `configs/` |
| environment file, launchers, asset fetcher, results table | `scripts/` |
| offline pipeline that made the world models and value functions | `dinowm/`, `robometer/`, `value/` |
| tests | `tests/` |
| verbatim provenance snapshots of the reference machine | `reference/` |

It is about 145 text files and under 3 MB. Weights, datasets, caches and run outputs are never committed
(`.gitignore`); they live under the asset roots of `scripts/env.sh`.

### 2.2 What is deliberately left out

- **Research-only code.** Other world models, readouts and RL methods of the research tree, and its analysis
  scripts. The modules `seghead`, `latent56`, `flowdpg`, `fixed_goal_env`, `jitter_env`, `insertion_env`,
  `grad_probe`, `reward_model` and the `mbpo` package are not shipped (section 2.5 says what that means for flags).
- **Machine-specific operations.** The reference machine ran the campaign through launch queues gated on memory and
  disk, evaluation daemons, checkpoint pruning guards and alert loops. None of that is here. Three plain tools
  replace it: `scripts/run_campaign.sh` (one run after another on one GPU, no gates), `scripts/eval_loop.sh` and
  `scripts/results_table.py`. Nothing in this repository watches memory, deletes a checkpoint or stops a process.
- **The critic study.** The study behind the claim that `gamma^8 V(f(s, a))` ranks nearby out-of-distribution actions
  better than `Q(s, a)` is kept verbatim under `reference/critic_ood_study/` and is not ported: its scripts use the
  old names and the reference machine's run directories. `reference/README.md` says what porting it would take.
- **Training of the frozen priors.** The priors are inputs (each checkpoint carries its own `train_config.json`).
  Fine-tuning a RoboMeter teacher is optional and documented in `robometer/finetune/NOTES.md`.
- **Pruning of `train_online.py`.** Deferred on purpose (section 2.5).

### 2.3 Which research code this is

- `reference/LIVE_COMMIT` names the commit of the research tree that every campaign run was launched from
  (`af8ba3aba926d54df4164ad4bdcc5ac9accf302d`, 2026-09-30).
- `rl/online_rl_mimicgen/` is that commit's package with three mechanical changes and one new file: comments and
  most docstrings removed (inherited from the anonymised paper supplement this tree was cut from), the method's old name
  `svg` spelled `dvd`, hard-coded machine paths replaced by the roots of `scripts/env.sh`, and the new `legacy.py`.
  **Expect almost no comments under `rl/`.** For the reasoning behind a line, read the research repository at that
  commit; the layouts of the simulator state are documented in
  `envs/lerobot_env_mimicgen/src/lerobot_env_mimicgen/environment.py`.
- Every remaining difference is enumerated and explained. `tests/rl/source_parity_allowlist.json` lists 70 hunks
  (contiguous blocks of differing lines) in 14 of the 23 ported files, each with its reason; the other 9 files are
  identical once comments, docstrings and the rename are ignored (`tests/rl/test_source_parity.py`, last run
  2026-10-13). The hunks are: asset-root defaults,
  package imports, message and help text, removed branches of unshipped modules, the legacy hooks of section 2.4
  (including that the DSRL evaluator reads only the weights of a checkpoint), and two small fixes (a `%` in a help
  string that crashed `--help`; DSRL no longer demands a RoboMeter spool path when `--no-value-distill` is given).
- `configs/` reproduces the argv of the reference launchers (`reference/launchers/`) token for token, with the same
  rename and the paths taken from the roots: 32 cells at 2 budgets, 5728 argv tokens; the trainers resolve them to the
  reference configuration on all 301 keys (DICE-RL) and 92 keys (DSRL) (`tests/configs/test_configs.py`).

### 2.4 Naming: `svg` in the research code and its artifacts, `dvd` here

The research code called the method SVG. Everything it wrote to disk carries that spelling: the checkpoints and
`config.json` files of the reference runs, their metric names and their wandb runs.

| kind | research code and reference artifacts | this repository |
|---|---|---|
| modules | `0811/svg_term.py`, `dino_svg.py` | `dvd_term.py`, `dino_dvd.py` |
| classes | `SvgTerm`, `DSRLSVG`, `DSRLSVGDino` | `DvdTerm`, `DSRLDVD`, `DSRLDVDDino` |
| flags and config keys | `--svg-*` / `svg_*`, `bon_svg_until`, `qsvg_combine`, `qsvg_normalise` (53 DICE-RL config fields, 15 DSRL arguments) | `--dvd-*` / `dvd_*`, `bon_dvd_until`, `qdvd_combine`, `qdvd_normalise` |
| option values | `--bon-source svg`, `--bc-filter-source svg`, `--actor-gradient-source-normal svg1` | `dvd`, `dvd`, `dvd1` |
| DICE-RL checkpoint | `extra["svg_dynamics"]`, `extra["opt"]["svg"]` | `extra["dvd_dynamics"]`, `extra["opt"]["dvd"]` |
| DSRL run | method tag `dsrl_na_svg_dino`, file `<step>_svg.pt` beside `<step>.zip` | `dsrl_na_dvd_dino`, `<step>_dvd.pt` |
| metric names | `svg_value`, `svg_scale`, `svg_feat_mse`, ... | `dvd_value`, `dvd_scale`, `dvd_feat_mse`, ... (old logs are not converted) |
| run names | `dsrl-na-svg1.0_..._svgrowsall_...`; DICE-RL names already say `dicerl-dvd_...` | `dsrl-na-dvd1.0_..._dvdrowsall_...`; DICE-RL names unchanged |

**The legacy layer.** `rl/online_rl_mimicgen/legacy.py` is the one module that knows the old names; a test fails if
any other file under `rl/` spells them. It also handles the second thing old artifacts carry: absolute asset paths of
the training machine. A stored path is used as it is when it exists; otherwise the same prior, V directory or world
model is looked up under `$CKPT_ROOT/priors`, `$CKPT_ROOT/value`, `$CKPT_ROOT/wm`; an explicit override always wins.

- **Readers of finished runs upgrade on the fly**, so reference runs can be read as they are: `eval/eval_bon16.py`,
  `baselines/eval_dsrl_ckpt.py` (through `scripts/eval_dsrl.sh`), `common.load_agent_checkpoint`,
  `scripts/results_table.py`, `scripts/wandb_eval_mirror.py`.
- **The trainers do not.** They read the new names only. `train_online --resume <old checkpoint>` and
  `--dvd-dynamics-init <old checkpoint>` stop with an error that names the old keys and the converter. Without that
  guard the fine-tuned world model and its optimiser would be skipped silently and training would continue from
  the pretrained world model. `--value-init`, `--encoder-init`, `--init-actor-from` and `--init-critic-from` read
  containers whose names never changed and accept old and converted checkpoints alike. The DSRL trainer
  `baselines/dsrl_dino.py` has no `--resume` at all (its replay is not checkpointed).
- **The converter:**

  ```bash
  python -m online_rl_mimicgen.legacy checkpoint <old.pt> <new.pt>     # [run] on a synthetic checkpoint; never overwrites
  python -m online_rl_mimicgen.legacy config <config.json>            # [run] prints the upgraded config
  ```

- An old DSRL `.zip` cannot be opened with stable-baselines3's own `.load()`: its pickled class data names a module
  that no longer exists. `eval_dsrl_ckpt` reads the weights only, which is all an evaluation needs.

### 2.5 Supported surface of the trainers

Only the flags that the files under `configs/` pass are supported and tested: 53 of the 296 options of
`train_online.py` and 63 of the 91 DSRL options. The rest of the two parsers is kept so that the files stay
comparable, line by line, with the research code. Some of those options lead to code that is not shipped: the
parser refuses the segmentation-head and latent world models (`--dvd-dynamics seghead_wm`, `latent56`); state-space
dynamics checkpoints for `--dvd-dynamics <stem>` do not exist here; `--method flowdpg`, the learned reward model,
the fixed-goal, jitter and insertion environments and the gradient probe fail at run time with `ImportError` or
`NameError`. What works is `--method dice_rl` with pixel observations and `--dvd-dynamics dino_wm`, and the DSRL
trainer. Help strings still describe the unshipped options.

Cutting `train_online.py` (8656 lines) down to what the campaign reaches is a separate step that has not been done.
Any such cut must keep the gates of section 10 green and should be followed by a seeded GPU comparison.

## 3. The method in one page

**Tasks.** Four MimicGen tasks in the robosuite simulator, controlled at 20 Hz and observed by two 224 x 224 cameras
(`agentview`, `eye_in_hand`): `square_d0` and `square_d1` (put a square nut on a peg; d1 randomises the peg position),
`threading_d0` (thread a needle through a tripod), `hammer_cleanup_d1` (open a drawer, put a hammer in, close it).
An episode ends at success or after 400 primitive steps (500 for hammer). The reward is sparse: 1 on success, else 0.

**The frozen prior.** Per task, one language-conditioned flow-matching DiT (diffusion transformer) policy:
LeRobot's built-in `multi_task_dit` policy with CLIP ViT-B/16 image and text encoders. It sees 2 frames of both
cameras and a 9-D proprioceptive state (end-effector position 3, quaternion 4, gripper 2) and plans 32 actions. The RL
code samples a plan with 10 Euler integration steps and executes 8 actions of it as one *chunk*, then replans. One
decision is therefore one chunk of 8 primitive steps. The prior's weights never change; everything below learns
around it.

**World model f.** A frozen DINOv2-S/14 image encoder turns each camera image (resized to 196 px) into 196 patch
features of width 384. A transformer predictor (6 layers, 6 heads, 14.4 M parameters) takes the patch features of
two context frames (8 steps ago and now), the 9-D proprio and the 8 x 7 action chunk, and predicts the patch
features of the frame 8 steps ahead as absolute features (not as a difference). A convolutional encoder turns the
chunk into one token; a small "proprio head" predicts the change of the 9-D proprio. The tag `ph_conv` in file names
means exactly this: proprio head, conv action encoder. Each task's model was trained offline on 160 of the
task's 200 prior rollouts (episodes 0-79 of the 100 successes and of the 100 failures; episodes 80-99 of each are
held out, for the V fit too) and then continued for the same number of steps (`..._cont<steps>`). In the DVD
cells the predictor is also fine-tuned online on the replay.

**Value function V.** V reads `[readout(patch features), proprio]`. The readout, called **pv4** (per-view, 4
queries), is an attention pool: for each camera 4 learned queries each take a softmax-weighted mean of that camera's
196 patch vectors; the 8 vectors are concatenated and projected to 768 numbers, so V's input has 768 + 9 = 777. V
itself is an ensemble of 3 small MLPs. It is pretrained offline by regression on **RoboMeter** labels: RoboMeter is
a video-language model that, given a clip and a task sentence, returns a task-progress score (and a success
probability) for every frame; here it is fine-tuned per task with a LoRA and called the *teacher*. Half of the
training rows are real frames and half are frames predicted by f, so that V is fitted on f's outputs too. Online, V
is trained by temporal-difference (TD) learning on the replay, and in some cells additionally regressed on fresh
RoboMeter scores of the policy's own episodes ("TD + distill").

**The DVD term** is `-gamma^8 V(f(s, a))`, averaged over the batch and divided by the batch mean of its absolute
value. The gradient flows from V through f into the action chunk `a`. No Q critic is involved. It is not the whole
actor loss: DICE-RL adds `100 x mean(residual^2)` on every row (`bc_loss_weight`; it pulls the corrected chunk
toward the prior's, in the DVD and the Q arm alike), and DSRL adds SAC's entropy term.

**DICE-RL** (`rl/online_rl_mimicgen/train_online.py`, `--method dice_rl`). A residual actor adds a correction to the
prior's chunk. At every decision 16 chunks are drawn from the prior, each is corrected, the 16 candidates are ranked
by a critic and the best is executed ("best of 16"). In the DVD arm the critic is `gamma^8 V(f(s, a))` and the actor
is trained with the DVD term; in the Q baseline the critic is the minimum over an ensemble of 10 pixel Q networks,
which also trains the actor. During training the probability of using best-of-16 ramps from 0 to 1 over the first
75k steps, and the share of demonstrations in each batch falls from 1.0 to 0.1 over the first 40k steps.

**DSRL** (`rl/online_rl_mimicgen/baselines/dsrl_dino.py`). A SAC actor does not output actions; it outputs the noise
the prior's flow starts from (56 numbers: the 8 executed steps x 7 action dimensions; the rest of the 32-step plan is
resampled). The prior turns noise into actions. In the DVD arms the noise actor ascends `gamma^8 V(f(s, prior(s,
w)))` through f and through the frozen prior, again without a Q critic. The Q baseline is the published DSRL-NA algorithm (10
critics over actions and 10 over noise) plus this code base's additions: demonstrations in the replay (their share
of a batch falls from 1.0 to 0.1 over 40k steps), the 75k-step exploration ramp, an image actor, policy delay 2,
and an actor Q term divided by its batch mean absolute value. DSRL builds on a fork of stable-baselines3.

**Values live in [0, 1].** The discount is 0.99 per primitive step, so 0.99^8 = 0.92 per chunk. A timeout is stored
as a terminal transition (`done = 1`) for V and for Q. With a 0/1 reward every attainable value is therefore between
0 and 1. A V or Q whose mean is negative, or far above 1, is a bug signal, not a calibration problem.

## 4. Repository map

Entry points are marked with (E). Outside `rl/`, every script's header documents its arguments and environment
variables; the entry points under `rl/` document theirs through `--help`.

| path | role |
|---|---|
| `README.md`, `AGENTS.md`, `environment.md` | overview; this guide; installation, versions, asset layout, sizes |
| `requirements.txt` | the complete package set of the reference environment (install with `--no-deps`) |
| **`scripts/`** | |
| `scripts/env.sh` | the only place paths are set; sourced by everything. Its header lists every variable |
| `scripts/run.sh` (E) | run ONE cell in the foreground: `bash scripts/run.sh configs/<method>/<arm>.sh TASK SEED [STEPS]` |
| `scripts/run_campaign.sh` (E) | run a list of cells on one GPU, one after another; `--list` shows what would run |
| `scripts/smoke_train.sh` (E) | a 2000-step version of one cell: collection, updates, a checkpoint, an evaluation |
| `scripts/eval_loop.sh` (E) | best-of-16 evaluation of every saved DICE-RL checkpoint |
| `scripts/eval_dsrl.sh` (E) | evaluate one DSRL checkpoint from any working directory |
| `scripts/results_table.py` (E) | success-rate table per cell and seed, with the protocol named |
| `scripts/wandb_eval_mirror.py` (E) | copy DICE-RL evaluation results to wandb as companion runs |
| `scripts/fetch_assets.py`, `scripts/fetch_assets.sh` (E) | download the published assets to the paths `configs/tasks.sh` names |
| `scripts/make_square_d1_statefix.py` (E) | build the square_d1 prior the runs load from the published one |
| `scripts/build_pixel_cache.sh` (E) | build a task's demo pixel cache |
| `scripts/robometer_teacher.sh` (E) | `start`, `check` or `show` the RoboMeter teacher of a task |
| `scripts/resolve_config.py` | print the configuration a trainer resolves from an argv (the dry run of DSRL) |
| `scripts/_tree_guard.sh` | makes the launchers exit 3 if `python` would import the RL package from another checkout |
| **`configs/`** | |
| `configs/tasks.sh` | the per-task asset table: prior, world-model tags, V directories, spool, demo dataset |
| `configs/dice_rl/dvd.sh`, `configs/dice_rl/q.sh` | the two DICE-RL arms |
| `configs/dsrl/ours.sh`, `oursdist.sh`, `oursmc.sh`, `randv.sh`, `q.sh`, `_common.sh` | the five DSRL arms and what they share |
| `configs/campaign/dice_rl.txt`, `configs/campaign/dsrl.txt` | the 36 + 48 runs of the campaign, in launch order |
| **`rl/online_rl_mimicgen/`** | the RL package (import name `online_rl_mimicgen`) |
| `train_online.py` (E) | the DICE-RL trainer: parser, collection loop, updates, checkpoints |
| `common.py` | task table, `FrozenPolicyPrior`, replay buffer, state scaler, residual actor, critics, `TrainConfig` |
| `dvd_term.py`, `dino_dvd.py` | the DVD term; the glue that makes the DINO world model its dynamics f |
| `dino_wm.py`, `dino_wm_any.py` | the world model (encoder, predictor, loading); a loader for world models on other image encoders (not a campaign path) |
| `value_model.py`, `pool_mqattn.py` | V ensemble and its loading; the multi-query attention readout (pv4) |
| `robometer_distill.py` | trainer side of online RoboMeter distillation (talks to the teacher through a spool directory) |
| `pixel.py`, `expert.py` | IMPALA image encoder for actors and pixel critics; the demonstration buffer |
| `pixel_cache.py` (E) | decode a demo dataset once into the pixel cache (use `scripts/build_pixel_cache.sh`) |
| `record_rollouts.py`, `aggregate_rollouts.py` (E) | record rollouts of the prior as LeRobot datasets (the released ones were made this way) |
| `legacy.py` (E) | old names and old paths (section 2.4); converter command line |
| `critical.py`, `gradvar.py` | imported by `train_online.py`, idle in every campaign cell; kept for source parity |
| `baselines/dsrl_dino.py` (E) | the DSRL trainer of all five DSRL arms, with in-run evaluation |
| `baselines/dsrl_mimicgen.py` | DSRL building blocks: noise-space prior, environment wrapper, parser; finds the stable-baselines3 fork |
| `baselines/dsrl_demos.py`, `baselines/build_demo_store.py` (E) | demonstrations in the DSRL replay; build their store ahead of a run |
| `baselines/eval_dsrl_ckpt.py` (E) | single-draw evaluation of a DSRL checkpoint |
| **`envs/lerobot_env_mimicgen/`** | gym wrapper around the MimicGen tasks; found through `PYTHONPATH`, not pip-installed |
| **`eval/`** | |
| `eval/eval_bon16.py` (E) | the DICE-RL evaluation protocol: best of 16, 100 episodes |
| `eval/eval_prior.py` (E) | the frozen prior alone; also measures the two input quirks of section 8 |
| `eval/compare_evals.py` (E) | paired comparison of two result files; `--require-identical` for replays |
| **`dinowm/`** | offline: world model |
| `dinowm/build_cache.py` (E) | decode a task's prior rollouts into the frame cache |
| `dinowm/train_wm.sh` (E), `dinowm/train_wm_ph.py` | train the base or the continued world model with the released recipe |
| `dinowm/common_ws.py` | shared data layer of the offline pipeline; reads `WS_TASK` |
| **`robometer/`** | offline and online: the teacher |
| `robometer/serve.sh` (E), `robometer/robometer_server.py`, `robometer/teachers.sh` | start a teacher; the server (runs in RoboMeter's own environment); the table of the five teachers |
| `robometer/score_rollouts.py` (E) | score the prior rollouts with a teacher (V's offline targets) |
| `robometer/env/uv.lock` | lock file of the reference RoboMeter environment |
| `robometer/finetune/` | optional LoRA fine-tuning: `train_lora.sh`, `make_rbm_from_seg.py`, `segment_square.py`, `segment_hammer.py`, `NOTES.md` |
| **`value/`** | offline: V |
| `value/fit_v.sh` (E), `value/fit_v_mqattn.py` | fit a V directory with one of the three released recipes |
| **`tests/`** | `run_all.sh` (E) and ten test scripts in `configs/`, `rl/`, `pipeline/` (section 10) |
| `tests/configs/golden/` | the resolved configuration of all 32 cells, generated from the reference code; never edited by hand |
| `tests/rl/source_parity_allowlist.json` | every difference between `rl/` and the research tree, with its reason |
| **`reference/`** | verbatim snapshots: `LIVE_COMMIT`, `launchers/`, `eval_env.sh`, `results_2026-10-12.md`, `critic_ood_study/`. Never imported or executed; the only place with the reference machine's paths and the old names |

**A run directory** (`$RUNS_ROOT/<run name>/`) holds `config.json`, `metrics.jsonl` (rows with `kind` =
`train`, `episode`, `eval` or `rm_distill`), `checkpoints/` and `evals/`. DICE-RL saves `<step>.pt` every 50k steps;
DSRL saves `<step>.zip` plus `<step>_dvd.pt` every 25k steps. File names are the zero-padded step count actually
reached, not the round number (`600000.pt`, `100011.pt`; DSRL `0600002.zip` with `0600002_dvd.pt`). DICE-RL also
writes `000000.pt` at start: it holds no trained module and the evaluator refuses it. An evaluation result is
`evals/<step>.json` with `success_rate` in percent and the per-episode outcomes. `code_version.json` and `wandb/`
sit in the run directory too.

## 5. End-to-end pipeline

In order. Step 5.5 is optional: the released world models and V directories are downloaded in step 5.2.

### 5.1 Environment and third-party checkouts

`environment.md` sections 1 to 4 are the full account (reference hardware, versions, why `--no-deps`, the EGL
rendering note, the one patched robosuite line). In short:

```bash
conda create -n dvd -c conda-forge python=3.12.14 -y && conda activate dvd       # [not executed] any 3.12 works
pip install --no-deps -r requirements.txt                                        # [not executed]
sudo apt-get install -y ffmpeg                                                   # [not executed] ffmpeg 6.x
mkdir -p third_party && cd third_party                                           # [not executed], as are the next 6 lines
git clone https://github.com/NVlabs/mimicgen                    && git -C mimicgen checkout 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo && git -C robosuite-task-zoo checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
git clone https://github.com/ajwagen/dsrl                        && git -C dsrl checkout 8d21b9cf55459f022c0e769bca1343edd9d30e5a
git -C dsrl -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init stable-baselines3
git clone https://github.com/robometer/robometer                 && git -C robometer checkout 352d160389daa964788de1ec933d1925f3a6de4f
cd ..
```

None of the checkouts is pip-installed; `scripts/env.sh` puts them on `PYTHONPATH` or hands their location to the
code. The reference machine has the same environment and the same checkouts already in place, which is why the
block above was never executed there. Then check the installation, on CPU and without any asset:

```bash
source scripts/env.sh                                                            # [run]
python -c "import torch, numpy, lerobot, robosuite, mimicgen, robosuite_task_zoo, lerobot_env_mimicgen
from online_rl_mimicgen import train_online
from online_rl_mimicgen.baselines import dsrl_dino
import stable_baselines3 as sb3
print(torch.__version__, numpy.__version__, lerobot.__version__, robosuite.__version__, mimicgen.__version__, sb3.__version__)"
#   [run] reference output: 2.11.0+cu130 2.3.5 0.6.2 1.4.1 1.0.1 2.6.0a1
bash tests/run_all.sh                                                            # [run] must end in "0 fail"
```

The RoboMeter teacher needs its own Python 3.10 environment. It is required only by the "TD + distill" cells
(section 5.6) and by rescoring rollouts (section 5.5):

```bash
# needs uv (https://docs.astral.sh/uv); the reference machine has 0.9.18
cp robometer/env/uv.lock third_party/robometer/uv.lock                           # [not executed]
(cd third_party/robometer && uv sync --frozen)                                   # [not executed]
```

### 5.2 Assets

All inputs are on the Hugging Face Hub under one account, `$HF_NAMESPACE` (default `chomeed`, set in
`scripts/env.sh`). `scripts/fetch_assets.py` maps every repository, folder and file to the path `configs/tasks.sh`
expects, renames where needed, and pins each repository to the revision recorded on 2026-10-12.

```bash
hf auth login                                                        # [not executed] several repositories are private
bash scripts/fetch_assets.sh --tier eval --dry-run                   # [run] lists every item, its target and size; downloads nothing
bash scripts/fetch_assets.sh --tier train --task square_d0           # [dry] download what RL on one task needs
bash scripts/fetch_assets.sh --tier train --check                    # [run] compare what is installed with the Hub (names, sizes)
```

| tier | for | size, all four tasks |
|---|---|---|
| `eval` | evaluating the prior or a checkpoint: the two base models, `meta/` of the demo datasets, the priors, both world models and the V directories of each task | about 5.3 GB |
| `train` | running RL cells: adds the whole demo datasets and the four teacher LoRAs | about 47 GB in total; about 9 GB with `--skip teacher` (each LoRA is 9.4 GB and only TD + distill cells need it) |
| `regenerate` | refitting world models and V: adds the prior-rollout datasets, the score tables, one more LoRA | about 57 GB in total |

The sizes are the output of the dry run. `--task`, `--only` and `--skip` narrow a download, `--check --sha256` also
hashes every file, and `--help` is the full description. A download that was interrupted continues on the next call.
`environment.md` section 5 has the complete layout table, and says which repositories are private (three
of the four priors, all world models, all V directories, the two square_d1 LoRAs): without a login that can read
them, the Hub answers "not found".

**Not published (as of 2026-10-12), and what each blocks:**

| asset | blocks | way around |
|---|---|---|
| the four `v_dino_mc_return_*` V directories | the DSRL `oursmc` cells: square_d0, square_d1, hammer_cleanup_d1 (9 of the 84 runs) | refit with `TASK=<task> RECIPE=mc bash value/fit_v.sh` after the `regenerate` tier and the frame cache; the result is equivalent, not identical |
| `rm_scores/square_d1_seg`, `rm_scores/threading_d0_seg` (score tables) | nothing in RL; only regenerating the two `pv4_seg` V directories, which are themselves published | rescore with `robometer/score_rollouts.py --rows` (section 5.5) |
| checkpoints, `config.json` and evaluation results of the reference runs | step 4 of the verification ladder (re-evaluating a reference checkpoint) | none; `reference/results_2026-10-12.md` holds their numbers |

Every other campaign cell can start from published assets: all DICE-RL cells and the DSRL `ours`, `oursdist`,
`randv` and `q` cells on their tasks. `scripts/fetch_assets.py` reports unpublished items as "unpublished", never as
an error.

### 5.3 The square_d1 prior

The square_d1 prior on the Hub was trained with a patched policy that reads the first 9 of 26 state dimensions; the
stock LeRobot policy used here cannot load it as published. The runs used a copy with three small files edited and
the weights untouched. `scripts/fetch_assets.py` downloads the published checkpoint and then runs the fix by itself:

```bash
python scripts/make_square_d1_statefix.py                            # [dry] (--help); builds priors/square_d1_45k_statefix
python scripts/make_square_d1_statefix.py --verify $CKPT_ROOT/priors/square_d1_45k_statefix/checkpoints/045000/pretrained_model   # [run]
```

On the reference machine the script's output was byte-identical to the prior the runs used (all 9 files), and
`--verify` passes on that prior. The other three priors are published already fixed.

### 5.4 Pixel caches and demo stores

Every RL cell mixes demonstrations into its batches and memory-maps their decoded frames from one file per task:

```bash
bash scripts/build_pixel_cache.sh square_d0 --print                  # [run] shows the command and the two paths
bash scripts/build_pixel_cache.sh square_d0                          # [not executed] CPU only; never overwrites
```

The files are large: 46.2 GB (square_d0), 45.9 (square_d1), 67.6 (threading_d0), 86.4 (hammer_cleanup_d1). The only
logged build time is 3 min 16 s for square_d1 (header of `scripts/build_pixel_cache.sh`).

DSRL additionally keeps a "demo store": the successful demonstration chunks in the layout of its replay, with the
prior's conditioning vector of every frame computed once (0.6 to 1.2 GB per task). The first DSRL run of a task
builds it in about 20 minutes and later runs reuse it. Its cache key contains the dataset path and the prior path,
so a store does not move between machines. To build one ahead of a run (`--executed-noise-only resample` is what
the campaign cells pass; with another value the run does not find the store and builds its own; `<prior>` is
`PRIOR` of `configs/tasks.sh`, for example `square_d0_40k_rs9`):

```bash
python -m online_rl_mimicgen.baselines.build_demo_store --task <task> --base-output $CKPT_ROOT/priors/<prior> --executed-noise-only resample   # [dry]
```

Then ask what a campaign cell still misses:

```bash
python tests/configs/test_configs.py                                 # [run] look at the test_installed_assets lines
```

`test_installed_assets` has three outcomes. FAIL lists what is missing or wrong (priors, world models, V
directories, state scaler, a pixel cache of the wrong shape). PASS may still print "NOT READY FOR TRAINING" lines
(pixel cache not built yet, demo dataset fetched with `meta/` only). `SKIP ... NO ASSETS FOUND - nothing was
checked` is not a pass.

### 5.5 Optional: regenerate world models and V

The released world models, score tables and V directories are the reference: none of the three trainings is
bit-reproducible on a GPU, so a regenerated artifact is statistically equivalent, not identical. The recipes are
provenance, and the way to extend the method to a new task. Every wrapper has `--print`, which shows the exact
command; `tests/pipeline/test_recipes.py` checks those commands against the metadata stored with the released files.

```bash
WS_TASK=<task> python dinowm/build_cache.py                          # [dry] frame cache of the prior rollouts, about 30 s, 17-24 GB
TASK=<task> STAGE=base bash dinowm/train_wm.sh                       # [dry] base world model, 5-10 h on one GPU
TASK=<task> STAGE=cont bash dinowm/train_wm.sh                       # [dry] continuation: --init the base model, the same number of steps again
SPOOL=<teacher> bash robometer/serve.sh                              # [dry] start a teacher (RoboMeter environment), about 11 GiB of GPU memory
WS_TASK=<task> python robometer/score_rollouts.py --spool $SPOOL_ROOT/<teacher> --out $CKPT_ROOT/rm_scores/<task>/predictions.parquet   # [dry]
TASK=<task> RECIPE=<recipe> bash value/fit_v.sh                      # [dry] <recipe> = pv4, pv4_seg or mc; one V directory, about 8 minutes, 10 GB of GPU memory
```

**These commands write to the released files' own paths and overwrite them without asking** (the world model at the
end of training, the score table and the V directory at once; `--resume` finds no state file for a downloaded model
and starts from scratch). To keep the reference inputs, regenerate under another root (`CKPT_ROOT=<scratch>/checkpoints`
with the inputs each step reads linked into it) or move the released directory away first.
`bash scripts/fetch_assets.sh --tier regenerate --check --sha256` shows whether a released file was replaced; the
asset test and a plain `--check` compare names and sizes only and will not notice.

| task | world-model steps | base tag | continuation tag | data seed of the continuation |
|---|---|---|---|---|
| square_d0 | 150000 | `ph_conv_150k` | `ph_conv_150k_cont150k` | 1 |
| square_d1 | 150000 | `ph_conv_150k` | `ph_conv_150k_cont150k` | 0 |
| threading_d0 | 168000 | `ph_conv_168k` | `ph_conv_168k_cont168k` | 1 |
| hammer_cleanup_d1 | 217000 | `ph_conv_217k` | `ph_conv_217k_cont217k` | 1 |

The base model is trained from scratch at peak learning rate 5e-4; the continuation loads it and trains the same
number of steps again with a fresh optimiser at 1e-4 (`dinowm/train_wm.sh`). Acceptance numbers for a retrain
(held-out feature skill, base / continuation): 0.467 / 0.453, 0.379 / 0.364, 0.548 / 0.539, 0.491 / 0.479 in the row
order above (`dinowm/train_wm_ph.py`).

| V recipe | target | score table | output directory under `$CKPT_ROOT/value` | read by |
|---|---|---|---|---|
| `pv4` | RoboMeter progress, first scoring | `rm_scores/<task>/` | `v_dino_robometer_pred_progress_<task>_<base tag>_mqattn_pv4` | DICE-RL DVD on every task; DSRL `ours` / `oursdist` / `randv` on square_d0 and hammer_cleanup_d1 |
| `pv4_seg` (square_d1 and threading_d0 only) | RoboMeter progress, rescored | `rm_scores/<task>_seg/` | `..._mqattn_pv4_seg` | DSRL `ours` / `oursdist` / `randv` on square_d1 and threading_d0 |
| `mc` | Monte Carlo return `gamma^(t_success - t)`, 0 on failed episodes | `rm_scores/<task>/` (row set only) | `v_dino_mc_return_<task>_<base tag>_mqattn_pv4` | DSRL `oursmc` |

Facts that are easy to get wrong:

- Every V was fitted through the **base** world model, even where the RL run loads the continuation.
- The row order of a score table is part of the V recipe (the fit samples rows by position). A first scoring
  writes success episodes first; a rescoring with `--rows` writes failure episodes first.
- The five teachers are in `robometer/teachers.sh`. A *segmented* LoRA (`-seg`) was fine-tuned on a set whose
  failure rollouts were cut where the attempt failed (`robometer/finetune/NOTES.md`); a non-segmented one on uncut
  rollouts. On
  square_d1 the plain `pv4` V was labelled by the NON-segmented LoRA and the `pv4_seg` V by the segmented one
  (teacher name `square_d1_seg`). On threading_d0 the two tables hold identical scores, value for value, in a
  different row order. The `config.json` of the released threading_d0 `pv4` and Monte Carlo V directories records
  the teacher under the name `robometer-4b-lora-threading-d0` (no `-seg`); given the identical scores this is the
  same labelling, but the two names were not traced to one upload.
- Held-out Spearman correlation of the released `pv4` fits with progress: 0.92, 0.89, 0.94, 0.92 for square_d0,
  square_d1, threading_d0, hammer_cleanup_d1 (`value/fit_v_mqattn.py`).
- Fine-tuning a teacher LoRA is optional and is not wrapped in a dry run: `robometer/finetune/train_lora.sh`
  `[not executed]`, described step by step in `robometer/finetune/NOTES.md`. It reads its training sets from the
  Hub account `$HF_NAMESPACE`, which is the default account unless you set another.

### 5.6 RL cells

```bash
bash scripts/run.sh configs/<method>/<arm>.sh TASK SEED [STEPS]      # [dry] STEPS defaults to 600000
TRAIN_ONLINE_DRYRUN=1 bash scripts/run.sh configs/dsrl/ours.sh square_d0 43      # [run] prints "DRYRUN_CONFIG <json>", starts nothing
RUN_PRINT_ARGV=1 bash scripts/run.sh configs/dice_rl/dvd.sh square_d0 43         # [run] prints run name, wandb group and the argv
bash scripts/run_campaign.sh configs/campaign/dsrl.txt 0 --list      # [run] the 48 DSRL runs and what would happen to each
bash scripts/run_campaign.sh configs/campaign/dice_rl.txt 0          # [not executed] run the 36 DICE-RL runs on GPU 0, in order
```

The campaign is every cell with **seeds 43 and 44 at 600000 env steps and seed 45 at 1000000**: 36 DICE-RL runs
and 48 DSRL runs. All 48 DSRL runs are finished; of the DICE-RL runs, the 12 with seed 45 had not been launched on
2026-10-13 (section 7). "Ran" below means the cell is part of the campaign; "defined" means the launcher accepts it but
no reference result exists. (T) marks a cell that needs the task's RoboMeter teacher for the whole run, (U) a cell
whose V directory is unpublished (section 5.2). The "paper recipe" of a task is the DICE-RL + DVD configuration the
paper reports for it; its "A/B twin" is the same cell with the other way of training V online, run to measure what
distillation changes.

| config file | V online | square_d0 | square_d1 | threading_d0 | hammer_cleanup_d1 |
|---|---|---|---|---|---|
| `configs/dice_rl/dvd.sh` (no `DISTILL`) | the task's paper recipe | ran: TD + distill (T) | ran: TD only | ran: TD only | ran: TD only |
| `configs/dice_rl/dvd.sh` with the other `DISTILL` value | the A/B twin | ran: `DISTILL=tdonly` | ran: `DISTILL=tddistill` (T) | ran: `DISTILL=tddistill` (T) | ran: `DISTILL=tddistill` (T) |
| `configs/dice_rl/q.sh` | no V; 10 pixel Q networks | ran | ran | ran | ran |
| `configs/dsrl/ours.sh` | TD only | ran | ran | ran | ran |
| `configs/dsrl/oursdist.sh` | TD + distill | ran (T) | defined (T) | defined (T) | ran (T) |
| `configs/dsrl/oursmc.sh` | TD only, V pretrained on Monte Carlo returns | ran (U) | ran (U) | defined (U) | ran (U) |
| `configs/dsrl/randv.sh` | TD only, V randomly initialised | ran | ran | defined | ran |
| `configs/dsrl/q.sh` | no V; 10 + 10 Q critics | ran | ran | ran | ran |

What each cell loads (`configs/tasks.sh` is the table; `configs/README.md` repeats it per arm):

- DICE-RL DVD loads the continued world model on square_d0, square_d1 and threading_d0 and the **base** world model
  on hammer_cleanup_d1, and the plain `pv4` V on every task. The hammer cell also differs in three flags: DVD on 256
  rows per update, one update per chunk, no `torch.compile`.
- DSRL DVD arms load the continued world model on all four tasks, the `pv4` V on square_d0 and hammer_cleanup_d1
  and the `pv4_seg` V on square_d1 and threading_d0 (`oursmc` instead loads the Monte Carlo V of the task).
- The Q cells load neither a world model nor a V.

**The teacher** (TD + distill cells: 12 DICE-RL runs and 6 DSRL runs of the campaign). It is a separate long-lived
process in RoboMeter's own environment; trainer and teacher exchange frames and scores through the directory
`$SPOOL_ROOT/<task>`.

```bash
bash scripts/robometer_teacher.sh show square_d0     # [run] prints the server command: LoRA, spool, instruction sentence
bash scripts/robometer_teacher.sh start square_d0    # [not executed] serves in the foreground; G=<gpu> pins it; about 11 GiB of GPU memory
bash scripts/robometer_teacher.sh check square_d0    # [run] exit 0 once a teacher answers (exit 1 on the reference test: none was serving that spool)
```

| task | LoRA under `$CKPT_ROOT/robometer/` | sentence passed as `--task` | spool |
|---|---|---|---|
| square_d0 | `robometer-4b-lora-square-d0-seg` | Pick up the square nut and place it on the peg. | `$SPOOL_ROOT/square_d0` |
| square_d1 | `robometer-4b-lora-square-d1` (the NON-segmented LoRA) | Pick up the square nut and place it on the peg. | `$SPOOL_ROOT/square_d1` |
| threading_d0 | `robometer-4b-lora-threading-d0-seg` | Pick up the needle and thread it through the hole in the tripod. | `$SPOOL_ROOT/threading_d0` |
| hammer_cleanup_d1 | `robometer-4b-lora-hammer-cleanup-seg` | Open the drawer, place the hammer inside, and close the drawer. | `$SPOOL_ROOT/hammer_cleanup_d1` |

The sentence must be the one the LoRA was fine-tuned with; a wrong sentence still returns plausible scores, so the
server has no default. `scripts/run_campaign.sh` skips a distilling run while no teacher answers. **`scripts/run.sh`
itself does not check**: a (T) cell started without a teacher prints `[rm-distill] DISABLED` after five scoring
failures (120 s timeout each) and trains on as TD only under its `tddistill` name, so run
`bash scripts/robometer_teacher.sh check <task>` first. A run whose log shows that line at any time lost its
teacher and is no longer a TD + distill run. The first start of a teacher on a machine must be online (`HF_OFFLINE`
unset): its base models are not fetched by `scripts/fetch_assets.py`. A distilling cell must use the
teacher that labelled the V it starts from; that holds for every cell that ran. The two `oursdist` cells that are
only "defined" would pair a `pv4_seg` V with the task's default teacher, which on square_d1 is a different LoRA.

Runs go to `$RUNS_ROOT/<run name>`; both trainers refuse to start in a populated directory, and
`scripts/run_campaign.sh` skips a run whose directory exists. wandb defaults to project `dvd` in offline mode;
export `WANDB_PROJECT` and `WANDB_MODE=online` in the shell to change that (the reference campaign logged online to
a project named `post_iclr2027`).

### 5.7 Evaluation

The two methods are scored with **different protocols, which must be named whenever numbers are shown**. Both use
100 episodes on environment seeds 10000 to 10099.

| | DICE-RL | DSRL | frozen prior alone |
|---|---|---|---|
| decision rule | best of 16 prior draws, each corrected by the deterministic residual, ranked by the run's own critic (`gamma^8 V(f)` for DVD, min-Q for Q) | ONE noise draw per chunk, no best-of-N | one prior draw per chunk |
| checkpoints | every 50k steps | every 25k steps | - |
| when | after the fact; the in-loop evaluation is off | during the run, by a background subprocess | any time |
| result file | `<run>/evals/<step>.json` | `<run>/evals/<step>.json` and `kind=eval` rows of `metrics.jsonl` | the `--out` file |

```bash
bash scripts/eval_loop.sh 0 --watch 600                              # [not executed] evaluate every DICE-RL checkpoint in $RUNS_ROOT on GPU 0
python eval/eval_bon16.py --ckpt <run>/checkpoints/<step>.pt --out <json>        # [dry] what the loop calls, plus --episodes 100
bash scripts/eval_dsrl.sh --ckpt <run>/checkpoints/<step>.zip --out <json>       # [dry] re-evaluate one DSRL checkpoint
python eval/eval_prior.py --task square_d0 --out <json>                          # [dry] the prior alone
python eval/compare_evals.py <a.json> <b.json> [--require-identical]             # [run] paired on shared seeds, exact McNemar test
```

Each episode reseeds torch, numpy, random and the environment with its own seed, so an episode replays on its own
and two result files can be compared seed by seed. The three evaluators were run for real on the reference
machine's GPU against stored results (section 6); `scripts/eval_loop.sh` was tested with a stand-in evaluator only.
One best-of-16 evaluation takes 10 to 45 minutes depending on the task; the reference machine required 12 GiB of
free GPU memory before starting one (`environment.md`, section 1).

### 5.8 Results table and wandb

```bash
python scripts/results_table.py [RUNS_DIR ...] [--bon16-dir DIR] [--runs] [--json FILE]   # [run] on the reference runs
WANDB_MODE=online python scripts/wandb_eval_mirror.py --once         # [not executed] against a wandb server; --dry-run lists what it would write ([dry]: --help)
```

`scripts/results_table.py` reads each run's `config.json` (never its name) and prints, per cell and seed, the mean
of the evaluations from 350k to 600k steps and the 600k evaluation, with the protocol in the heading.
`reference/results_2026-10-12.md` is its output on the reference runs. `scripts/wandb_eval_mirror.py` uploads the
DICE-RL evaluations as companion runs so that one wandb chart shows both methods; it needs the training runs to
exist on the wandb server.

## 6. Verification ladder for a new machine

Climb it in order; each step is much cheaper than the next and catches a different class of problem.

| step | command | checked | what to expect |
|---|---|---|---|
| 1. code and environment | `bash tests/run_all.sh` on a fresh clone, then the import check of section 5.1 | `[run]` | `0 fail` (42 pass, 28 skip in about a minute on a clone without the third-party checkouts; fewer checks skip once they are installed; what must hold is `0 fail`); the six version numbers of section 5.1 |
| 2. assets | `bash scripts/fetch_assets.sh --tier train --check`, `python tests/configs/test_configs.py`, then `bash tests/run_all.sh` again | `[run]` | `test_installed_assets` PASS without "NOT READY" lines for the task you will run; fewer SKIPs than in step 1, still `0 fail` |
| 3. the frozen prior | `python eval/eval_prior.py --task <task> --out <json>` for each task (one GPU, 100 episodes) | `[dry]` | 74 / 38 / 64 / 40 % on square_d0 / square_d1 / threading_d0 / hammer_cleanup_d1 (`eval/eval_prior.py`). On the reference machine a rerun matches the stored result episode for episode; on other hardware expect the rate within a few episodes |
| 4. a reference checkpoint | `python eval/eval_bon16.py --ckpt <ckpt> --out <new.json>` or `bash scripts/eval_dsrl.sh --ckpt <ckpt> --out <new.json>`, then `python eval/compare_evals.py <stored.json> <new.json>` | `[dry]`, `[dry]`, `[run]` | the stored rate within evaluation noise. Not possible today: no reference checkpoint is published (section 11) |
| 5. a short training run | `bash scripts/smoke_train.sh dsrl q square_d0` (or another METHOD ARM TASK) | `[dry]` | four lines `[smoke] reached ...` for collection, updates, checkpoint, evaluation, and `[smoke] PASS`. The numbers it prints mean nothing |
| 6. one real cell | `bash scripts/run.sh configs/dsrl/ours.sh square_d0 43`, then `python scripts/results_table.py --runs` | `[dry]`, `[run]` | the mean of the 350k-600k evaluations within about 5 points of the same seed in `reference/results_2026-10-12.md` (section 7 explains the tolerance and its exceptions) |

Step 3 is the cheapest check of everything that touches the simulator, rendering and the prior at once: a wrong
image size, a wrong prior, missing EGL or a different text input all show up as a different success rate.
Step 5 needs the cell's full assets and its start-up memory (an estimated 31-46 GiB of RAM for DICE-RL DVD and 23-37
GiB for DICE-RL Q; not known for DSRL; header of `scripts/smoke_train.sh`).

**Verified on the reference machine** (details in `tests/README.md` and in the test logs):

- Source: `rl/` against the research tree at the commit of `reference/LIVE_COMMIT`: 9 files identical after
  normalisation, 70 explained differences, none unexplained.
- Configuration: all 64 cell-budgets resolve to the reference launchers' configuration, and the `config.json` of all
  69 reference runs equals what this repository resolves for the same task, arm, seed and budget.
- Numerics on CPU: readout, V ensemble and world model give bit-identical outputs under both code trees for all 10
  V directories and all 8 world models.
- Old artifacts: every reference `config.json` upgrades with no value changed; all stored asset paths resolve under
  the roots to the identical files; a real DICE-RL checkpoint and real DSRL weights load.
- Evaluators on a GPU: 20-episode replays of stored results were identical episode by episode for DICE-RL DVD (two
  checkpoints, including every per-episode score statistic), DICE-RL Q, DSRL DVD, DSRL Q and the frozen prior.
- Assets: every Hub repository, folder and file name of `scripts/fetch_assets.py` exists; the installed datasets,
  priors, world models, V directories and score tables equal the Hub by SHA-256; the state-fix script reproduces the
  square_d1 prior byte for byte.
- The whole suite with the reference machine's inputs: 69 pass, 1 skip (the GPU test), 0 fail.

**Not verified:**

- **No training step of this repository has been executed on the reference machine.** The update loops of both
  trainers, `torch.compile`, the DSRL in-run evaluation subprocess and the RoboMeter exchange during training are
  covered only by source parity, configuration equality and CPU parity of the building blocks.
  `scripts/smoke_train.sh` was checked by dry run only.
- The installation: `requirements.txt` was not installed into a clean environment, the checkout block of section
  5.1 was not executed, the RoboMeter environment was not rebuilt from its lock file.
- Large downloads: `scripts/fetch_assets.py` downloaded only small JSON files for real. No pixel cache was built
  through `scripts/build_pixel_cache.sh`.
- The offline pipeline (frame cache, world-model training, scoring against a live teacher, V fits, LoRA
  fine-tuning) was not executed from this repository; its wrappers were checked against the released artifacts'
  metadata and its data rules by tests.
- Another GPU model, driver or CUDA version. The world model runs under bfloat16 (bf16) autocast, DSRL enables TF32 matmuls, the six
  DICE-RL DVD cells on square_d0, square_d1 and threading_d0 use `torch.compile`, and no deterministic-algorithm switch is set anywhere: expect statistical, not
  bitwise, agreement across machines.

## 7. Reference results

`reference/results_2026-10-12.md` is the table to reproduce, with one line per run below the per-cell tables. The
numbers here are its per-cell means of the 350k-600k evaluation window, in percent success.

DSRL, single noise draw per chunk, mean over seeds 43, 44, 45 (all 48 runs finished):

| arm | square_d0 | square_d1 | threading_d0 | hammer_cleanup_d1 |
|---|---|---|---|---|
| DVD, TD only (`ours`) | 89.9 | 53.8 | 84.4 | 61.1 |
| DVD, TD + distill (`oursdist`) | 89.3 | - | - | 53.6 |
| DVD, Monte Carlo initialised V (`oursmc`) | 88.2 | 55.5 | - | 56.2 |
| DVD, random V (`randv`) | 89.1 | 49.7 | - | 59.1 |
| Q (`q`) | 73.2 | 40.1 | 86.7 | 43.2 |

DICE-RL, best of 16, seeds 43 and 44 at 600k; the number of finished runs behind each mean is in brackets:

| arm | square_d0 | square_d1 | threading_d0 | hammer_cleanup_d1 |
|---|---|---|---|---|
| DVD, TD only | 90.8 (2) | 58.7 (1) | 83.9 (2) | 53.0 (1) |
| DVD, TD + distill | 91.3 (2) | 56.5 (1) | 88.8 (2) | 56.7 (1) |
| Q | 81.9 (2) | 40.3 (1) | 94.2 (1) | 41.3 (1) |

Frozen prior alone, single draw: 74 / 38 / 64 / 40 (`eval/eval_prior.py`).

- **The DICE-RL half is incomplete.** When the file was written 17 of the 36 DICE-RL runs were finished, 4 were
  training and 15 had not started; no seed-45 run existed. Regenerate the file when the reference queues are done.
- **Never put the two tables side by side as if they were one comparison**: best-of-16 and single-draw are
  different protocols. Compare DVD with Q inside one method.
- **Noise.** One 100-episode evaluation has a binomial standard error of 3 to 5 points. Inside one run the
  evaluations of the window scatter with a standard deviation between about 1 and 10 points (the `win (n, sd)`
  column of the per-run table). Across the three seeds of a DSRL cell the window means span 3 points in the median
  and up to 9; single 600k evaluations span 8.5 in the median and up to 21 (`scripts/results_table.py`).
- **So compare window means, not single evaluations or peaks.** A working rule: a reproduced run's 350k-600k mean
  within about 5 points of the reference run of the same seed, and a three-seed cell mean within about 3. A difference
  between two arms smaller than that is not established by one seed. The rule is loose on purpose and still has
  exceptions: a rerun on another machine is not seed-identical, and in the reference itself the window means of the
  three seeds span more than 4.5 points in three DSRL cells (`ours` on threading_d0 9.3, `ours` on hammer_cleanup_d1
  5.4, `q` on hammer_cleanup_d1 4.6).

## 8. Behaviours kept on purpose

These look like mistakes. They are how every reference run was trained and evaluated, so they are the default here
and no config exposes a switch for training. Changing any of them makes new numbers incomparable with the reference.

**1. The prior is conditioned on the task id, not on its training sentence.** `FrozenPolicyPrior` passes the string
`"square_d0"` (and so on) as the language input, although each prior was trained with a sentence. The training
sentences, also the sentences of the RoboMeter teachers:

| task | training sentence |
|---|---|
| square_d0, square_d1 | Pick up the square nut and place it on the peg. |
| threading_d0 | Pick up the needle and thread it through the hole in the tripod. |
| hammer_cleanup_d1 | Open the drawer, place the hammer inside, and close the drawer. |

**2. The prior's 2-frame history is the current frame twice.** The policy's observation queue is reset before every
chunk, so LeRobot fills both slots with the current observation. The prior was trained on two consecutive frames.

Measured effect, frozen prior alone, 100 paired episodes on seeds 10000-10099, percent success in the order
square_d0 / square_d1 / threading_d0 / hammer_cleanup_d1:

| variant | per task | pooled over 400 episodes | exact McNemar test, pooled |
|---|---|---|---|
| as run: task id, duplicated frame | 74 / 38 / 64 / 40 | 54.0 | - |
| training sentence (`--text train`) | 74 / 43 / 71 / 45 | 58.25 (+4.25 points) | p = 0.14 (50 against 67 discordant episodes) |
| real 2-frame history (`--history real`) | 73 / 37 / 60 / 42 | 53.0 (-1 point) | p = 0.79 (64 against 60) |

Neither difference is significant, on any task (per-task p >= 0.32) or pooled. Both are reproducible with
`eval/eval_prior.py`, which swaps the input inside its own process and checks on every call that the intended text
and frames reach the model:

```bash
python eval/eval_prior.py --task square_d1 --text train --out <a.json>           # [dry] the training sentence
python eval/eval_prior.py --task square_d1 --history real --out <b.json>         # [dry] real history
python eval/eval_prior.py --compare <taskid.json> <a.json>                       # [run] on the stored reference results
```

The simulator's own instruction string (`TaskSpec.instruction`, `--text spec`) is a third text: it differs from
the training sentence on three of the four tasks. One place does use real history: `record_rollouts.py`, the
recorder of prior rollouts, queries the prior once per primitive step. It also feeds the task id as text.

**3. DSRL's random-V arm starts every seed from the same random V.** In `configs/dsrl/randv.sh` the trainer's
start-up check calls `torch.manual_seed(0)` after the run seed is set and before V is built. The three seeds of
that arm differ in everything except V's initial weights.

**4. The wandb group `<task>_dicerl_dvd` means different things on different tasks.** Without `DISTILL`,
`configs/dice_rl/dvd.sh` runs the paper recipe of the task: TD + distill on square_d0, TD only on the other three.
The reference launcher gave the bare group name to that default and a suffixed name to the A/B twin
(`<task>_dicerl_dvd_tdonly` on square_d0, `<task>_dicerl_dvd_tddistill` elsewhere). Group and compare runs by
`config.json` (`value_distill_robometer` empty or set), never by group or run name. Related: `--value-distill-every
20` counts updates, and hammer_cleanup_d1 updates once per chunk where the other tasks update every fourth chunk, so
hammer distils about four times more often per environment step.

Smaller as-run facts, all encoded in `configs/` and checked by the tests:

- The DSRL TD-only arms still pass the eight `--value-distill-*` flags and switch them off with
  `--no-value-distill`; the spool path in their `config.json` is inert.
- The reference DICE-RL Q launcher builds `--nstep 1` and unsets it before use; no Q run was launched with the flag
  (1 is the default).
- The demonstration subset of the DICE-RL expert buffer is drawn with a fixed random generator and is the same for
  every seed.
- In `baselines/dsrl_dino.py`, `--value-distill-pred` is only validated and printed; DSRL always distils on
  world-model-predicted features.

## 9. Traps

### Environment

- **`PIXEL_IMG=224` is mandatory.** The image encoder reads it at import and defaults to 84. `scripts/env.sh` sets
  it; a Python process started without that file trains or evaluates on the wrong image size, or fails on a shape.
- **Rendering.** MuJoCo renders through EGL on the GPU; the EGL device must be the CUDA device (`G=<gpu>` sets
  both). A container that exposes the GPU without NVIDIA's EGL libraries fails at the first render;
  `environment.md` section 2 has the workaround. `EGLError` tracebacks printed when a finished process exits are
  harmless.
- **Video decoding.** torchcodec decodes the dataset videos with the system ffmpeg libraries (6.x). The pixel-cache
  build uses the ffmpeg binary bundled with imageio-ffmpeg instead.
- **Hugging Face offline mode.** Every reference run had the Hub switched off. Fill the cache first (the `eval` tier
  fetches the two base models), then `export HF_OFFLINE=1` in the shell; `scripts/env.sh` reads it when it is
  sourced. With the switch on and an empty cache the first model load fails. The base models are loaded by name, not
  by revision.
- **`WS_TASK` has no default.** The offline scripts in `dinowm/` and `value/` and `robometer/score_rollouts.py`
  take the task from this variable, read once at import; it names the cache files and every output. Set it per
  command (`WS_TASK=square_d0 python ...`). Never export it in a shell that also launches RL runs. The research code
  silently meant threading_d0 when it was unset; here an unset, unknown or contradicted value is refused.
- **RoboMeter lives in its own Python 3.10 environment** and must not see the main `PYTHONPATH`; the wrappers
  unset it. Installing either environment into the other breaks one of them.
- **`torch.compile` needs an executable cache directory.** Where the temporary directory is mounted `noexec`,
  point `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` at a normal disk (`dinowm/train_wm.sh` does this for the
  world-model trainer).
- **robosuite** comes from the PyPI wheel with one line that overflows under numpy 2 when a segmentation image is
  rendered. RL does not reach it; the segmenters under `robometer/finetune/` do (`environment.md`, section 3).
  hammer_cleanup_d1 additionally needs `robosuite_task_zoo` importable, or the environment silently never registers.

### Which code runs

- **`python -m` puts the working directory first on the import path.** Typed inside another tree that holds an
  `online_rl_mimicgen/` directory (the research tree, a second checkout), `python -m online_rl_mimicgen...` runs
  that tree's code, silently if that tree's environment is also active. Run module entry points from this
  repository's root, or through `scripts/` (`scripts/eval_dsrl.sh`, `scripts/build_pixel_cache.sh`).
- **`CODE_ROOT` is always the checkout `scripts/env.sh` lies in.** A different value left in the shell is replaced
  with one warning line. `scripts/run.sh`, `scripts/smoke_train.sh`, `scripts/eval_dsrl.sh` and
  `scripts/build_pixel_cache.sh` exit with status 3 if `python` would import the RL package from elsewhere, or if
  no `python` is on `PATH`.
- **The launchers start real training.** `scripts/run.sh`, `scripts/run_campaign.sh` and `scripts/smoke_train.sh`
  train whenever the asset roots hold real priors. To test anything around them use `TRAIN_ONLINE_DRYRUN=1`,
  `RUN_PRINT_ARGV=1` or `--list`. A stub `python` placed in a `noexec` directory is skipped by `PATH` lookup and the
  real one runs.
- **Asset roots made of symbolic links are written through.** `scripts/fetch_assets.py` and
  `scripts/build_pixel_cache.sh` write into wherever a linked target directory points. Use `--dry-run`, `--check`
  or `--print` against such a tree.

### Reading runs

- **Read `config.json`, not the run name or the wandb group.** Names are long and encode the cell, but they are
  copied from the reference launchers and are ambiguous in places (section 8, point 4).
- **Checkpoints must be real files inside `<run>/checkpoints/`.** Both evaluators resolve symbolic links first.
  `eval_dsrl_ckpt` then looks for `config.json` two levels above the real file, so a linked checkpoint is evaluated
  with the configuration next to the link's target. `eval/eval_bon16.py` reads the configuration from the
  checkpoint itself, so there a link only mislabels the `run` field of the result. Download a run into a plain directory, not out of the Hugging Face hub
  cache (whose files are links). `eval_dsrl_ckpt` accepts `--config` for a checkpoint that sits elsewhere.
- **Prior checkpoints have no `last` directory.** They are `priors/<name>/checkpoints/<step>/pretrained_model`
  with steps 040000, 045000, 055000, 035000 for square_d0, square_d1, threading_d0, hammer_cleanup_d1. An explicit
  `--base-checkpoint` (`eval/eval_bon16.py`) must name that `pretrained_model` directory; `--base-output` (both
  trainers, `eval_dsrl_ckpt`) names the prior's run directory `$CKPT_ROOT/priors/<name>`.
- **Evaluation files are matched by time.** `scripts/eval_loop.sh` counts a result only if it is newer than its
  checkpoint.
- **Do not rebuild a DSRL command line from the `config.json` of an old run.** Configs written before a flag
  existed lack its key, and the parser's default today is not always what such a run used (the activation, for
  example). `eval_dsrl_ckpt` fills in the old behaviour; a hand-built argv does not.

### Metrics

- **Collection success is not evaluation success.** The `episode_success` value in `kind=train` rows is a rolling
  mean over exploration episodes. Use the `evals/` results; for learning curves use the `kind=episode` rows.
- **`mc_return` is a literal 0.0 on online replay rows** in the campaign's configuration, and the diagnostics treat
  0.0 as a valid return. `q_vs_mc_mae`, `q_vs_mc_rank` and their `v_vs_mc_*` counterparts therefore compare mostly
  against zeros as the demonstration share falls. Recompute returns from reward and done flags instead.
- **`dvd_feat_mse` and `dvd_dynamics_loss` are training error**: they are measured on the replay batch the world
  model was just fitted on. They can fall while error on unseen states rises.
- **A falling `dvd_scale` is not an instability.** It is the batch mean of `|gamma^8 V(f)|`, the denominator of the
  DVD term, so it follows V's level.
- **A negative mean V or Q is a bug signal** (section 3).
- **A value scored on the buffer it was trained on is in-sample.** Agreement of V with returns on the replay says
  little about its quality on new states; score across seeds or on held-out episodes.
- **V agreeing with returns is not policy progress.** Only the evaluations measure the policy.
- **Seeds.** A single seed cannot resolve a 5-point effect (section 7).

### Resources

All numbers are measured on the reference machine (`environment.md`, section 1).

| run | GPU memory | host RAM at the end of 600k / 1M steps | wall clock for 600k |
|---|---|---|---|
| DICE-RL DVD | 37 GiB | 95-110 / 115-131 GiB | 8-15 h (hammer 17-30 h) |
| DICE-RL Q | 13 GiB | 66-80 / 79-93 GiB | 11-18 h |
| DSRL DVD | not measured | 74-76 GiB including its two evaluation processes | 8.5-17 h |
| DSRL Q | not measured | 65 / 79 GiB | 22-34 h |
| RoboMeter teacher | 11 GiB | 2.3 GiB | - |

- **Memory grows for the whole run** because the replay holds images. A run that fits at start-up can exceed the
  machine hours later; `scripts/run_campaign.sh` does not watch this. Plan with the end-of-run numbers.
- **Disk.** One checkpoint is 0.34 GB (DICE-RL DVD), 0.14 (DICE-RL Q), 0.38 (DSRL DVD) or 0.33 (DSRL Q). The whole
  campaign with every checkpoint kept is 669 GB, plus 246 GB of pixel caches: about 960 GB with the assets. Deleting
  intermediate checkpoints once their evaluation exists brings it to about 320 GB. Nothing in this repository
  deletes checkpoints; `environment.md` section 1 has a snippet that lists the candidates.
- **CPU threads.** On a shared machine set `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` for manual evaluations, as
  `scripts/eval_loop.sh` does. On the busy 16-CPU reference machine the same 20-episode evaluation took 405 s with
  unbounded threads and 154 s with 2.

## 10. Tests and gates

```bash
bash tests/run_all.sh                    # [run] every script, one summary line each, then the totals
python tests/configs/test_configs.py     # [run] one script on its own (after `source scripts/env.sh`)
```

The tests are ten plain Python scripts (no pytest). Each check prints `PASS`, `SKIP` or `FAIL`. They run on CPU,
write only into temporary directories and start no training. **SKIP means an input was absent and nothing was
verified**; the line names the input. It is expected on a fresh clone and is a to-do on a machine that is about to
train. `tests/README.md` lists the variables that turn SKIPs into checks.

| script | what a PASS proves | needs |
|---|---|---|
| `tests/configs/test_configs.py` | `scripts/run.sh` produces the reference argv, run name and wandb group for all 32 cells at both budgets; both trainers resolve it to the reference configuration key by key; the campaign lists are complete; the fetch table targets the paths the configs name; the smoke run is the cell with only its budget changed; installed assets are what the cells need | nothing; more with assets, the DSRL checkout, reference runs |
| `tests/rl/test_source_parity.py` | `rl/` differs from the research tree only by the explained hunks; no file but `legacy.py` spells the old name | the research tree (`LIVE_TREE`) for the comparison itself |
| `tests/rl/test_legacy.py` | the rename tables, the upgrade functions, the converter, and that the trainers' loaders refuse unconverted checkpoints; with reference runs, that every real config and checkpoint upgrades unchanged | nothing; more with `LIVE_TREE`, `LEGACY_RUNS` |
| `tests/rl/test_cpu_parity.py` | readout, V and world model compute bit-identical numbers under both code trees | `LIVE_TREE` and the weights |
| `tests/pipeline/test_pool_mqattn.py` | the pv4 readout's math and that released readouts reload with the expected shape | nothing; more with V directories |
| `tests/pipeline/test_dino_wm.py` | world-model loading, gradient path, freezing, one encode-predict-pool pass | a world model and DINOv2-S in the cache |
| `tests/pipeline/test_recipes.py` | the offline wrappers print the released recipes; a missing or wrong `WS_TASK` is refused; no machine-specific path, address or credential anywhere outside `reference/` | nothing; more with released artifacts |
| `tests/pipeline/test_offline_data.py` | the context-window, row-order and Monte Carlo target rules | nothing; more with frame caches and score tables |
| `tests/pipeline/test_eval_tools.py` | `eval/compare_evals.py`; the prior's as-run defaults; `eval/eval_bon16.py` upgrades old checkpoints and refuses bad overrides | nothing; more with the DSRL checkout, reference runs |
| `tests/pipeline/test_bon16_parity.py` | `eval/eval_bon16.py` assembles the trainer's scorer correctly; always SKIP in `run_all.sh` | a GPU and a decision dump (`tests/README.md`) |

Last measured: fresh clone 42 pass, 28 skip, 0 fail; reference machine with all inputs 69 pass, 1 skip, 0 fail.

**The three gates.** Any change to the RL code or the configs must keep these three green; together they are the
evidence that this repository is the code that produced the reference results.

1. **Source parity.** An edit to a file under `rl/online_rl_mimicgen/` shows up as an unexplained hunk. On a machine
   that has the research tree: `LIVE_TREE=<research tree> python tests/rl/test_source_parity.py --show` `[run]` to
   read the hunks, `--update` `[not executed]` to add them to the allow-list, then write the reason of every new
   entry (the test fails while a `TODO` reason is left). The allow-list was generated under Python 3.12.
2. **Configuration equality.** After any edit to `configs/`, `scripts/run.sh` or a trainer's arguments, run
   `python tests/configs/test_configs.py`. The golden files can only be regenerated on the reference machine
   (`tests/configs/make_golden.py`).
3. **CPU parity** of V, readout and world model.

None of the gates executes a training step (section 6).

**Rules when you change this repository.**

- Do not edit anything under `reference/`: the files are verbatim copies with checksums
  (`cd reference && sha256sum -c SHA256SUMS` `[run]`). Do not edit the golden files by hand.
- Keep every path behind the roots of `scripts/env.sh`. No machine-specific path, user name, address or credential
  may appear outside `reference/`; `tests/pipeline/test_recipes.py` scans for them.
- Spell the method's old name in `rl/online_rl_mimicgen/legacy.py` only.
- Never commit weights, datasets, caches or run outputs (`.gitignore` lists the patterns).
- After editing a shell script run `bash -n <script>` `[run]`; after any edit run `bash tests/run_all.sh`.

## 11. Known gaps and next steps

**For the owner of the assets:**

1. Publish the four `v_dino_mc_return_*` V directories (13 MB each). Until then the three DSRL `oursmc` cells cannot
   be reproduced exactly on another machine.
2. Publish the two `_seg` score tables (`rm_scores/square_d1_seg`, `rm_scores/threading_d0_seg`).
3. Publish the reference runs: at least `config.json`, the evaluation results and the last checkpoint of each run
   (28 GB for the whole campaign, `environment.md` section 1). Without them step 4 of the verification ladder is
   impossible and the reference numbers exist only as a table.
4. Give whoever reproduces the work read access to the private repositories of `$HF_NAMESPACE`.
5. Regenerate `reference/results_2026-10-12.md` with `scripts/results_table.py` once the DICE-RL runs have finished.

**First things to do on the new machine, because they were never done:**

6. Install the environment from `requirements.txt` and the checkout block, and rebuild the RoboMeter environment
   from `robometer/env/uv.lock`; compare versions with `environment.md`.
7. Run `scripts/smoke_train.sh` for one cell of each method. It has never been executed anywhere.
8. Run one seeded cell and compare it with the reference (step 6 of the ladder). A short GPU comparison of this
   code against the research tree on the same cell and seed is the one missing piece of direct evidence.

**Untested paths of the tools:** a full download of a large item and its resume after an interruption
(`scripts/fetch_assets.py`; LoRA weight shards were compared by name and size, not hashed); a namespace other than
the default (the pinned revisions belong to it; the script falls back to each repository's head with a warning); a
real build through `scripts/build_pixel_cache.sh`; `scripts/eval_dsrl.sh` beyond `--help`; `scripts/eval_loop.sh`
with the real evaluator; `scripts/wandb_eval_mirror.py` against a server; `scripts/robometer_teacher.sh start`. The
base models of the RoboMeter side are not fetched by `scripts/fetch_assets.py`; the teacher environment downloads
them at its first online start (`environment.md`, section 4).

**Code:**

- `train_online.py` is not pruned (section 2.5).
- Four DSRL cells are defined and tested for configuration but have no reference result: `oursdist` on square_d1 and
  threading_d0, `oursmc` and `randv` on threading_d0.
- No segmentation script exists for the threading_d0 and square_d1 teacher LoRAs, and the arguments used for the two
  segmenters that are shipped were not logged; those LoRAs can only be used as downloaded
  (`robometer/finetune/NOTES.md`).
- The offline trainings (world model, V, LoRA) are not bit-reproducible; the released files are the reference.

## 12. Glossary

| term | meaning |
|---|---|
| prior | the frozen flow-matching DiT policy of a task; proposes chunks |
| chunk | the 8 actions executed between two decisions |
| DVD | the method: train the actor by ascending `gamma^8 V(f(s, a))`; called SVG in the research code |
| DICE-RL (`dice_rl`) | residual actor on the prior, plus best-of-16 selection by a critic |
| DSRL | noise-space SAC on the prior (Wagenmaker et al., CoRL 2025); `na` in run names is the DSRL variant used |
| f, world model, DINO-WM | the predictor of DINOv2 patch features 8 steps ahead |
| `ph_conv_<N>k` | world-model tag: proprio head, conv action encoder, N thousand training steps; `_cont<N>k` = continued for N thousand more |
| V | the value function: readout + 3 MLPs on predicted features and proprio |
| mqattn, pv4 | the multi-query attention readout; pv4 = per view (camera), 4 queries |
| `pv4_seg` | the same V recipe on labels from a rescoring (on square_d1: by the segmented LoRA) |
| dpinit, `dpinitvtd` | V initialised from the offline fit on RoboMeter progress, then trained online with TD |
| mcinit, `mcinitvtd` | V initialised from an offline fit on Monte Carlo returns |
| randv | V (heads and readout) randomly initialised |
| pixq | the DICE-RL Q baseline: an ensemble of 10 pixel Q networks |
| TD only / TD + distill (`tdonly` / `tddistill`) | how V is trained online: TD learning alone, or plus regression on RoboMeter scores |
| RoboMeter, teacher, LoRA | the video-language progress model; one fine-tuned copy per task that serves scores; its fine-tuned adapter weights |
| spool | the directory through which trainer and teacher exchange frames and scores |
| best of 16, bon16, bo16 | the DICE-RL decision rule and evaluation protocol |
| single draw | the DSRL evaluation protocol: one noise sample per chunk |
| window mean, `win` | mean of a run's evaluations from 350k to 600k steps |
| statefix | a prior whose config declares the 9-D state it was trained on |
| pixel cache | decoded demonstration frames of a task, one large `.npy` |
| frame cache | decoded prior-rollout frames of a task (offline pipeline only) |
| demo store | DSRL's demonstration chunks in the layout of its replay, with the prior's conditioning computed once |
| env seed | the seed of one evaluation episode; 10000 to 10099 everywhere |
| sidecar | the `<step>_dvd.pt` file beside a DSRL `<step>.zip`; in the DVD arms it holds V, the fine-tuned world model and their optimisers |
| reference machine, research tree | section 1 |
