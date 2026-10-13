#!/usr/bin/env python3
"""Evaluation of the frozen DiT prior alone (no RL): the "base prior" row of every table.

THE PROTOCOL (version base-single-draw-v1)
  prior    FrozenPolicyPrior(task, <prior>, action_steps 8, inference_steps 10, prior_temperature 1.0): the values in
           every campaign config. The prior's native n_action_steps 24 / 100 integration steps are NOT used.
  decision train_online.flatten_chunk(prior.action_chunk_samples(obs, 1)): ONE prior draw per 8-step chunk, no
           best-of-N, no residual, executed by train_online.rollout_chunk
  env      train_online.make_env(task, None, False, None): TaskSpec horizon (square / threading 400, hammer 500)
  seeds    episode i: s = seed_start + i; torch.manual_seed(s); np.random.seed(s); random.seed(s); env.reset(seed=s)
  success  any primitive step with info["is_success"] (the env terminates on success and truncates at the horizon)
  globals  online_rl_mimicgen.dino_dvd is imported, as in every trainer and evaluator. Its import disables cuDNN SDPA,
           which selects the attention kernel the DiT and CLIP run on.

TWO INPUTS OF THE PRIOR ARE NOT WHAT IT WAS TRAINED WITH, in every RL run, evaluation and reference number. The defaults
of this script reproduce that as-run behaviour; the options exist to measure what the difference costs.

  --text     what the prior's language input is
     taskid  (default, as run) the bare task id, e.g. "square_d0": FrozenPolicyPrior sets batch["task"] = [task id]
     train   the sentence the prior was trained with (TRAIN_TEXT, checked against the dataset's meta/tasks.parquet)
     spec    the env TaskSpec.instruction, which LeRobot's own evaluation would feed
     <str>   any other value is used as that literal string
  --history  what the prior's 2-frame observation window (n_obs_steps = 2) holds
     dup     (default, as run) [o_t, o_t]: FrozenPolicyPrior calls policy.reset() before every chunk, so LeRobot fills
             the window with the current observation twice
     real    [o_{t-1}, o_t] from the second chunk of an episode on, o_{t-1} being the observation the env returned one
             primitive step before the current one (the prior was trained on consecutive 20 Hz frames). The first
             chunk keeps [o_0, o_0], as LeRobot does after a reset.

Both are swapped IN THIS PROCESS ONLY (no file is edited) and verified on every call: the language tokens that reach
the model are compared with the tokenizer's encoding of the intended string, and with --history real (or
--check-history) the queued window is compared with the intended frames; a mismatch raises. With the defaults nothing
but the token check is installed, and that check replaces the task id with itself.
Reference success rates, 100 episodes on env seeds 10000-10099, in the order square_d0 / square_d1 / threading_d0 /
hammer_cleanup_d1:   as run (taskid, dup) 74 / 38 / 64 / 40 %;   --text train 74 / 43 / 71 / 45 %;   --history real
73 / 37 / 60 / 42 %. Paired on the same seeds neither change is significant on any task (exact McNemar p >= 0.32).

Output: ONE JSON (--out), never overwritten. An interrupted run resumes from <out stem>.partial.json; the resume is
exact because episodes are seeded independently.

usage: python eval/eval_prior.py --task square_d0 --out <json> [--text taskid] [--history dup] [--episodes 100]
           [--seed-start 10000] [--prior <pretrained_model dir>] [--dataset-root <demo dataset>]
       python eval/eval_prior.py --compare <a.json> <b.json>      paired on shared seeds, exact McNemar test
One GPU: CUDA_VISIBLE_DEVICES=g MUJOCO_EGL_DEVICE_ID=g.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "rl"
for _p in (str(Path(__file__).resolve().parent), str(REPO / "envs/lerobot_env_mimicgen/src"), str(R)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)                     # this repository's packages, whatever PYTHONPATH says

import compare_evals  # noqa: E402

TASKS_EVAL = ("square_d0", "square_d1", "threading_d0", "hammer_cleanup_d1")
# The sentence each prior (and each RoboMeter LoRA) was trained with: the single entry of the task dataset's
# meta/tasks.parquet. Not the env's TaskSpec.instruction on three of the four tasks.
TRAIN_TEXT = {
    "square_d0": "Pick up the square nut and place it on the peg.",
    "square_d1": "Pick up the square nut and place it on the peg.",
    "threading_d0": "Pick up the needle and thread it through the hole in the tripod.",
    "hammer_cleanup_d1": "Open the drawer, place the hammer inside, and close the drawer.",
}
ACTION_STEPS, INFERENCE_STEPS, PRIOR_TEMPERATURE, STEP_PENALTY = 8, 10, 1.0, 0.0
PROTOCOL_VERSION = "base-single-draw-v1"

_RL: dict = {}
# history bookkeeping of the running episode (see install_history)
H: dict = dict(mode="dup", obs=[], prev=None, chunks=0, chunks_with_prev=0, state_gap_sum=0.0, checked=0)


def rl() -> dict:
    """Import the RL code once (lazily, so --compare needs no torch)."""
    if not _RL:
        import torch
        import online_rl_mimicgen
        assert R in Path(online_rl_mimicgen.__file__).resolve().parents, (
            f"online_rl_mimicgen resolved to {online_rl_mimicgen.__file__}, not to {R}: another copy was imported first")
        from online_rl_mimicgen import dino_dvd  # noqa: F401  (import side effect = the protocol's global: cuDNN SDPA off)
        from online_rl_mimicgen import train_online as T
        from online_rl_mimicgen.common import TASKS, FrozenPolicyPrior, find_latest_checkpoint
        _RL.update(torch=torch, T=T, TASKS=TASKS, FrozenPolicyPrior=FrozenPolicyPrior,
                   find_latest_checkpoint=find_latest_checkpoint)
    return _RL


def resolve_text(task: str, mode: str) -> tuple[str, str]:
    if mode == "taskid":
        return "taskid", task
    if mode == "train":
        return "train", TRAIN_TEXT[task]
    if mode == "spec":
        from lerobot_env_mimicgen.environment import normalize_task
        return "spec", normalize_task(task).instruction
    return "literal", mode


def train_text_evidence(task: str, prior_dir: Path, dataset_root: Path) -> dict:
    """Check TRAIN_TEXT[task] against the task dataset's meta/tasks.parquet; record which dataset the prior was trained on."""
    import pandas as pd
    pq = Path(dataset_root) / "meta/tasks.parquet"
    df = pd.read_parquet(pq)
    strings = [str(s) for s in (df["task"] if "task" in df.columns else df.index)]
    if strings != [TRAIN_TEXT[task]]:
        raise SystemExit(f"{pq} lists {strings}, but TRAIN_TEXT[{task!r}] = {TRAIN_TEXT[task]!r}; refusing to guess")
    try:
        prior_ds = json.loads((prior_dir / "train_config.json").read_text())["dataset"]["repo_id"]
    except Exception as e:                                           # noqa: BLE001
        prior_ds = f"unreadable ({e})"
    return dict(tasks_parquet=str(pq), strings=strings, prior_train_dataset=prior_ds,
                same_dataset=str(prior_ds).split("/")[-1] == Path(dataset_root).name)


class TextOverride:
    """Stands in for prior.preprocessor. It swaps batch["task"] (which must be the task id) for `text`, runs the original
    preprocessor, and checks the produced language tokens against tokenizer(text) on every call."""

    def __init__(self, prior, text: str):
        torch = rl()["torch"]
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        self.prior, self.inner, self.text = prior, prior.preprocessor, text
        self.task_id = prior.env_cfg.task
        self.k_ids, self.k_mask = OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        steps = [s for s in self.inner.steps if type(s).__name__ == "TokenizerProcessorStep"]
        if len(steps) != 1:
            raise RuntimeError(f"expected one TokenizerProcessorStep in the prior's preprocessor, found {len(steps)}")
        self.tok_step = steps[0]
        if self.tok_step.task_key != "task":
            raise RuntimeError(f"tokenizer reads complementary_data[{self.tok_step.task_key!r}], not 'task'")
        enc = self.tok_step._tokenize_text([text])
        self.exp_ids, self.exp_mask = enc["input_ids"], enc["attention_mask"].to(torch.bool)
        self._dev: dict = {}
        self.calls = self.rows = 0
        self.seen: dict[str, int] = {}                 # string -> rows that reached the tokenizer with it
        self.produced: list[int] | None = None         # unmasked ids of the first produced row
        self._had_attr = "_tokenize_text" in vars(self.tok_step)
        self._prev_attr = vars(self.tok_step).get("_tokenize_text")
        orig = self.tok_step._tokenize_text

        def spy(t, _orig=orig):
            for s in ([t] if isinstance(t, str) else list(t)):
                self.seen[s] = self.seen.get(s, 0) + 1
            return _orig(t)

        self.tok_step._tokenize_text = spy
        prior.preprocessor = self

    def _expected(self, device, rows: int):
        if device not in self._dev:
            self._dev[device] = (self.exp_ids.to(device), self.exp_mask.to(device))
        ids, mask = self._dev[device]
        return ids.expand(rows, -1), mask.expand(rows, -1)

    def __call__(self, batch):
        torch = rl()["torch"]
        task = batch.get("task")
        if not (isinstance(task, list) and task and all(t == self.task_id for t in task)):
            raise RuntimeError(f"prior call arrived with batch['task'] = {task!r}, expected [{self.task_id!r}] * B; the "
                               "call site changed, so this override no longer knows what it replaces")
        b = dict(batch)
        b["task"] = [self.text] * len(task)
        out = self.inner(b)
        ids, mask = out[self.k_ids], out[self.k_mask].to(torch.bool)
        e_ids, e_mask = self._expected(ids.device, len(task))
        if not (torch.equal(ids, e_ids) and torch.equal(mask, e_mask)):
            raise RuntimeError(f"language tokens of call {self.calls} differ from tokenizer({self.text!r})")
        if self.produced is None:
            self.produced = ids[0][mask[0]].tolist()
        self.calls += 1
        self.rows += len(task)
        return out

    def __getattr__(self, name):
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def restore(self) -> None:
        self.prior.preprocessor = self.inner
        if self._had_attr:
            self.tok_step._tokenize_text = self._prev_attr
        else:
            del self.tok_step._tokenize_text

    def evidence(self) -> dict:
        tok = self.tok_step.input_tokenizer
        exp = self.exp_ids[0][self.exp_mask[0]].tolist()
        got = self.produced
        return dict(replaced_from=self.task_id, text=self.text, task_key=self.tok_step.task_key,
                    tokenizer=getattr(tok, "name_or_path", None), tokenizer_class=type(tok).__name__,
                    max_length=self.tok_step.max_length, prior_calls=self.calls, rows=self.rows,
                    every_call_matched_tokenizer_of_text=True,       # a mismatch raises, so a finished run means True
                    strings_reaching_tokenizer=self.seen,
                    produced_token_ids=got, produced_tokens=None if got is None else tok.convert_ids_to_tokens(got),
                    produced_decoded=None if got is None else tok.decode(got, skip_special_tokens=True),
                    expected_token_ids=exp, produced_equals_expected=got == exp)


def _prepared(prior, observation):
    """The batch select_action queues for `observation`: action_chunk_samples(observation, 1)'s preparation, then the
    policy's own _prepare_batch (stacks the camera images), with nothing else."""
    from lerobot.envs.utils import preprocess_observation
    batch = prior._narrow(preprocess_observation(observation))
    batch = {k: v.repeat(1, *([1] * (v.ndim - 1))) for k, v in batch.items()}
    batch["task"] = [prior.env_cfg.task]
    batch = prior.policy._prepare_batch(prior.preprocessor(batch))
    batch.pop("action", None)          # select_action drops it too; queued, it would fill the ACTION queue with None
    return batch


def install_history(prior, env, mode: str) -> None:
    """--history real (and the --check-history verification of dup). In this process only:
      - env.step keeps copies of the last two raw observations;
      - policy.reset pre-populates the queues with the previous observation (real, not the first chunk), prepared through
        exactly the path action_chunk_samples uses for the current one; select_action's populate_queues then appends o_t;
      - select_action checks after every plan that the queue holds [prev, cur] (real) or [cur, cur] (dup)."""
    from lerobot.policies.utils import populate_queues
    H["mode"] = mode
    pol = prior.policy
    orig_reset, orig_step, orig_select = pol.reset, env.step, pol.select_action

    def reset():
        orig_reset()
        if H["prev"] is not None:                       # real mode, not the first chunk: [prev, prev] -> select appends cur
            populate_queues(pol._queues, H["prev"], exclude_keys=["action"])

    def step(action):
        out = orig_step(action)
        H["obs"].append(copy.deepcopy(out[0]))
        del H["obs"][:-2]
        return out

    def select_action(batch):
        torch = rl()["torch"]
        out = orig_select(batch)                       # populate_queues + one plan, exactly as before
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
        cur_b = pol._prepare_batch(dict(batch))
        for key in (OBS_STATE, OBS_IMAGES):
            q = list(pol._queues[key])
            if len(q) != 2 or not torch.equal(q[1], cur_b[key]):
                raise RuntimeError(f"the last queued {key} is not the current observation")
            want0 = H["prev"][key] if H["prev"] is not None else cur_b[key]
            if not torch.equal(q[0], want0):
                raise RuntimeError(f"the first queued {key} is not the intended previous frame")
        H["checked"] += 1
        if H["prev"] is not None:
            q = list(pol._queues[OBS_STATE])
            H["state_gap_sum"] += float((q[1] - q[0]).abs().mean())
        return out

    pol.reset, env.step, pol.select_action = reset, step, select_action


def build_prior(task: str, prior_dir: Path, device: str = "cuda"):
    return rl()["FrozenPolicyPrior"](task, Path(prior_dir), ACTION_STEPS, INFERENCE_STEPS, device=device,
                                     prior_temperature=PRIOR_TEMPERATURE)


def build_env(task: str):
    return rl()["T"].make_env(task, None, False, None, holder_sigma=0.0, fixed_goal=False, fixed_goal_rot=False,
                              full_fixed=False, jitter_goal_m=0.0)


def seed_all(seed: int) -> None:
    torch = rl()["torch"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def decide(prior, observation):
    """One prior draw for this observation, flattened (1, 8*7)."""
    return rl()["T"].flatten_chunk(prior.action_chunk_samples(observation, 1))


def run_episode(prior, env, seed: int) -> dict:
    T = rl()["T"]
    seed_all(seed)
    H["obs"].clear()
    tr = time.perf_counter()
    observation, _ = env.reset(seed=seed)          # robosuite hard reset: rebuilds the sim (and, on the first call, the env)
    t_reset = time.perf_counter() - tr
    done, success, length, chunks, t_prior, t_env, first_md5 = False, False, 0, 0, 0.0, 0.0, None
    while not done:
        t0 = time.perf_counter()
        H["prev"] = None
        if H["mode"] == "real" and len(H["obs"]) == 2:
            H["prev"] = _prepared(prior, H["obs"][0])
            H["chunks_with_prev"] += 1
        H["chunks"] += 1
        chosen = decide(prior, observation)
        H["prev"] = None
        action = chosen.view(1, prior.action_steps, prior.action_dim)
        a_np = action[0].cpu().numpy()
        t1 = time.perf_counter()
        if first_md5 is None:
            first_md5 = hashlib.md5(np.ascontiguousarray(a_np, dtype=np.float32).tobytes()).hexdigest()
        observation, _, done, chunk_success, steps = T.rollout_chunk(env, a_np, None, STEP_PENALTY)
        t_prior += t1 - t0
        t_env += time.perf_counter() - t1
        length += steps
        chunks += 1
        success = success or chunk_success
    return dict(seed=seed, success=bool(success), length=int(length), chunks=chunks, first_chunk_md5=first_md5,
                reset_seconds=round(t_reset, 3), prior_seconds=round(t_prior, 3), env_seconds=round(t_env, 3))


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_md5s() -> dict:
    """md5 of every imported module file of this repository and of the lerobot policy / tokenizer code the prior runs on."""
    import lerobot
    lr = Path(lerobot.__file__).resolve().parent
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        p = Path(f).resolve()
        if p.is_relative_to(REPO):
            out[str(p.relative_to(REPO))] = md5_file(p)
        elif p.is_relative_to(lr / "policies/multi_task_dit") or p in (lr / "processor/tokenizer_processor.py",
                                                                        lr / "envs/utils.py"):
            out["lerobot/" + str(p.relative_to(lr))] = md5_file(p)
    return dict(sorted(out.items()))


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def atomic_write(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def evaluate(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite it")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.json")
    t_start, started = time.time(), datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    task = args.task
    m = rl()
    torch, T = m["torch"], m["T"]
    prior_dir = Path(args.prior) if args.prior else m["find_latest_checkpoint"](Path(m["TASKS"][task]["il_output"]))
    dataset_root = Path(args.dataset_root) if args.dataset_root else Path(m["TASKS"][task]["dataset_root"])
    text_mode, text = resolve_text(task, args.text)
    version = PROTOCOL_VERSION + ("+history-real" if args.history == "real" else "")
    seeds = [args.seed_start + i for i in range(args.episodes)]
    key = dict(task=task, text=text, prior=str(prior_dir), seed_start=args.seed_start, protocol=version)
    episodes: list[dict] = []
    if partial.exists():
        prev = json.loads(partial.read_text())
        if prev.get("key") != key:
            raise SystemExit(f"{partial} belongs to another run ({prev.get('key')}); remove it or choose another --out")
        episodes = prev["episodes"][: args.episodes]
        print(f"[prior] resuming after {len(episodes)} finished episodes from {partial.name}", flush=True)
    resumed = len(episodes)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    evidence_train = train_text_evidence(task, prior_dir, dataset_root) if text_mode == "train" else None
    prior = build_prior(task, prior_dir)
    override = TextOverride(prior, text)
    env = build_env(task)
    check_history = args.history == "real" or args.check_history
    if check_history:
        install_history(prior, env, args.history)
    horizon = int(env._max_episode_steps)
    env_desc = getattr(env, "task_description", None)
    print(f"[prior] {task}: text ({text_mode}) = {text!r}; history {args.history}; prior {prior_dir}; {args.episodes} "
          f"episodes, env seeds {seeds[0]}-{seeds[-1]}; action_steps {prior.action_steps}, inference_steps "
          f"{prior.inference_steps}, horizon {horizon}; cuDNN SDPA "
          f"{'on' if torch.backends.cuda.cudnn_sdp_enabled() else 'off'}", flush=True)
    t_setup = time.time() - t_start
    try:
        for i in range(len(episodes), args.episodes):
            te = time.time()
            e = run_episode(prior, env, seeds[i])
            e["seconds"] = round(time.time() - te, 2)
            episodes.append(e)
            sr = 100.0 * np.mean([x["success"] for x in episodes])
            print(f"ep {i} seed {e['seed']}: {'success' if e['success'] else 'fail'} len {e['length']}  running {sr:.1f}% "
                  f"of {len(episodes)} [{e['seconds']:.0f}s: reset {e['reset_seconds']:.0f}s, prior {e['prior_seconds']:.0f}s, "
                  f"env {e['env_seconds']:.0f}s; total {time.time() - t_start:.0f}s, peak RSS {peak_rss_gib():.2f} GiB]",
                  flush=True)
            atomic_write(partial, dict(key=key, episodes=episodes))
    finally:
        env.close()
    succ = [x["success"] for x in episodes]
    native = json.loads((prior_dir / "config.json").read_text())
    res = dict(
        task=task, text_mode=text_mode, text=text, task_id=task, env_task_description=env_desc,
        episodes=len(episodes), episodes_resumed=resumed, success_rate=100.0 * float(np.mean(succ)),
        n_success=int(sum(succ)),
        successes=succ, lengths=[x["length"] for x in episodes], env_seeds=[x["seed"] for x in episodes],
        seed_start=args.seed_start, avg_length=float(np.mean([x["length"] for x in episodes])),
        per_episode=episodes,
        protocol=dict(
            version=version, name="frozen DiT prior, one draw per chunk (base prior, no RL)",
            decision="train_online.flatten_chunk(FrozenPolicyPrior.action_chunk_samples(obs, 1)), executed by "
                     "train_online.rollout_chunk", draws_per_chunk=1, best_of_n=None, residual=None,
            action_steps=prior.action_steps, inference_steps=prior.inference_steps, prior_temperature=PRIOR_TEMPERATURE,
            episode_horizon=horizon, episode_horizon_source="TaskSpec (make_env horizon=None)",
            env=dict(factory="train_online.make_env", cls=type(env).__name__, reward_shaping=False, insertion=None,
                     holder_sigma=0.0, fixed_goal=False, fixed_goal_rot=False, full_fixed=False, jitter_goal_m=0.0,
                     observation_hw=224, pixel_cameras=list(T.PIXEL_CAMERAS)),
            seeding="episode i: s = seed_start + i; torch.manual_seed(s); np.random.seed(s); random.seed(s); "
                    "env.reset(seed=s)",
            success="any primitive step with info['is_success'] (env terminates on success, truncates at the horizon)",
            step_penalty=STEP_PENALTY, cudnn_sdp_enabled=bool(torch.backends.cuda.cudnn_sdp_enabled()),
            history=args.history,
            history_note=("real: window [o_{t-1}, o_t] from the 2nd chunk of an episode on (o_{t-1} = the env observation "
                          "one primitive step before the current one); first chunk [o_0, o_0]" if args.history == "real"
                          else "dup: window [o_t, o_t] (policy.reset before every chunk), the as-run behaviour")),
        prior_checkpoint=str(prior_dir), prior_checkpoint_resolved=str(prior_dir.resolve()),
        prior_native=dict(type=prior.cfg.type, objective=native.get("objective"),
                          n_action_steps=native.get("n_action_steps"),
                          num_integration_steps=native.get("num_integration_steps"),
                          horizon=prior.horizon, n_obs_steps=native.get("n_obs_steps"),
                          state_dims_used=prior.prior_state_dim or int(m["TASKS"][task]["state_dim"])),
        prior_files_md5={f: md5_file(prior_dir / f) for f in ("config.json", "policy_preprocessor.json",
                                                              "model.safetensors") if (prior_dir / f).exists()},
        text_check=override.evidence(), train_text_source=evidence_train,
        history_check=(dict(chunks=H["chunks"], chunks_with_prev=H["chunks_with_prev"], queue_checks_passed=H["checked"],
                            mean_abs_state_gap_prev_vs_cur=(H["state_gap_sum"] / H["chunks_with_prev"]
                                                            if H["chunks_with_prev"] else None))
                       if check_history else None),
        code_md5=code_md5s(), script_md5=md5_file(Path(__file__).resolve()),
        resources=dict(peak_rss_gib=round(peak_rss_gib(), 3),
                       cuda_max_allocated_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                       cuda_max_reserved_gib=round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
                       gpu_name=torch.cuda.get_device_name()),
        timing=dict(seconds=round(time.time() - t_start, 1), setup_seconds=round(t_setup, 1),
                    seconds_per_episode=round(float(np.mean([x["seconds"] for x in episodes])), 2),
                    prior_seconds_per_chunk=round(float(sum(x["prior_seconds"] for x in episodes)
                                                        / max(1, sum(x["chunks"] for x in episodes))), 4),
                    env_seconds_per_step=round(float(sum(x["env_seconds"] for x in episodes)
                                                     / max(1, sum(x["length"] for x in episodes))), 4)),
        started=started, finished=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        argv=sys.argv)
    override.restore()
    prior.close()
    with open(out, "x") as f:                                       # never clobbers a file that appeared meanwhile
        f.write(json.dumps(res, indent=1) + "\n")
    if partial.exists():
        partial.unlink()
    print(f"[prior] {task} text {text_mode} {text!r}, history {args.history}: {res['success_rate']:.1f}% over "
          f"{len(episodes)} episodes; {override.calls} prior calls, all with tokens of {text!r}"
          + (f"; {H['checked']} window checks passed ({H['chunks_with_prev']} chunks with the real previous frame)"
             if check_history else "")
          + f"; peak RSS {res['resources']['peak_rss_gib']:.2f} GiB; {res['timing']['seconds'] / 60:.1f} min -> {out}",
          flush=True)


def compare(a_path: str, b_path: str) -> int:
    """Paired comparison of two finished runs of this script on their shared seeds (exact McNemar test). Refuses pairs
    that are not the same prior under the same protocol; text and history are what may differ."""
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    keys = ("action_steps", "inference_steps", "prior_temperature", "episode_horizon", "draws_per_chunk", "step_penalty",
            "cudnn_sdp_enabled")
    bad = [k for k in keys if a["protocol"].get(k) != b["protocol"].get(k)]
    if a["task"] != b["task"]:
        bad.append("task")
    ma, mb = (r.get("prior_files_md5", {}).get("model.safetensors") for r in (a, b))
    if (ma != mb) if (ma and mb) else (a["prior_checkpoint"] != b["prior_checkpoint"]):
        bad.append("prior")
    if bad:
        print(f"NOT PAIRED ({', '.join(bad)} differ): {a_path} vs {b_path}")
        return 2
    r = compare_evals.compare(compare_evals.load(a_path), compare_evals.load(b_path))
    la, lb = (f"text {x['text_mode']} / history {x['protocol'].get('history', 'dup')}" for x in (a, b))
    print(f"{a['task']}: [{la}] {r['rate_a']:.1f}% vs [{lb}] {r['rate_b']:.1f}% (diff {r['rate_b'] - r['rate_a']:+.1f} "
          f"pts over {r['shared']} paired seeds); identical episodes {r['same_success_and_length']}/{r['shared']}; "
          f"discordant: {r['only_a']} only-first, {r['only_b']} only-second; exact McNemar p = {r['mcnemar_p']:.3g}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=TASKS_EVAL)
    ap.add_argument("--text", default="taskid", help="taskid (as run) | train | spec | <literal string>")
    ap.add_argument("--history", choices=("dup", "real"), default="dup",
                    help="dup (as run): the current frame twice; real: the true previous frame")
    ap.add_argument("--check-history", action="store_true",
                    help="with --history dup: also verify on every chunk that the prior's window holds [o_t, o_t]")
    ap.add_argument("--out")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=10_000)
    ap.add_argument("--prior", default="", help="pretrained_model dir (default: the latest checkpoint below "
                                                "$CKPT_ROOT/priors/<the task's prior>)")
    ap.add_argument("--dataset-root", default="", help="the task's demo dataset, read only by --text train "
                                                       "(default $DATA_ROOT/mimicgen_<task>_224x224)")
    ap.add_argument("--compare", nargs=2, metavar="JSON", help="paired comparison of two finished runs")
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    if not args.task or not args.out:
        ap.error("--task and --out are required")
    if args.episodes < 1:
        ap.error("--episodes must be >= 1")
    evaluate(args)


if __name__ == "__main__":
    main()
