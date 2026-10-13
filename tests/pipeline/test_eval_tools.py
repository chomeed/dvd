"""CPU test of the evaluators' plumbing (eval/compare_evals.py, eval/eval_prior.py, eval/eval_bon16.py and
online_rl_mimicgen.baselines.eval_dsrl_ckpt): result-file comparison, the prior's text table and as-run defaults,
loading a checkpoint of the pre-rename research code with its asset paths resolved on this machine, and how the DSRL
evaluator finds a run's config.json (symlinked checkpoint, --config). No environment is stepped and no prior is loaded.

With LEGACY_RUNS set (run directories of the research code), the last checkpoint of one DVD run and one Q run is loaded
and its scorer built on CPU with the trainer's own classes -- the step that precedes every best-of-16 episode.
Run: python tests/pipeline/test_eval_tools.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import _pipeline_util as U

U.use_repo("eval")

import compare_evals as CE  # noqa: E402


def _write(d, name, obj):
    p = Path(d) / name
    p.write_text(json.dumps(obj))
    return p


def test_compare_evals():
    assert CE.mcnemar_exact(0, 0) == 1.0 and abs(CE.mcnemar_exact(0, 5) - 2 / 32) < 1e-12
    assert abs(CE.mcnemar_exact(13, 18) - 0.4731) < 1e-3          # the square_d1 text comparison: p = 0.473
    with tempfile.TemporaryDirectory() as d:
        bon = _write(d, "bon.json", dict(run="r", ckpt="600000.pt", strategy="bon16", task="square_d0", seed_start=10000,
                                         per_episode=[True, False, True, True], per_episode_length=[100, 400, 120, 90]))
        # eval_prior.py files ALSO carry a per_episode key (a list of dicts): the outcome lists must win
        prior = _write(d, "prior.json", dict(task="square_d0", text_mode="taskid", protocol=dict(history="dup"),
                                             successes=[True, False, False], lengths=[100, 400, 400],
                                             env_seeds=[10000, 10001, 10002], per_episode=[dict(seed=10000)] * 3))
        dsrl = _write(d, "dsrl.json", dict(run="d", ckpt="x/0600000.zip", seed_start=10002, successes=[True, True],
                                           lengths=[120, 90]))
        a, b, c = CE.load(bon), CE.load(prior), CE.load(dsrl)
        assert a["episodes"][10001] == (False, 400) and b["episodes"][10002] == (False, 400) and c["episodes"][10003] == (True, 90)
        r = CE.compare(a, b)
        assert (r["shared"], r["same_success"], r["same_success_and_length"], r["differing_seeds"]) == (3, 2, 2, [10002])
        assert (r["only_a"], r["only_b"]) == (1, 0)
        r = CE.compare(a, c)
        assert (r["shared"], r["same_success_and_length"], r["differing_seeds"]) == (2, 2, [])
        assert CE.main([str(bon), str(dsrl), "--require-identical"]) == 0
        assert CE.main([str(bon), str(prior), "--require-identical"]) == 1
    print("compare_evals: reads best-of-16, prior and DSRL result files; pairs on shared env seeds; "
          "--require-identical fails on one differing episode")


def test_prior_text_table():
    import eval_prior as P
    from lerobot_env_mimicgen.environment import normalize_task
    assert (P.ACTION_STEPS, P.INFERENCE_STEPS, P.PRIOR_TEMPERATURE, P.STEP_PENALTY) == (8, 10, 1.0, 0.0)
    assert set(P.TRAIN_TEXT) == set(P.TASKS_EVAL) == set(U.TASKS)
    differs = []
    for task in U.TASKS:
        assert P.resolve_text(task, "taskid") == ("taskid", task), "the as-run text is the task id"
        assert P.resolve_text(task, "train") == ("train", P.TRAIN_TEXT[task])
        spec = normalize_task(task).instruction
        assert P.resolve_text(task, "spec") == ("spec", spec)
        assert P.resolve_text(task, "Do the thing.") == ("literal", "Do the thing.")
        if spec != P.TRAIN_TEXT[task]:
            differs.append(task)
    assert sorted(differs) == ["hammer_cleanup_d1", "square_d0", "square_d1"], differs
    r = subprocess.run([sys.executable, str(U.REPO / "eval/eval_prior.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--history {dup,real}" in r.stdout
    ap_defaults = dict(text="taskid", history="dup", episodes=100, seed_start=10_000)
    src = (U.REPO / "eval/eval_prior.py").read_text()
    for k, v in ap_defaults.items():
        flag = "--" + k.replace("_", "-")
        line = next(ln for ln in src.splitlines() if f'add_argument("{flag}"' in ln)
        assert f"default={v!r}".replace("'", '"') in line.replace("10_000", "10000"), (flag, line)
    print(f"prior text: as-run default = task id; training sentence != TaskSpec.instruction on {sorted(differs)}; "
          f"defaults text=taskid history=dup, 100 episodes from seed 10000")


def _fake_legacy_checkpoint(path, bon_source_new="dvd", extra_keys=(), extra_cfg=None):
    """A checkpoint payload spelled the way the pre-rename code wrote it (names built from the legacy tables)."""
    import torch
    from online_rl_mimicgen import legacy as L
    old_key = {new: old for old, new in L.KEY_RENAMES.items()}
    old_val = {new: old for old, new in L.VALUE_RENAMES["bon_source"].items()}
    cfg = {"task": "square_d0", "method": "dice_rl", "bon_source": old_val.get(bon_source_new, bon_source_new),
           old_key["dvd_weight"]: 1.0, old_key["dvd_dynamics"]: "dino_wm",
           "base_checkpoint": "/another/machine/priors/square_d0_40k_rs9/checkpoints/040000/pretrained_model",
           "value_model": "/another/machine/v/v_dino_x", "dino_wm_ckpt": "/another/machine/wm/wm.pt"}
    cfg.update(dict.fromkeys(extra_keys, 1))
    cfg.update(extra_cfg or {})
    extra = {next(iter(L.EXTRA_RENAMES)): {"w": torch.zeros(1)}} if bon_source_new == "dvd" else {}
    torch.save({"config": cfg, "extra": extra, "actor": {}, "critics": {}, "env_steps": 7}, path)


def test_load_payload_legacy_and_overrides():
    import eval_bon16 as E
    from online_rl_mimicgen import legacy as L
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "000001.pt"
        _fake_legacy_checkpoint(ck)
        over = {"base_checkpoint": f"{d}/prior", "value_model": f"{d}/v", "dino_wm_ckpt": f"{d}/wm.pt"}
        try:                              # an explicit override that does not exist is named, not passed on to a loader
            E.load_payload(ck, over, verbose=False)
            raise AssertionError("a missing override was accepted")
        except FileNotFoundError as e:
            assert "override" in str(e) and f"{d}/prior" in str(e), e
        Path(over["base_checkpoint"]).mkdir(); Path(over["value_model"]).mkdir(); Path(over["dino_wm_ckpt"]).touch()
        payload, was_legacy, assets = E.load_payload(ck, over, verbose=False)
        cfg = payload["config"]
        assert was_legacy and cfg["bon_source"] == "dvd" and cfg["dvd_dynamics"] == "dino_wm" and cfg["dvd_weight"] == 1.0
        assert not any(L.OLD in k for k in cfg) and "dvd_dynamics" in payload["extra"]
        assert assets == over and all(cfg[k] == v for k, v in over.items()), "explicit overrides win"
        try:                              # nothing at the stored paths, nothing under the roots, no override: refuse loudly
            E.load_payload(ck, {}, verbose=False)
            raise AssertionError("missing assets were accepted")
        except FileNotFoundError as e:
            assert "does not exist here" in str(e)
        q = Path(d) / "000002.pt"
        _fake_legacy_checkpoint(q, bon_source_new="q")
        payload, _, assets = E.load_payload(q, {"base_checkpoint": f"{d}/prior"}, verbose=False)
        assert list(assets) == ["base_checkpoint"], "a Q checkpoint opens neither the V directory nor the world model"
        try:
            E.load_payload(q, {"base_checkpoint": f"{d}/prior", "value_model": f"{d}/v"}, verbose=False)
            raise AssertionError("an override the checkpoint never reads was accepted")
        except SystemExit as e:
            assert "does not read" in str(e)
        bad = Path(d) / "000003.pt"
        _fake_legacy_checkpoint(bad, extra_keys=("a_flag_this_code_never_had",))
        payload, _, _ = E.load_payload(bad, over, verbose=False)
        import torch
        try:
            E.build(payload, torch.device("cpu"), verbose=False)
            raise AssertionError("unknown config keys were dropped silently")
        except SystemExit as e:
            assert "a_flag_this_code_never_had" in str(e) and "--allow-dropped-config-keys" in str(e)
        deep = Path(d) / "000004.pt"      # a two-step best-of-N checkpoint: this evaluator rebuilds depth 1 only
        _fake_legacy_checkpoint(deep, extra_cfg={"bon_lookahead": 2, "output_dir": str(Path(d) / "unused")})
        try:
            E.build(E.load_payload(deep, over, verbose=False)[0], torch.device("cpu"), verbose=False)
            raise AssertionError("a bon_lookahead 2 checkpoint would have been scored at depth 1")
        except SystemExit as e:
            assert "unsupported checkpoint" in str(e) and "bon_lookahead 2" in str(e), e
    print("eval_bon16.load_payload: pre-rename names upgraded; overrides win; absent assets, missing overrides and "
          "unused overrides are refused; build() refuses config keys it would drop and a depth-2 best-of-N checkpoint")


def test_eval_dsrl_config_lookup():
    """eval_dsrl_ckpt finds config.json two levels above the REAL checkpoint file (as the research code does); a
    checkpoint that is a link into another store gets a message that says so, and --config names the config directly.
    The module is run up to the point where it has read the config (the fake config is then refused by its method tag),
    so nothing is built. Needs the DSRL stable-baselines3 fork and the simulator imports; SKIP without them."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = os.pathsep.join([str(U.RL), str(U.REPO / "envs/lerobot_env_mimicgen/src")]
                                        + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])

    def run(*argv):
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "online_rl_mimicgen.baselines.eval_dsrl_ckpt", *argv],
                           cwd=U.REPO, env=env, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    rc, text = run("--help")
    if rc != 0:
        U.skip("online_rl_mimicgen.baselines.eval_dsrl_ckpt is not importable here: " + text.strip().splitlines()[-1][:160])
    assert "--config" in text and "--base-output" in text
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "store").mkdir(); (d / "store/blob").touch()
        (d / "run/checkpoints").mkdir(parents=True)
        (d / "run/checkpoints/0000001.zip").symlink_to(d / "store/blob")
        (d / "run/config.json").write_text(json.dumps({"method": "not_a_dsrl_dino_run"}))
        rc, text = run("--ckpt", str(d / "run/checkpoints/0000001.zip"), "--out", str(d / "out.json"))
        assert rc != 0 and "no config.json at" in text and "real files inside <run>/checkpoints/" in text and "--config" in text, text[-600:]
        assert "Traceback" not in text, text[-600:]
        rc, text = run("--ckpt", str(d / "run/checkpoints/0000001.zip"), "--out", str(d / "out.json"),
                       "--config", str(d / "run/config.json"))
        assert rc != 0 and f"{d / 'run'} is not a dsrl_dino run (method='not_a_dsrl_dino_run')" in text, text[-600:]
        rc, text = run("--ckpt", str(d / "run/checkpoints/0000001.zip"), "--out", str(d / "out.json"),
                       "--config", str(d / "nowhere/config.json"))
        assert rc != 0 and "no config.json at" in text and "nowhere" in text, text[-600:]
        # a real file in its run directory: the config beside it is found without --config (the default path)
        (d / "real/checkpoints").mkdir(parents=True); (d / "real/checkpoints/0000001.zip").touch()
        (d / "real/config.json").write_text(json.dumps({"method": "not_a_dsrl_dino_run"}))
        rc, text = run("--ckpt", str(d / "real/checkpoints/0000001.zip"), "--out", str(d / "out.json"))
        assert rc != 0 and "is not a dsrl_dino run" in text and "real" in text, text[-600:]
        assert not (d / "out.json").exists()
    print("eval_dsrl_ckpt: config.json is looked up beside the real checkpoint file; a linked checkpoint without one "
          "gets a message naming the rule and --config; --config is read when given")


def test_build_on_research_checkpoints():
    """Build the best-of-16 scorer of real research-code checkpoints on CPU (one DVD run, one Q run)."""
    import torch
    import eval_bon16 as E
    ckpts = U.legacy_checkpoints()
    if not ckpts:
        U.skip("LEGACY_RUNS is not set (directories of DICE-RL runs of the research code)")
    seen = {}
    for ck in ckpts:
        kind = "q" if "-q_" in ck.parent.parent.name else "dvd"
        if kind in seen:
            continue
        payload, was_legacy, assets = E.load_payload(ck, verbose=False)
        b = E.build(payload, torch.device("cpu"), verbose=False)
        info = b["info"]
        assert was_legacy and info["bon_source"] == kind and info["n_candidates"] == 16
        assert info["config_keys_not_in_trainconfig"] == []
        if kind == "dvd":
            assert info["loaded_from_checkpoint"] == ["value_fn", "attn_pool (mqattn)", "dvd_dynamics"], info
            assert b["dvd_term"] is not None and info["value_nets"] == 3 and info["state_head"]
            assert abs(info["discount_h"] - 0.99 ** 8) < 1e-9
        else:
            assert info["loaded_from_checkpoint"] == ["critics"] and b["critics"] is not None and b["dvd_term"] is None
        seen[kind] = ck
        print(f"built {kind} scorer of {ck.parent.parent.name}/{ck.name}: {info['scorer']}; assets {sorted(assets)}")
        del b, payload
    if len(seen) < 2:
        print(f"note: only {sorted(seen)} checkpoints among LEGACY_RUNS")


if __name__ == "__main__":
    U.main(globals())
