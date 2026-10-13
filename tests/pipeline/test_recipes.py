"""CPU test of the per-task recipes of the offline pipeline: the wrappers must print the commands that produced the
released world models, V directories and teachers, and no entry point may fall back to a default task.

  dinowm/train_wm.sh, value/fit_v.sh, robometer/serve.sh   --print output against the recipe table below (taken from
                                                           the metadata of the released artifacts) and, when the
                                                           artifacts are under $CKPT_ROOT, against their own json
  WS_TASK                                                  every offline entry point refuses an unset / unknown /
                                                           contradicted task before touching data
  online_rl_mimicgen.pixel_cache                           --help works; an unset DATASET is a one-line usage error
  hygiene                                                  no machine-specific path, user name, address or credential
                                                           anywhere in the repository outside reference/ (this file
                                                           included); HYGIENE_EXTRA_WORDS=<word>,<word> adds forbidden
                                                           words, e.g. the reference machine's root and user name
Run: python tests/pipeline/test_recipes.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import _pipeline_util as U

REPO = U.REPO
# task -> (steps, base tag, extra arguments of the base run, data seed of the continuation)
WM = {"square_d0": (150000, "ph_conv_150k", [], 1),
      "square_d1": (150000, "ph_conv_150k", ["--snapshot-steps", "44000,80000"], 0),
      "threading_d0": (168000, "ph_conv_168k", [], 1),
      "hammer_cleanup_d1": (217000, "ph_conv_217k", [], 1)}
# teacher -> (task, LoRA, sentence)
TEACHERS = {
    "threading_d0": ("threading_d0", "robometer-4b-lora-threading-d0-seg",
                     "Pick up the needle and thread it through the hole in the tripod."),
    "square_d0": ("square_d0", "robometer-4b-lora-square-d0-seg", "Pick up the square nut and place it on the peg."),
    "hammer_cleanup_d1": ("hammer_cleanup_d1", "robometer-4b-lora-hammer-cleanup-seg",
                          "Open the drawer, place the hammer inside, and close the drawer."),
    "square_d1": ("square_d1", "robometer-4b-lora-square-d1", "Pick up the square nut and place it on the peg."),
    "square_d1_seg": ("square_d1", "robometer-4b-lora-square-d1-seg-c095",
                      "Pick up the square nut and place it on the peg."),
}
# (task, recipe) -> (score table, teacher, suffix, output prefix)
V = {(t, "pv4"): (t, t, "mqattn_pv4", "v_dino_robometer_pred_progress") for t in U.TASKS}
V.update({(t, "mc"): (t, t, "mqattn_pv4", "v_dino_mc_return") for t in U.TASKS})
V[("square_d1", "pv4_seg")] = ("square_d1_seg", "square_d1_seg", "mqattn_pv4_seg", "v_dino_robometer_pred_progress")
V[("threading_d0", "pv4_seg")] = ("threading_d0_seg", "threading_d0", "mqattn_pv4_seg", "v_dino_robometer_pred_progress")


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("WS_TASK", "TASK", "STAGE", "RECIPE", "SPOOL", "MODEL", "G")}
    env.update(CKPT_ROOT="/assets/ckpt", CACHE_ROOT="/assets/cache", SPOOL_ROOT="/assets/spool",
               ROBOMETER_ROOT="/opt/robometer", PIXEL_IMG="224", CUDA_VISIBLE_DEVICES="")
    env.update(extra)
    return env


def _printed(script, **extra):
    r = subprocess.run(["bash", str(REPO / script), "--print"], env=_env(**extra), capture_output=True, text=True)
    assert r.returncode == 0, (script, extra, r.stderr)
    return r.stdout.splitlines()


def _opts(tokens):
    """argv tokens -> {option: value or True}"""
    out, i = {}, 0
    while i < len(tokens):
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[tokens[i]] = tokens[i + 1]; i += 2
        else:
            out[tokens[i]] = True; i += 1
    return out


def test_train_wm_commands():
    for task, (steps, tag, base_extra, seed) in WM.items():
        k = steps // 1000
        head = [f"WS_TASK={task}", "python", "dinowm/train_wm_ph.py", "--task", task, "--act", "conv", "--steps", str(steps)]
        base = _printed("dinowm/train_wm.sh", TASK=task, STAGE="base")
        assert base == head + base_extra + ["--resume"], base
        cont = _printed("dinowm/train_wm.sh", TASK=task, STAGE="cont")
        init = f"/assets/ckpt/wm/mimicgen_{task}_224x224_dinowm_{tag}/wm_dinov2s14_{tag}.pt"
        assert cont == head + ["--init", init, "--lr", "1e-4", "--seed", str(seed), "--tag-suffix", f"_cont{k}k",
                               "--snapshot-steps", "50000,100000", "--resume"], cont
        print(f"train_wm {task}: base {steps} steps{' ' + ' '.join(base_extra) if base_extra else ''}; continuation "
              f"+{steps} from {tag}, data seed {seed}")
    r = subprocess.run(["bash", str(REPO / "dinowm/train_wm.sh"), "--print"], env=_env(STAGE="base"), capture_output=True)
    assert r.returncode != 0, "train_wm.sh ran without TASK"


def test_train_wm_matches_released_metadata():
    """The json beside each released world model records how it was trained; the wrapper must agree with it."""
    checked = 0
    for task, (steps, tag, base_extra, seed) in WM.items():
        k = steps // 1000
        for stage, t in (("base", tag), ("cont", f"{tag}_cont{k}k")):
            meta_p = U.wm_file(task, t).with_suffix(".json")
            if not meta_p.exists():
                continue
            m = json.loads(meta_p.read_text())
            o = _opts(_printed("dinowm/train_wm.sh", TASK=task, STAGE=stage)[3:])
            assert m["tag"] == t and m["action_encoder"] == o["--act"] and m["history"] == 2 and m["state_head"] is True
            assert m["dino_wm_arch"] == "6,6,2048" and m["trunk"] == "facebook/dinov2-small" and m["dino_size"] == 196
            assert int(m.get("schedule_steps") or m["steps"]) == int(o["--steps"]) == steps, (task, stage)
            if stage == "cont":
                assert m["init_tag"] == tag and m["init_steps"] == steps and m["tag_suffix"] == o["--tag-suffix"]
                assert abs(m["lr"] - float(o["--lr"])) < 1e-12 and m["seed"] == int(o["--seed"]) == seed
                assert m["snapshot_steps"] == [int(x) for x in o["--snapshot-steps"].split(",")]
            else:                       # base runs: trainer defaults (peak LR 5e-4, seed 0) unless recorded otherwise
                assert m.get("lr", 5e-4) == 5e-4 and m.get("seed", 0) == 0
                assert m.get("snapshot_steps", []) == ([int(x) for x in o["--snapshot-steps"].split(",")]
                                                       if "--snapshot-steps" in o else [])
            skill = m["final_val"]["clean"]["skill"]
            print(f"released {task} {t}: {m.get('steps')} steps, lr {m.get('lr', 5e-4)}, seed {m.get('seed', 0)}; "
                  f"held-out clean skill {skill:.3f}")
            checked += 1
    if not checked:
        U.skip("no released world-model json below $CKPT_ROOT/wm")


def test_fit_v_commands():
    for (task, recipe), (scores, teacher, suffix, prefix) in V.items():
        tag = U.BASE_TAG[task]
        got = _printed("value/fit_v.sh", TASK=task, RECIPE=recipe)
        want = [f"WS_TASK={task}", "python", "value/fit_v_mqattn.py", "--task", task, "--wm",
                f"/assets/ckpt/wm/mimicgen_{task}_224x224_dinowm_{tag}/wm_dinov2s14_{tag}.pt", "--tag", tag,
                "--per-view", "--queries", "4", "--out-dim", "768", "--suffix", suffix,
                "--scores", f"/assets/ckpt/rm_scores/{scores}/predictions.parquet", "--teacher", TEACHERS[teacher][1]]
        assert got == want + (["--target", "mc"] if recipe == "mc" else []), (task, recipe, got)
        print(f"fit_v {task} {recipe}: rows rm_scores/{scores}, teacher {TEACHERS[teacher][1]} -> "
              f"{prefix}_{task}_{tag}_{suffix}")
    for task in ("square_d0", "hammer_cleanup_d1"):          # their first table already comes from the segmented LoRA
        r = subprocess.run(["bash", str(REPO / "value/fit_v.sh"), "--print"], env=_env(TASK=task, RECIPE="pv4_seg"),
                           capture_output=True)
        assert r.returncode != 0, f"pv4_seg accepted for {task}"


def test_fit_v_matches_released_metadata():
    checked = 0
    for (task, recipe), (scores, teacher, suffix, prefix) in V.items():
        tag = U.BASE_TAG[task]
        cfg_p = U.root("CKPT_ROOT") / "value" / f"{prefix}_{task}_{tag}_{suffix}" / "config.json"
        if not cfg_p.exists():
            continue
        c = json.loads(cfg_p.read_text())
        o = _opts(_printed("value/fit_v.sh", TASK=task, RECIPE=recipe)[3:])
        assert c["dino_pool"] == "mqattn" and c["mqattn_queries"] == int(o["--queries"]) == 4
        assert c["mqattn_per_view"] is True and c["mqattn_out_dim"] == int(o["--out-dim"]) == 768
        assert c["input_dim"] == 777 and c["hidden"] == [256, 256, 256] and int(c["seeds"]) == 3
        assert c.get("fit_steps", 20000) == 20000 and c["task"] == task
        assert Path(c["world_model"]).name == Path(o["--wm"]).name, "every V was fitted through the BASE world model"
        assert ("mc_success_return" in c["estimator"]) == (recipe == "mc"), (task, recipe, c["estimator"])
        files = sorted(p.name for p in cfg_p.parent.rglob("*") if p.is_file())
        assert files == ["config.json", "mqattn_pool.pt", "state_scaler.json", "v_td_seed0.pt", "v_td_seed1.pt",
                         "v_td_seed2.pt"], files
        sp = [c["metrics"][f"seed{i}"]["actual"]["spearman"] for i in range(3)]
        print(f"released V {cfg_p.parent.name}: held-out spearman vs its target {min(sp):.3f}..{max(sp):.3f}")
        checked += 1
    if not checked:
        U.skip("no released V directory below $CKPT_ROOT/value")


def test_teacher_table():
    sys.path.insert(0, str(REPO / "eval"))
    import eval_prior
    for name, (task, lora, text) in TEACHERS.items():
        got = _printed("robometer/serve.sh", SPOOL=name)
        assert got == ["uv", "run", "--no-sync", "python", str(REPO / "robometer/robometer_server.py"), "--model",
                       f"/assets/ckpt/robometer/{lora}/", "--spool", f"/assets/spool/{name}", "--task", text], got
        assert text == eval_prior.TRAIN_TEXT[task], "a teacher is served with the sentence its task was trained with"
        print(f"teacher {name}: {lora}  '{text}'")
    r = subprocess.run(["bash", str(REPO / "robometer/serve.sh"), "--print"], env=_env(SPOOL="threading"), capture_output=True)
    assert r.returncode != 0, "unknown teacher accepted"
    src = (REPO / "robometer/robometer_server.py").read_text()
    assert '"--task", required=True' in src, "the server must not default the task sentence"


def test_no_default_task():
    """Every offline entry point stops, before reading data, when WS_TASK is unset, unknown or contradicted."""
    py = sys.executable
    cases = [
        (["dinowm/build_cache.py"], None, "no default task"),
        (["dinowm/build_cache.py"], "threading", "no default task"),
        (["dinowm/build_cache.py", "--task", "square_d0"], "square_d1", "make them agree"),
        (["dinowm/build_cache.py", "--flow", "mimicgen_square_d1_224x224_mtdit_flow_45k"], "square_d0", "is not a square_d0 rollout set"),
        (["dinowm/train_wm_ph.py", "--act", "conv"], None, "no default task"),
        (["dinowm/train_wm_ph.py", "--act", "conv", "--task", "square_d0"], "hammer_cleanup_d1", "make them agree"),
        (["robometer/score_rollouts.py", "--spool", "/nonexistent", "--out", "/nonexistent/x.parquet"], None, "no default task"),
        (["value/fit_v_mqattn.py", "--task", "square_d0", "--wm", "x", "--tag", "t", "--scores", "s", "--suffix", "u"],
         None, "no default task"),
        (["value/fit_v_mqattn.py", "--task", "square_d0", "--wm", "x", "--tag", "t", "--scores", "s", "--suffix", "u"],
         "threading_d0", "make them agree"),
    ]
    for argv, ws, needle in cases:
        env = _env(DATA_ROOT="/nonexistent/data")
        if ws is not None:
            env["WS_TASK"] = ws
        r = subprocess.run([py, "-W", "ignore", str(REPO / argv[0]), *argv[1:]], env=env, capture_output=True, text=True)
        assert r.returncode != 0 and needle in (r.stderr + r.stdout), (argv, ws, r.returncode, r.stderr[-400:])
        print(f"refused: WS_TASK={ws!r} {' '.join(argv)[:70]} -> ...{needle}")


def test_pixel_cache_usage():
    """python -m online_rl_mimicgen.pixel_cache keeps its two-environment-variable interface, but --help works and a
    missing DATASET or a stray argument ends in a one-line usage instead of a traceback. Nothing is decoded."""
    env = {k: v for k, v in _env(PYTHONDONTWRITEBYTECODE="1").items() if k not in ("DATASET", "OUT")}

    def run(*argv, **extra):
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "online_rl_mimicgen.pixel_cache", *argv],
                           cwd=str(U.RL), env={**env, **extra}, capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    rc, text = run("--help")
    assert rc == 0 and text.startswith("usage: DATASET=") and "OUT" in text and "PIXEL_IMG" in text, (rc, text[-300:])
    for argv, extra, needle in (((), {}, "DATASET is not set"), (("--dataset", "x"), {"DATASET": "/nonexistent/ds"}, "no arguments")):
        rc, text = run(*argv, **extra)
        assert rc != 0 and "Traceback" not in text and len(text.splitlines()) == 1, (argv, rc, text[-400:])
        assert text.startswith("usage: DATASET=") and needle in text, (argv, text)
    print("pixel_cache: --help prints the usage; no DATASET / a stray argument -> one usage line, non-zero exit, no traceback")


# ---- hygiene ----------------------------------------------------------------------------------------------------------
# This file is scanned too, so it spells none of the things it forbids: only generic shapes are written down. What is
# specific to the machine a checkout was made on is found in two ways that need no name in the source:
#   * an absolute or ~/ path literal that EXISTS on the machine running the test (outside the system directories)
#   * the words of HYGIENE_EXTRA_WORDS (comma separated, matched as substrings): set it on the reference machine to its
#     root directory name, Unix user name and project code
_HYGIENE_PATTERNS = (
    ("home directory", r"(?<![\w$}.<>)~-])/(?:home|Users)/(?!user\b|username\b|you\b|me\b|name\b)[A-Za-z0-9_]"),
    ("e-mail address", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|io|ai|kr|jp|de|uk)\b"),
    ("access token", r"\b(?:hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}|glpat-[\w-]{20,}"
                     r"|sk-[A-Za-z0-9_-]{24,}|xox[abprs]-[\w-]{10,}|AKIA[0-9A-Z]{16})"),
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("wandb key", r"WANDB_API_KEY\s*[=:]\s*['\"]?[0-9a-f]{40}"),
)
_HYGIENE_ALLOWED = ("git" + "@" + "github.com",)                     # the SSH URL prefix of a git remote, not an address
_ABS_PATH = re.compile(r"(?<![\w$}.<>)~:/-])(~?)/([A-Za-z0-9_][\w.+-]*)/([\w.+-]+)")
_SYSTEM_TOPS = frozenset("usr bin sbin lib lib32 lib64 etc dev proc sys tmp var run boot opt".split())
# never scanned: the verbatim provenance snapshots (they are the reference machine's files), git's own directory, and
# the default asset / output / checkout locations of scripts/env.sh, which hold data and other people's code
_HYGIENE_SKIP_DIRS = frozenset((".git", "reference", "assets", "runs", "third_party", "wandb", "logs", "__pycache__",
                                ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache"))
_HYGIENE_SKIP_SUFFIXES = frozenset((".pyc", ".pt", ".pth", ".ckpt", ".safetensors", ".zip", ".npz", ".npy", ".parquet",
                                    ".mp4", ".pkl", ".log", ".png", ".jpg", ".pdf", ".h5", ".hdf5"))
_HYGIENE_SKIP_NAMES = frozenset((".netrc", ".env"))                  # git-ignored credential files


def _local_path(match) -> bool:
    """True if a path literal (~?)/<a>/<b> names something that exists on this machine outside the system directories."""
    tilde, top, second = match.groups()
    if tilde:
        return not top.startswith(".") and (Path.home() / top).exists()
    return top not in _SYSTEM_TOPS and Path("/", top, second).exists()


def hygiene_hits(line: str, extra_words=()) -> list[str]:
    """What is machine-specific or secret in one line of text: the kinds found, [] for a clean line."""
    for allowed in _HYGIENE_ALLOWED:
        line = line.replace(allowed, "")
    kinds = [kind for kind, pattern in _HYGIENE_PATTERNS if re.search(pattern, line)]
    if any(_local_path(m) for m in _ABS_PATH.finditer(line)):
        kinds.append("path that exists on this machine")
    kinds += [f"word {word!r}" for word in extra_words if word in line]
    return kinds


def hygiene_files() -> list[Path]:
    """Every text file of the repository outside the directories named in _HYGIENE_SKIP_DIRS."""
    out = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = sorted(d for d in dirnames if d not in _HYGIENE_SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath, name)
            if name in _HYGIENE_SKIP_NAMES or path.suffix in _HYGIENE_SKIP_SUFFIXES or path.is_symlink():
                continue
            out.append(path)
    return out


def test_no_machine_specific_strings():
    """No machine-specific path, user name, address or credential anywhere in the repository outside reference/
    (this file included). HYGIENE_EXTRA_WORDS adds forbidden words; see the comment above."""
    extra = tuple(w.strip() for w in os.environ.get("HYGIENE_EXTRA_WORDS", "").split(",") if w.strip())
    # the scanner must catch what it is for (probes assembled here, so that this file stays clean)
    probes = {"home directory": "/ho" + "me/" + "alice/data", "e-mail address": "first.last" + "@" + "example.org",
              "access token": "hf" + "_" + "a1B2" * 9, "private key": "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
              "wandb key": "WANDB_API" + "_KEY=" + "0f" * 20}
    for kind, text in probes.items():
        assert kind in hygiene_hits(f'x = "{text}"'), (kind, hygiene_hits(text))
    here = Path(__file__).resolve()
    if len(here.parts) > 3 and here.parts[1] not in _SYSTEM_TOPS:
        assert hygiene_hits(f"cd {here.parent}") == ["path that exists on this machine"], hygiene_hits(f"cd {here.parent}")
    assert hygiene_hits("see " + "zq" + "xj", ("zqxj",)) == ["word 'zqxj'"]
    clean = ("$CKPT_ROOT/value/<V dir>/config.json", "<run>/checkpoints/0600000.zip", 'f"{root}/data/chunk-000"',
             "/nonexistent-root-xyz/data", "git clone " + _HYGIENE_ALLOWED[0] + ":owner/repo.git", "a @ b.T", "/ho" + "me/user/dvd",
             "https://github.com/NVlabs/mimicgen", "#!/usr/bin/env bash", "sha256 " + "0f" * 32)
    for text in clean:
        assert not hygiene_hits(text), (text, hygiene_hits(text))

    files, hits = hygiene_files(), []
    assert Path(__file__).resolve() in [f.resolve() for f in files], "this test must scan itself"
    for path in files:
        if path.stat().st_size > 8 * 2**20:
            hits.append(f"{path.relative_to(REPO)}: {path.stat().st_size / 2**20:.0f} MB file in the repository")
            continue
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for kind in hygiene_hits(line, extra):
                hits.append(f"{path.relative_to(REPO)}:{i}: {kind}: {line.strip()[:100]}")
    print(f"scanned {len(files)} files (the whole repository outside reference/, .git and the asset / output "
          f"directories; this file included): {len(hits)} machine-specific strings; {len(extra)} extra word(s) from "
          f"HYGIENE_EXTRA_WORDS")
    assert not hits, hits[:10]


if __name__ == "__main__":
    U.main(globals())
