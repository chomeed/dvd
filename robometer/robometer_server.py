"""RoboMeter scoring sidecar -- runs in the ROBOMETER venv, not the trainer's. Imports nothing from this repository.

The RL code lives in the main environment (python 3.12, transformers 5.x); robometer needs its own
(python 3.10, transformers 4.57). Installing one into the other breaks whichever loses. So the 4B
teacher runs as a separate long-lived process, one per spool, and the two talk through a spool directory:

    <spool>/READY            written once the model is loaded (holds the pid)
    <spool>/req/<id>.npy     uint8 (T,224,224,3) agentview clip, oldest frame first, written by the client
    <spool>/res/<id>.npy     float32 (2,T) written back here: row 0 progress, row 1 success probability
                             ((1,T) if the model has no success head)
    <spool>/res/<id>.err     the exception text, if scoring failed

Atomic handoff both ways: everything is written to `.tmp` and renamed, so the reader can never
observe a partial file. The trainer polls with a timeout and simply skips the episode on failure --
a teacher outage must never take down an RL run. Clients: robometer/score_rollouts.py (offline V
targets) and online_rl_mimicgen.robometer_distill (online distillation, the "TD+distill" cells).

--task is the instruction the model is conditioned on and MUST be the sentence the LoRA was
fine-tuned with; it is required because a wrong sentence still returns plausible numbers.
robometer/serve.sh holds the model / spool / sentence of every teacher of the campaign.

Launch (from the robometer repo root, in its venv; or: SPOOL=<name> bash robometer/serve.sh):
    uv run --no-sync python <this repo>/robometer/robometer_server.py \\
        --model <LoRA checkpoint dir or hf id> --spool $SPOOL_ROOT/<name> --task "<training sentence>"
"""
from __future__ import annotations

import argparse, dataclasses, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch


def load(model_path: str, device):
    import yaml, tempfile
    from robometer.configs.experiment_configs import CustomEvaluationConfig
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator
    src = Path(model_path)
    path = model_path
    cfgp = src / "config.yaml"
    if cfgp.exists():
        c = yaml.safe_load(cfgp.read_text())
        ok = {f.name for f in dataclasses.fields(CustomEvaluationConfig)}
        ce = c.get("custom_eval", {})
        if set(ce) - ok:      # adapters from an older robometer carry keys the dataclass rejects
            dst = Path(tempfile.mkdtemp(prefix="rmsrv_"))
            for f in src.iterdir():
                if f.name != "config.yaml":
                    (dst / f.name).symlink_to(f)
            c["custom_eval"] = {k: v for k, v in ce.items() if k in ok}
            (dst / "config.yaml").write_text(yaml.safe_dump(c))
            path = str(dst)
    cfg, tok, proc, model = load_model_from_hf(model_path=path, device=device)
    model.eval()
    import collections
    if len(collections.Counter(p.dtype for p in model.parameters())) > 1:
        model = model.to(torch.bfloat16)          # base ships mixed dtype; LayerNorm rejects it
    col = setup_batch_collator(proc, tok, cfg, is_eval=True)
    disc = getattr(cfg.model, "progress_loss_type", "discrete") == "discrete"
    bins = int(getattr(cfg.model, "progress_discrete_bins", 10) or 10)
    return model, tok, col, disc, bins


@torch.no_grad()
def score(model, tok, col, disc, bins, frames, task, device):
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    t = Trajectory(frames=frames, frames_shape=tuple(frames.shape), task=task, id="0",
                   metadata={"subsequence_length": int(frames.shape[0])}, video_embeddings=None)
    batch = col([ProgressSample(trajectory=t, sample_type="progress")])
    mdt = next(model.parameters()).dtype
    inp = {}
    for k, v in batch["progress_inputs"].items():
        if hasattr(v, "to"):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(mdt)
        inp[k] = v
    # is_discrete_mode MUST match training: with False this returns raw per-bin logits, silently.
    out = compute_batch_outputs(model, tok, inp, sample_type="progress",
                                is_discrete_mode=disc, num_bins=bins)
    prog = np.asarray(out["progress_pred"][0], dtype=np.float32)
    # THE SUCCESS HEAD. The progress head was masked out on every failure trajectory during
    # training (should_compute_progress -> 0.0 for quality_label in {suboptimal, failure, failed},
    # data/collators/rbm_heads.py:87), so progress is a pure t/T ramp learned from successes ONLY
    # and cannot separate a timeout from a success. The success head DID train on failures -- BCE
    # with label 0 on every one of their frames -- so it is the head that carries outcome
    # information. compute_batch_outputs already computes it; returning only progress threw it away.
    succ = out.get("outputs_success", {}).get("success_probs")
    if succ is None:
        return prog[None, :]                      # (1, T): progress only, no success head
    succ = np.asarray(succ[0], dtype=np.float32).reshape(-1)
    if succ.shape[0] != prog.shape[0]:            # never silently broadcast a mismatched head
        return prog[None, :]
    return np.stack([prog, succ], axis=0)         # (2, T): row 0 progress, row 1 success_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spool", required=True)
    ap.add_argument("--task", required=True, help="the instruction the LoRA was fine-tuned with")
    ap.add_argument("--poll", type=float, default=0.25)
    a = ap.parse_args()
    spool = Path(a.spool); req = spool / "req"; res = spool / "res"
    req.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, col, disc, bins = load(a.model, device)
    (spool / "READY").write_text(f"{os.getpid()}\n")
    print(f"[rm-server] ready pid={os.getpid()} discrete={disc} bins={bins} spool={spool}", flush=True)
    n = 0
    while True:
        pending = sorted(p for p in req.glob("*.npy"))
        if not pending:
            time.sleep(a.poll); continue
        for p in pending:
            rid = p.stem
            try:
                frames = np.load(p)
                out = score(model, tok, col, disc, bins, frames, a.task, device)
                tmp = res / f"{rid}.tmp"          # np.save would append .npy to "*.npy.tmp"
                with open(tmp, "wb") as fh:
                    np.save(fh, out)
                tmp.rename(res / f"{rid}.npy")
                n += 1
                _p = out[0]
                _msg = f"progress {_p.min():.3f}..{_p.max():.3f}"
                if out.shape[0] > 1:
                    _s = out[1]; _msg += f"  success {_s.min():.3f}..{_s.max():.3f}"
                print(f"[rm-server] {rid}: {frames.shape[0]} frames -> {_msg}  (served {n})",
                      flush=True)
            except Exception:
                tb = traceback.format_exc()
                (res / f"{rid}.err").write_text(tb)
                # The client truncates the .err it reads and then deletes it, so the server log is
                # the only place a full traceback survives. Print it.
                print(f"[rm-server] {rid} FAILED\n{tb}", flush=True)
            finally:
                p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
