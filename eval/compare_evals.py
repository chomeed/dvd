"""Paired comparison of two evaluation result files on the env seeds they share.

Every evaluator in this repository seeds episode i with seed_start + i (env, torch, numpy, random), so two runs of the
same policy on the same seeds must agree episode for episode, and two policies can be compared paired.

  python eval/compare_evals.py <a.json> <b.json>                       agreement + exact McNemar test
  python eval/compare_evals.py <a.json> <b.json> --require-identical   exit 1 unless every shared episode has the same
                                                                       success AND the same length (a replay check)

Reads the files of eval/eval_bon16.py (per_episode, per_episode_length, seed_start), eval/eval_prior.py (successes,
lengths, env_seeds) and online_rl_mimicgen.baselines.eval_dsrl_ckpt (successes, lengths, seed_start). No torch needed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load(path) -> dict:
    """{seed: (success, length)} of one result file, plus a short label and the fields that describe what was run."""
    d = json.loads(Path(path).read_text())
    if "successes" in d:                                      # eval_prior.py, eval_dsrl_ckpt
        succ, lens = d["successes"], d.get("lengths")
        label = (f"prior {d.get('task')} text {d.get('text_mode')} history {d.get('protocol', {}).get('history', 'dup')}"
                 if "text_mode" in d else f"{d.get('run')} {Path(str(d.get('ckpt'))).name}")
    elif "per_episode" in d:                                  # eval_bon16.py (eval_prior.py's per_episode is a list of dicts)
        succ, lens = d["per_episode"], d.get("per_episode_length")
        label = f"{d.get('run')} {d.get('ckpt')} ({d.get('strategy')})"
    else:
        raise SystemExit(f"{path}: no per-episode outcomes (neither 'per_episode' nor 'successes')")
    seeds = d.get("env_seeds") or [int(d.get("seed_start", 10_000)) + i for i in range(len(succ))]
    if lens is None:
        lens = [None] * len(succ)
    assert len(seeds) == len(succ) == len(lens), f"{path}: ragged per-episode lists"
    assert all(isinstance(x, (bool, int)) for x in succ), f"{path}: per-episode outcomes are not booleans"
    return dict(path=str(path), label=label, task=d.get("task"), n=len(succ),
                rate=100.0 * sum(map(bool, succ)) / max(1, len(succ)),
                episodes={int(s): (bool(x), None if n is None else int(n)) for s, x, n in zip(seeds, succ, lens)})


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(a: dict, b: dict) -> dict:
    shared = sorted(set(a["episodes"]) & set(b["episodes"]))
    if not shared:
        raise SystemExit(f"no shared env seeds between {a['path']} and {b['path']}")
    pa, pb = [a["episodes"][s] for s in shared], [b["episodes"][s] for s in shared]
    only_a = sum(x[0] and not y[0] for x, y in zip(pa, pb))
    only_b = sum(y[0] and not x[0] for x, y in zip(pa, pb))
    have_len = all(x[1] is not None and y[1] is not None for x, y in zip(pa, pb))
    differing = [s for s, x, y in zip(shared, pa, pb) if x[0] != y[0] or (have_len and x[1] != y[1])]
    return dict(shared=len(shared), seeds=(shared[0], shared[-1]),
                rate_a=100.0 * sum(x[0] for x in pa) / len(shared), rate_b=100.0 * sum(y[0] for y in pb) / len(shared),
                same_success=sum(x[0] == y[0] for x, y in zip(pa, pb)),
                same_success_and_length=(sum(x == y for x, y in zip(pa, pb)) if have_len else None),
                differing_seeds=differing, only_a=only_a, only_b=only_b, mcnemar_p=mcnemar_exact(only_a, only_b))


def report(a: dict, b: dict, r: dict) -> str:
    both = "n/a (a file has no lengths)" if r["same_success_and_length"] is None else f"{r['same_success_and_length']}/{r['shared']}"
    return (f"A: {a['label']}  [{a['path']}]\nB: {b['label']}  [{b['path']}]\n"
            f"shared env seeds: {r['shared']} ({r['seeds'][0]}..{r['seeds'][1]}); success A {r['rate_a']:.1f}% vs B "
            f"{r['rate_b']:.1f}% (diff {r['rate_b'] - r['rate_a']:+.1f} pts)\n"
            f"identical episodes: success {r['same_success']}/{r['shared']}, success+length {both}"
            + (f"; differing seeds {r['differing_seeds']}" if r["differing_seeds"] else "") + "\n"
            f"discordant: {r['only_a']} only-A, {r['only_b']} only-B; exact McNemar p = {r['mcnemar_p']:.3g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--require-identical", action="store_true",
                    help="exit 1 unless every shared episode has the same success and length")
    args = ap.parse_args(argv)
    a, b = load(args.a), load(args.b)
    if a["task"] and b["task"] and a["task"] != b["task"]:
        raise SystemExit(f"different tasks: {a['task']} vs {b['task']}")
    r = compare(a, b)
    print(report(a, b, r))
    if args.require_identical and r["differing_seeds"]:
        print(f"NOT IDENTICAL on {len(r['differing_seeds'])} of {r['shared']} shared episodes")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
