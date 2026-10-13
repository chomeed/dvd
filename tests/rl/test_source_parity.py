"""Source parity of rl/online_rl_mimicgen with the research tree it was cut from.

Every file is compared with its counterpart in LIVE_TREE after a normalisation that removes what the cut removed on
purpose: comments and docstrings (both sides, through the AST) and the method's old name (svg -> dvd, live side only).
Argparse help strings are KEPT. What still differs is a list of hunks, and every hunk must be in
source_parity_allowlist.json with a reason. So any edit to a ported file fails here until it is written down, and a new
commit in the live tree shows up as hunks nobody has explained.

    LIVE_TREE=<research tree> python tests/rl/test_source_parity.py           # the gate
    LIVE_TREE=<research tree> python tests/rl/test_source_parity.py --show    # print every hunk in full
    LIVE_TREE=<research tree> python tests/rl/test_source_parity.py --update  # rewrite the allow-list (keeps reasons)

Without LIVE_TREE only the checks that need no second tree run. The allow-list stores the new side of a hunk verbatim
and the live side as a SHA-256 (live lines contain paths of the machine they were written on). It was generated with
Python 3.12; ast.unparse may format differently on another minor version, in which case regenerate with --update
against the same live commit and confirm that only formatting moved.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

import _util

ALLOWLIST = Path(__file__).with_name("source_parity_allowlist.json")
# new file (relative to rl/online_rl_mimicgen) -> live file (relative to LIVE_TREE) when the name or place changed
MOVED = {"dvd_term.py": "0811/svg_term.py", "dino_dvd.py": "online_rl_mimicgen/dino_svg.py"}
# files with no live counterpart
NEW_ONLY = {"legacy.py"}
# the only file allowed to spell the old name
OLD_NAME_FILES = {"legacy.py"}


def new_files() -> list[str]:
    return sorted(p.relative_to(_util.PACKAGE).as_posix() for p in _util.PACKAGE.rglob("*.py"))


def live_path(root: Path, rel: str) -> Path:
    return root / MOVED.get(rel, f"online_rl_mimicgen/{rel}")


def normalise(path: Path, rename: bool) -> list[str]:
    source = path.read_text()
    if rename:
        source = source.replace("svg", "dvd").replace("SVG", "DVD").replace("Svg", "Dvd")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).splitlines()


def digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def hunks(root: Path, rel: str) -> list[dict]:
    live, new = normalise(live_path(root, rel), True), normalise(_util.PACKAGE / rel, False)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, live, new, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append({"live": live[i1:i2], "new": new[j1:j2]})
    return out


def key(hunk: dict) -> tuple:
    return (hunk["live_sha256"], tuple(hunk["new"])) if "live_sha256" in hunk else (digest(hunk["live"]), tuple(hunk["new"]))


def show(rel: str, hunk: dict) -> str:
    return "\n".join([f"  {rel}:"] + [f"    - {line}" for line in hunk.get("live", [f"<{hunk.get('live_lines')} live line(s), sha256 {hunk.get('live_sha256', '')[:12]}>"])]
                     + [f"    + {line}" for line in hunk["new"]])


def test_file_set():
    """The package holds exactly the ported files plus the legacy layer; nothing slipped in or out."""
    files = new_files()
    allow = json.loads(ALLOWLIST.read_text())["files"]
    ported = [rel for rel in files if rel not in NEW_ONLY]
    print(f"[files] {len(files)} python files, {len(ported)} ported, new-only {sorted(NEW_ONLY)}")
    assert set(NEW_ONLY) <= set(files), f"missing new-only files: {sorted(set(NEW_ONLY) - set(files))}"
    assert sorted(allow) == ported, (f"allow-list and package disagree: only in package {sorted(set(ported) - set(allow))}, "
                                     f"only in allow-list {sorted(set(allow) - set(ported))}")


def test_old_name_only_in_legacy():
    """No file but legacy.py spells the method's old name, in code, comments or strings."""
    hits = {}
    for rel in new_files():
        if rel in OLD_NAME_FILES:
            continue
        found = sorted(set(re.findall(r"[\w.\-]*svg[\w.\-]*", (_util.PACKAGE / rel).read_text(), flags=re.I)))
        if found:
            hits[rel] = found
    print(f"[names] old-name tokens outside {sorted(OLD_NAME_FILES)}: {hits or 'none'}")
    assert not hits, hits


def test_allowlist_reasons():
    """Every allow-listed hunk says why it exists."""
    data = json.loads(ALLOWLIST.read_text())
    entries = [(rel, h) for rel, hs in data["files"].items() for h in hs]
    missing = [rel for rel, h in entries if not h.get("why", "").strip() or h["why"].startswith("TODO")]
    print(f"[allow-list] {len(entries)} hunks in {sum(1 for hs in data['files'].values() if hs)} files "
          f"(generated with python {data['generated_with']['python']} against live commit {data['generated_with']['live_commit']}); "
          f"{len(missing)} without a reason")
    assert not missing, f"hunks without a reason in: {sorted(set(missing))}"


def test_source_parity():
    """Normalised diff of every ported file against LIVE_TREE == the allow-list, hunk for hunk."""
    root = _util.live_tree()
    allow = json.loads(ALLOWLIST.read_text())["files"]
    problems, total, identical = [], 0, 0
    for rel in new_files():
        if rel in NEW_ONLY:
            continue
        if not live_path(root, rel).exists():
            problems.append(f"  {rel}: no live counterpart at {live_path(root, rel)}")
            continue
        got, want = hunks(root, rel), allow.get(rel, [])
        total += len(got)
        identical += not got
        got_keys, want_keys = [key(h) for h in got], [key(h) for h in want]
        if got_keys != want_keys:
            for h in got:
                if key(h) not in want_keys:
                    problems.append("UNEXPLAINED hunk (in the code, not in the allow-list):\n" + show(rel, h))
            for h in want:
                if key(h) not in got_keys:
                    problems.append("STALE allow-list entry (no longer in the code):\n" + show(rel, h))
            if sorted(got_keys) == sorted(want_keys):
                problems.append(f"  {rel}: same hunks in a different order; run --update")
    print(f"[parity] {len(new_files()) - len(NEW_ONLY)} files against {root}: {identical} identical after normalisation, "
          f"{total} hunks in the rest, {len(problems)} problem(s)")
    assert not problems, "source parity broken:\n" + "\n".join(problems)


def update(root: Path) -> None:
    old = json.loads(ALLOWLIST.read_text()) if ALLOWLIST.exists() else {"files": {}, "generated_with": {}}
    reasons = {(rel, key(h)): h.get("why", "") for rel, hs in old["files"].items() for h in hs}
    files = {}
    for rel in new_files():
        if rel in NEW_ONLY:
            continue
        files[rel] = [{"why": reasons.get((rel, key(h)), "TODO: explain this edit"), "live_lines": len(h["live"]),
                       "live_sha256": digest(h["live"]), "new": h["new"]} for h in hunks(root, rel)]
    generated = {"python": f"{sys.version_info[0]}.{sys.version_info[1]}",
                 "live_commit": old["generated_with"].get("live_commit", "unknown")}
    ALLOWLIST.write_text(json.dumps({"generated_with": generated, "files": files}, indent=1) + "\n")
    todo = sum(h["why"].startswith("TODO") for hs in files.values() for h in hs)
    print(f"wrote {ALLOWLIST.name}: {sum(map(len, files.values()))} hunks, {todo} need a reason")


if __name__ == "__main__":
    if "--update" in sys.argv:
        update(_util.live_tree())
    elif "--show" in sys.argv:
        tree = _util.live_tree()
        for name in new_files():
            if name not in NEW_ONLY:
                for item in hunks(tree, name):
                    print(show(name, item))
    else:
        _util.main(globals())
