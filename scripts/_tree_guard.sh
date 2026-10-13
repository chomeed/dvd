# Sourced by the scripts that start `python -m online_rl_mimicgen...` (run.sh, smoke_train.sh, eval_dsrl.sh,
# build_pixel_cache.sh), after scripts/env.sh. Defines one function:
#
#   require_this_tree    exit 3 unless the `python` on PATH, started in $CODE_ROOT as those scripts start it, would import
#                        online_rl_mimicgen from $CODE_ROOT/rl
#
# scripts/env.sh puts $CODE_ROOT/rl first on PYTHONPATH, so this only fails when something outside the repository
# interferes: no `python` on PATH (the main environment is not activated), an interpreter that ignores PYTHONPATH, or a
# sitecustomize / .pth file that pushes another checkout in front. It imports nothing (find_spec only), so it costs one
# interpreter start. The scripts in eval/, dinowm/, value/ and robometer/ carry their own check of the same kind.

require_this_tree() {
  if ! command -v python > /dev/null 2>&1; then
    echo "no \`python\` on PATH: activate the main environment first (environment.md, section 2)" >&2
    exit 3
  fi
  if ! (cd "$CODE_ROOT" && python - "$CODE_ROOT" <<'PY'
import importlib.util
import os
import sys

want = os.path.realpath(os.path.join(sys.argv[1], "rl", "online_rl_mimicgen"))
spec = importlib.util.find_spec("online_rl_mimicgen")
found = [os.path.realpath(p) for p in (spec.submodule_search_locations or [])] if spec else []
if found[:1] != [want]:
    where = found[0] if found else "nowhere"
    print(f"python ({sys.executable}) would import online_rl_mimicgen from {where}, not from this checkout ({want}).\n"
          f"Run the scripts of the checkout whose code you mean, in a shell without another tree's PYTHONPATH.",
          file=sys.stderr)
    sys.exit(1)
PY
  ); then
    exit 3
  fi
}
