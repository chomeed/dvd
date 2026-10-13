#!/usr/bin/env bash
# Run every test script of tests/ and print one line per script, then the totals.
#
#   bash tests/run_all.sh                    # all scripts, with whatever the environment provides
#   bash tests/run_all.sh rl test_recipes    # only the scripts whose path contains one of these words
#   bash tests/run_all.sh --list             # the scripts, in the order they run
#
# Two ways to run it (tests/README.md has the details and what each script covers):
#
#   fresh clone        nothing set. Only the tests that need no asset, no third-party checkout and no second source
#                      tree run; the others print SKIP. CPU only, about a minute.
#   reference machine  asset roots (ASSET_ROOT or DATA_ROOT / CKPT_ROOT / CACHE_ROOT), the third-party checkouts
#                      (THIRD_PARTY or DSRL_ROOT / MIMICGEN_ROOT / TASK_ZOO_ROOT), HF_HOME with the two base models,
#                      LIVE_TREE=<research tree>, LEGACY_RUNS=<dir of reference runs>[:<dir>...] and
#                      HYGIENE_EXTRA_WORDS=<root dir name>,<user name>. Still CPU only, about four minutes.
#
# scripts/env.sh is sourced here, so only the variables above have to be set; values already in the environment are
# kept. Other switches:
#   PYTHON         interpreter [python on PATH]
#   TEST_LOG_DIR   directory for the per-script logs [a new directory under ${TMPDIR:-/tmp}]
#
# Exit status 0 iff no script failed. SKIP is neither: it means an optional input was absent, and the line says which.
# tests/pipeline/test_bon16_parity.py needs a GPU and a decision dump and therefore always prints SKIP here; its header
# says how to run it.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../scripts/env.sh"
export PYTHONDONTWRITEBYTECODE=1           # leave no __pycache__ in the repository
PY=${PYTHON:-python}

SCRIPTS=(configs/test_configs.py rl/test_source_parity.py rl/test_legacy.py rl/test_cpu_parity.py
         pipeline/test_pool_mqattn.py pipeline/test_dino_wm.py pipeline/test_recipes.py pipeline/test_offline_data.py
         pipeline/test_eval_tools.py pipeline/test_bon16_parity.py)
# a test script that exists but is not listed above would silently never run
for f in "$HERE"/*/test_*.py; do
  case " ${SCRIPTS[*]} " in *" ${f#"$HERE"/} "*) ;; *) echo "tests/run_all.sh: ${f#"$HERE"/} is not in SCRIPTS" >&2; exit 2;; esac
done

if [ "${1:-}" = --list ]; then printf 'tests/%s\n' "${SCRIPTS[@]}"; exit 0; fi
if [ $# -gt 0 ]; then
  picked=()
  for s in "${SCRIPTS[@]}"; do for w in "$@"; do case $s in *"$w"*) picked+=("$s"); break;; esac; done; done
  [ ${#picked[@]} -gt 0 ] || { echo "no test script matches: $*" >&2; exit 2; }
  SCRIPTS=("${picked[@]}")
fi
command -v "$PY" > /dev/null 2>&1 || { echo "no $PY on PATH: activate the main environment (environment.md)" >&2; exit 2; }

LOGS=${TEST_LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/dvd_tests.XXXXXX")}
mkdir -p "$LOGS"
set_or_unset() { if [ -n "${!1:-}" ]; then echo "$1=${!1}"; else echo "$1 unset"; fi; }
echo "tests of $CODE_ROOT"
echo "  CKPT_ROOT=$CKPT_ROOT  DATA_ROOT=$DATA_ROOT  CACHE_ROOT=$CACHE_ROOT"
echo "  DSRL_ROOT=$DSRL_ROOT  $(set_or_unset LIVE_TREE)  $(set_or_unset LEGACY_RUNS)  $(set_or_unset HYGIENE_EXTRA_WORDS)"
echo "  logs: $LOGS"

pass=0; skip=0; fail=0; broken=0
for s in "${SCRIPTS[@]}"; do
  log=$LOGS/$(echo "${s%.py}" | tr / _).log
  t0=$(date +%s)
  "$PY" -W ignore "$HERE/$s" > "$log" 2>&1; rc=$?
  p=$(grep -c '^PASS ' "$log"); k=$(grep -c '^SKIP ' "$log"); f=$(grep -c '^FAIL ' "$log")
  pass=$((pass + p)); skip=$((skip + k)); fail=$((fail + f))
  note=""
  if [ "$rc" -ne 0 ] && [ "$f" -eq 0 ]; then broken=$((broken + 1)); note="  (exit $rc with no FAIL line: the script itself broke, see the log)"; fi
  if [ "$rc" -eq 0 ] && [ "$f" -ne 0 ]; then broken=$((broken + 1)); note="  (FAIL lines but exit 0)"; fi
  printf '%-36s %3d pass %3d skip %3d fail  %4ds%s\n' "tests/$s" "$p" "$k" "$f" "$(( $(date +%s) - t0 ))" "$note"
  grep '^SKIP ' "$log" | cut -c1-220 | sed 's/^/    /'
  grep '^FAIL ' "$log" | sed 's/^/    /'
done
echo "TOTAL ${#SCRIPTS[@]} scripts: $pass pass, $skip skip, $fail fail$( [ "$broken" -gt 0 ] && echo ", $broken script(s) broken" )"
[ "$fail" -eq 0 ] && [ "$broken" -eq 0 ]
