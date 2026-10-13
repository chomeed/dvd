#!/usr/bin/env bash
# Download the published assets into the scripts/env.sh layout:
#
#   bash scripts/fetch_assets.sh --tier eval|train|regenerate [--task TASK ...] [--dry-run | --check [--sha256]] ...
#
# = scripts/fetch_assets.py (its --help is the full description) with the roots and HF_NAMESPACE of scripts/env.sh. This
# command exists to download, so the offline switches are cleared for it alone: an HF_OFFLINE=1 exported in the shell
# stays in force for everything else. Log in first for the private repositories: hf auth login (environment.md, section 5).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
exec python "$CODE_ROOT/scripts/fetch_assets.py" "$@"
