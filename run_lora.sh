#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
[[ -x .venv/bin/lora ]] || { echo 'Run bash scripts/install_unix.sh first.'; exit 1; }
if [[ $# -eq 0 ]]; then set -- start; fi
exec .venv/bin/lora "$@"
