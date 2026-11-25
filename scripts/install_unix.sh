#!/usr/bin/env bash
set -euo pipefail
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
uv sync --locked --inexact --python 3.12 --extra dev
if command -v npm >/dev/null; then
  npm ci --no-audit --no-fund || echo 'JavaScript test dependencies could not be installed; continuing desktop setup.' >&2
else
  echo 'npm not installed: skipping JavaScript test dependencies (not needed to run Lora).'
fi
.venv/bin/python scripts/setup_runtime.py "$@"
printf 'Setup complete. Start Lora with %s/run_lora.sh\n' "$ROOT"
