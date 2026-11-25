#!/usr/bin/env bash
set -euo pipefail
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
uv sync --locked --python 3.12 --extra dev
npm ci --no-audit --no-fund
.venv/bin/python -m playwright install chromium
if [[ "${1:-}" != --skip-models ]]; then
  .venv/bin/python scripts/download_models.py
  .venv/bin/python scripts/download_cactus_models.py
  .venv/bin/python -c 'from faster_whisper import WhisperModel; WhisperModel("base.en", device="cpu", compute_type="int8")'
fi
