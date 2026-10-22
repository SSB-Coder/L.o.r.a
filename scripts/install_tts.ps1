# Install the locked runtime and set up CPU-only Kokoro speech.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw "Runtime installation failed" }
uv run python scripts/download_models.py --kokoro-only
if ($LASTEXITCODE -ne 0) { throw "Kokoro model setup failed" }
uv run python -c "from lora.voice.tts.worker import KokoroTtsWorkerProcess; w=KokoroTtsWorkerProcess(); c=w.synthesize('LORA voice ready.'); print(w.diagnosis()); raise SystemExit(bool(c.error) or not len(c.samples))"
if ($LASTEXITCODE -ne 0) { throw "Kokoro CPU synthesis failed; see the error above" }
Write-Host "Kokoro CPU voice ready. Start with: uv run lora start" -ForegroundColor Green
