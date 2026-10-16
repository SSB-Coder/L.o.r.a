# Sets up Kokoro-ONNX TTS (CPU, 0 MB GPU VRAM, default voice af_sky) and downloads model files to C:\Models\kokoro.
# Pass -LegacyQwen if you want the old .venv-tts GPU Qwen3-TTS environment instead.
param(
    [switch]$LegacyQwen = $false
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $LegacyQwen) {
    Write-Host "==> Installing kokoro-onnx + soundfile into main environment..." -ForegroundColor Cyan
    $MainPy = Join-Path ".venv" "Scripts\python.exe"
    uv pip install --python $MainPy -U kokoro-onnx soundfile

    $KokoroDir = Join-Path (Get-Location) "models\kokoro"
    New-Item -ItemType Directory -Force -Path $KokoroDir | Out-Null

    $OnnxFile = Join-Path $KokoroDir "kokoro-v1.0.onnx"
    $VoicesFile = Join-Path $KokoroDir "voices-v1.0.bin"

    # Fast cache check from global C:\Models\kokoro if present
    if (Test-Path "C:\Models\kokoro\kokoro-v1.0.onnx" -and (-not (Test-Path $OnnxFile))) {
        Copy-Item -Path "C:\Models\kokoro\kokoro-v1.0.onnx" -Destination $OnnxFile -Force
    }
    if (Test-Path "C:\Models\kokoro\voices-v1.0.bin" -and (-not (Test-Path $VoicesFile))) {
        Copy-Item -Path "C:\Models\kokoro\voices-v1.0.bin" -Destination $VoicesFile -Force
    }

    if (-not (Test-Path $OnnxFile)) {
        Write-Host "==> Downloading kokoro-v1.0.onnx (~310 MB) to $KokoroDir..." -ForegroundColor Cyan
        curl.exe -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" -o $OnnxFile
    }
    if (-not (Test-Path $VoicesFile)) {
        Write-Host "==> Downloading voices-v1.0.bin (~27 MB) to $KokoroDir..." -ForegroundColor Cyan
        curl.exe -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -o $VoicesFile
    }

    & $MainPy -c "from kokoro_onnx import Kokoro; k = Kokoro(r'$OnnxFile', r'$VoicesFile'); s, sr = k.create('LORA voice ready with Sky.', voice='af_sky', speed=1.0, lang='en-us'); print('Kokoro TTS ready: samples=', len(s), 'sr=', sr)"
    Write-Host "==> Done. Start LORA normally: uv run lora (default voice: af_sky)" -ForegroundColor Green
    exit 0
}

Write-Host "==> Creating .venv-tts (separate GPU environment for the legacy Qwen voice)..." -ForegroundColor Cyan
if (-not (Test-Path ".venv-tts\Scripts\python.exe")) {
    uv venv .venv-tts --python 3.11
}
$Py = (Resolve-Path ".venv-tts\Scripts\python.exe").Path

Write-Host "==> Installing CUDA PyTorch into .venv-tts..." -ForegroundColor Cyan
uv pip install --python $Py --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

Write-Host "==> Installing qwen-tts with compatible transformers into .venv-tts..." -ForegroundColor Cyan
uv pip install --python $Py "qwen-tts" "transformers>=4.52.0,<4.54.0" accelerate safetensors soundfile librosa einops
