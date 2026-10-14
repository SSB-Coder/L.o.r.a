# ==============================================================================
# L.O.R.A - natural voice setup (Qwen3-TTS on your GPU)
#
# Creates a SEPARATE Python environment (.venv-tts) with the GPU build of PyTorch and qwen-tts.
# LORA finds it automatically and runs the voice there. Your main environment is not touched
# (`uv run` would undo a GPU PyTorch installed there, which is why the voice kept using the Windows voice).
#
# Run once from the project folder:   powershell -ExecutionPolicy Bypass -File scripts\install_tts.ps1
# Then restart LORA and say "run diagnostics".
# ==============================================================================
[CmdletBinding()]
param (
    [string]$Cuda = "cu128"   # cu126 or cu124 if your NVIDIA driver is older
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Env = Join-Path $ProjectRoot ".venv-tts"
$Py = Join-Path $Env "Scripts\python.exe"
$Uv = Get-Command uv -ErrorAction SilentlyContinue

if (-not (Test-Path $Py)) {
    Write-Host "Creating the voice environment..." -ForegroundColor Yellow
    if ($Uv) { & uv venv $Env --python 3.12 } else { & py -3.12 -m venv $Env }
    if ($LASTEXITCODE -ne 0) { Write-Error "Could not create .venv-tts (Python 3.12 needed)."; exit 1 }
}

function Install-Pkg([string[]]$PipArgs) {
    if ($Uv) { & uv pip install --python $Py @PipArgs } else { & $Py -m pip install @PipArgs }
    if ($LASTEXITCODE -ne 0) { Write-Error "Install failed: $PipArgs"; exit 1 }
}

Write-Host "Installing GPU PyTorch ($Cuda). This downloads about 2.5 GB..." -ForegroundColor Yellow
Install-Pkg @("torch", "--index-url", "https://download.pytorch.org/whl/$Cuda")

Write-Host "Installing qwen-tts and compatible transformers (4.53.x)..." -ForegroundColor Yellow
Install-Pkg @("qwen-tts", "transformers>=4.52.0,<4.54.0", "accelerate", "numpy")

Write-Host "`nChecking GPU support..." -ForegroundColor Yellow
& $Py -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU FOUND')"

Write-Host @"

Done. Put the model files in C:\Models:
  - Qwen3-TTS-12Hz-0.6B-CustomVoice   (about 2 GB of GPU memory: recommended on a 6 GB GPU that also runs the language model)
  - Qwen3-TTS-12Hz-1.7B-CustomVoice   (about 4.5 GB; used only if it fits)
LORA picks the biggest one that fits. Restart LORA, then say "run diagnostics".
"@ -ForegroundColor Green
