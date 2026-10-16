# ==============================================================================
# L.O.R.A - Native Windows Installer & Desktop Environment Setup
# ==============================================================================
[CmdletBinding()]
param (
    [switch]$SkipShortcuts,
    [switch]$SkipAcl,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  L.O.R.A - Native Windows Desktop Platform Installer       " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray

# 1. Check Python
Write-Host "`n[1/6] Checking Python runtime..." -ForegroundColor Yellow
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    Write-Error "Python is not installed or not in system PATH. Please install Python 3.11 or higher."
    exit 1
}
$PythonVersion = & python --version 2>&1
Write-Host "Found: $PythonVersion" -ForegroundColor Green

# 2. Virtual Environment Setup
Write-Host "`n[2/6] Setting up virtual environment (.venv)..." -ForegroundColor Yellow
$VenvPython = "$ProjectRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment at .venv..." -ForegroundColor Gray
    & python -m venv "$ProjectRoot\.venv"
} else {
    Write-Host "Virtual environment already present." -ForegroundColor Green
}

# 3. Installing Dependencies
Write-Host "`n[3/6] Installing package and dependencies..." -ForegroundColor Yellow
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($UvCmd) {
    Write-Host "Using uv for fast dependency installation..." -ForegroundColor Gray
    & uv pip install -e .
} else {
    Write-Host "Using pip for dependency installation..." -ForegroundColor Gray
    & "$ProjectRoot\.venv\Scripts\pip.exe" install -e .
}

# 4. Setup Models in models/
if (-not $SkipModels) {
    Write-Host "`n[4/6] Setting up models in models/ (Kokoro TTS, Laya Router, Embedding Gemma 2)..." -ForegroundColor Yellow
    & "$VenvPython" "$ProjectRoot\scripts\download_models.py"
} else {
    Write-Host "`n[4/6] Skipping models setup (-SkipModels specified)." -ForegroundColor Gray
}

# 5. Initialize ~/.lora/ and lock NTFS ACLs
Write-Host "`n[5/6] Initializing ~/.lora/ ecosystem & securing NTFS permissions..." -ForegroundColor Yellow
$InitCmd = "from lora.config.manager import ConfigManager; mgr = ConfigManager(); mgr.ensure_ecosystem(apply_acls=$(-not $SkipAcl))"
& "$VenvPython" -c "$InitCmd"
Write-Host "User ecosystem initialized at: $env:USERPROFILE\.lora" -ForegroundColor Green

# 6. Create Windows Desktop & Start Menu Shortcuts
if (-not $SkipShortcuts) {
    Write-Host "`n[6/6] Creating Windows Desktop and Start Menu shortcuts..." -ForegroundColor Yellow
    $WshShell = New-Object -ComObject WScript.Shell
    
    $RunnerBat = "$ProjectRoot\run_lora.bat"
    
    # Desktop Shortcut
    $DesktopDir = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
    $DesktopShortcutPath = Join-Path $DesktopDir "L.O.R.A.lnk"
    $Shortcut = $WshShell.CreateShortcut($DesktopShortcutPath)
    $Shortcut.TargetPath = $RunnerBat
    $Shortcut.WorkingDirectory = $ProjectRoot
    $Shortcut.Description = "L.O.R.A - Layer Oriented Research Assistant"
    $Shortcut.Save()
    Write-Host "Created Desktop shortcut: $DesktopShortcutPath" -ForegroundColor Green
    
    # Start Menu Shortcut
    $StartMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
    $StartMenuShortcutPath = Join-Path $StartMenuDir "L.O.R.A.lnk"
    $StartShortcut = $WshShell.CreateShortcut($StartMenuShortcutPath)
    $StartShortcut.TargetPath = $RunnerBat
    $StartShortcut.WorkingDirectory = $ProjectRoot
    $StartShortcut.Description = "L.O.R.A - Layer Oriented Research Assistant"
    $StartShortcut.Save()
    Write-Host "Created Start Menu shortcut: $StartMenuShortcutPath" -ForegroundColor Green
} else {
    Write-Host "`n[5/5] Skipping shortcut creation (-SkipShortcuts specified)." -ForegroundColor Gray
}

# Create lora CLI command wrapper in ~/.lora/bin
$LoraBin = Join-Path "$env:USERPROFILE\.lora" "bin"
if (-not (Test-Path $LoraBin)) {
    New-Item -ItemType Directory -Path $LoraBin -Force | Out-Null
}
$LoraCmdPath = Join-Path $LoraBin "lora.cmd"
$LoraCmdContent = "@echo off`r`n`"$VenvPython`" -m lora.main %*`r`n"
Set-Content -Path $LoraCmdPath -Value $LoraCmdContent -Encoding ASCII
Write-Host "Created CLI wrapper at: $LoraCmdPath" -ForegroundColor Green

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  Installation Complete!                                    " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "You can now run L.O.R.A:" -ForegroundColor White
Write-Host "  1. Double-click the L.O.R.A icon on your Desktop" -ForegroundColor Gray
Write-Host "  2. Launch from Start Menu: Search 'L.O.R.A'" -ForegroundColor Gray
Write-Host "  3. Run from terminal: .\run_lora.bat" -ForegroundColor Gray
Write-Host "============================================================`n" -ForegroundColor Green
