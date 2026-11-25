# ==============================================================================
# L.O.R.A - Native Windows Installer & Desktop Environment Setup
# ==============================================================================
[CmdletBinding()]
param (
    [switch]$SkipShortcuts,
    [switch]$SkipAcl,
    [switch]$SkipModels,
    [switch]$SkipBrowser,
    [ValidateSet("faiss", "neural-routing", "wakeword", "experimental", "dev")]
    [string[]]$Extras = @()
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $ProjectRoot

function Invoke-Checked {
    param ([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE. See the output above."
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  L.O.R.A - Native Windows Desktop Platform Installer       " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray

Write-Host "`n[1/4] Installing Python packages into .venv..." -ForegroundColor Yellow
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($UvCmd) {
    $SyncArgs = @("sync", "--locked", "--inexact", "--python", "3.12")
    foreach ($Extra in $Extras) { $SyncArgs += @("--extra", $Extra) }
    Invoke-Checked -Program "uv" -Arguments $SyncArgs
} else {
    if (-not (Test-Path $VenvPython)) {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            throw "Install uv (https://docs.astral.sh/uv/) or Python 3.12 before running setup."
        }
        Invoke-Checked -Program "python" -Arguments @("-c", "import platform, sys; sys.exit(0 if sys.version_info >= (3,11) and platform.machine().lower() in ('amd64','x86_64') else 'Install uv or an x64 Python 3.12 interpreter first.')")
        Invoke-Checked -Program "python" -Arguments @("-m", "venv", "$ProjectRoot\.venv")
    }
    $PackageSpec = if ($Extras.Count) { ".[" + ($Extras -join ",") + "]" } else { "." }
    Invoke-Checked -Program $VenvPython -Arguments @("-m", "pip", "install", "-e", $PackageSpec)
}

Write-Host "`n[2/4] Preparing browser and model assets..." -ForegroundColor Yellow
$RuntimeArgs = @("scripts/setup_runtime.py")
if ($SkipModels) { $RuntimeArgs += "--skip-models" }
if ($SkipBrowser) { $RuntimeArgs += "--skip-browser" }
Invoke-Checked -Program $VenvPython -Arguments $RuntimeArgs

# 5. Initialize ~/.lora/ and lock NTFS ACLs
Write-Host "`n[3/4] Initializing ~/.lora/ ecosystem & securing NTFS permissions..." -ForegroundColor Yellow
$ApplyAcls = if ($SkipAcl) { "False" } else { "True" }
$InitCmd = "from lora.config.manager import ConfigManager; ConfigManager().ensure_ecosystem(apply_acls=$ApplyAcls)"
Invoke-Checked -Program $VenvPython -Arguments @("-c", $InitCmd)
Write-Host "User ecosystem initialized at: $env:USERPROFILE\.lora" -ForegroundColor Green

# 6. Create Windows Desktop & Start Menu Shortcuts
if (-not $SkipShortcuts) {
    Write-Host "`n[4/4] Creating Windows Desktop and Start Menu shortcuts..." -ForegroundColor Yellow
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
    Write-Host "`n[4/4] Skipping shortcut creation (-SkipShortcuts specified)." -ForegroundColor Gray
}

# Create lora CLI command wrapper in ~/.lora/bin
$LoraBin = Join-Path "$env:USERPROFILE\.lora" "bin"
if (-not (Test-Path $LoraBin)) {
    New-Item -ItemType Directory -Path $LoraBin -Force | Out-Null
}
$LoraCmdPath = Join-Path $LoraBin "lora.cmd"
if ($VenvPython -cmatch '[^ -~]') {
    Write-Warning "The checkout path needs Unicode support. Use run_lora.bat or an installed shortcut instead of the ASCII CLI wrapper."
} else {
    $LoraCmdContent = "@echo off`r`n`"$VenvPython`" -m lora.main %*`r`n"
    Set-Content -Path $LoraCmdPath -Value $LoraCmdContent -Encoding ASCII
    Write-Host "Created CLI wrapper at: $LoraCmdPath" -ForegroundColor Green
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  Installation Complete!                                    " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "You can now run L.O.R.A:" -ForegroundColor White
Write-Host "  1. Double-click the L.O.R.A icon on your Desktop" -ForegroundColor Gray
Write-Host "  2. Launch from Start Menu: Search 'L.O.R.A'" -ForegroundColor Gray
Write-Host "  3. Run from terminal: .\run_lora.bat" -ForegroundColor Gray
Write-Host "============================================================`n" -ForegroundColor Green
