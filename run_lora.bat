@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please run scripts\install.ps1 first to install L.O.R.A.
    pause
    exit /b 1
)

echo Starting L.O.R.A Desktop Platform...
".venv\Scripts\python.exe" -m lora.main %*
endlocal
