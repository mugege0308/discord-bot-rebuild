@echo off
setlocal
cd /d "%~dp0"

rem Keep the batch wrapper small; the PowerShell launcher owns health checks and logging.
start "" /b powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0launch_generation_backend.ps1"
exit /b 0
