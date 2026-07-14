@echo off
chcp 65001 > nul
title Shogou-chan + Codex Auto Start

echo Shogou-chan + Codex Auto Start
echo ========================================
echo.

set "PROJECT_DIR=%~dp0"

echo Project folder:
echo "%PROJECT_DIR%"
echo.

echo Waiting for network...
timeout /t 45 /nobreak

echo.
echo Moving to project folder...
cd /d "%PROJECT_DIR%"

echo.
echo Checking main.py...
if not exist "main.py" (
    echo ERROR: main.py was not found in this folder.
    echo Current folder:
    cd
    echo.
    echo Files:
    dir
    echo.
    pause
    exit /b 1
)

echo.
echo Starting Shogou-chan Bot in a new window...
start "Shogou-chan Bot" cmd /k python main.py

echo.
echo Waiting a little before opening Codex...
timeout /t 5 /nobreak

echo.
echo Starting Codex Windows app...
echo Searching Start Menu apps for ChatGPT, then Codex...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$app = Get-StartApps | Where-Object { $_.Name -like '*ChatGPT*' } | Select-Object -First 1; if (-not $app) { $app = Get-StartApps | Where-Object { $_.Name -like '*Codex*' } | Select-Object -First 1 }; if ($app) { Write-Host ('Found app: ' + $app.Name); Write-Host ('AppID: ' + $app.AppID); Start-Process -FilePath explorer.exe -ArgumentList ('shell:AppsFolder\' + $app.AppID); exit 0 } else { exit 1 }"

if errorlevel 1 (
    echo WARNING: ChatGPT/Codex Windows app could not be found in Start Menu apps.
    echo The app will not start from this bat.
    echo.
    echo Please start ChatGPT or Codex manually from the Start Menu.
)

echo.
echo Done.
echo Shogou-chan Bot was started.
echo If ChatGPT/Codex did not start, check the messages above.
echo.
pause
exit /b 0
