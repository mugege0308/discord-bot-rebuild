@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo Starting Discord secretary bot...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo Failed to start with: python main.py
    echo Trying: py main.py
    echo.
    py main.py
)

echo.
pause
