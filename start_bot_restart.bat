@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"

echo RESTART_LAUNCHER_START timestamp=%DATE% %TIME% cwd=%CD%

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "C:\Users\PC_User\AppData\Local\Python\pythoncore-3.14-64\python.exe" set "PYTHON_EXE=C:\Users\PC_User\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if defined PYTHON_EXE (
    echo RESTART_LAUNCHER_PYTHON=%PYTHON_EXE%
    "%PYTHON_EXE%" main.py
    set "EXIT_CODE=%ERRORLEVEL%"
) else (
    echo RESTART_LAUNCHER_PYTHON=python
    python main.py
    set "EXIT_CODE=%ERRORLEVEL%"
)

echo RESTART_LAUNCHER_EXIT code=%EXIT_CODE%
exit /b %EXIT_CODE%
