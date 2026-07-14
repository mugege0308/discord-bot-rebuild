@echo off
chcp 65001 > nul
echo Starting ComfyUI...
echo.
echo Checking Git...
set "GIT_EXE=C:\Program Files\Git\cmd\git.exe"
if not exist "%GIT_EXE%" (
    echo Git executable was not found.
    echo Expected:
    echo %GIT_EXE%
    echo.
    echo Please install Git for Windows, or update GIT_EXE in this bat file.
    echo.
    pause
    exit /b 1
)
set "GIT_PYTHON_GIT_EXECUTABLE=%GIT_EXE%"
set "PATH=C:\Program Files\Git\cmd;%PATH%"
echo Git found:
echo %GIT_EXE%
echo.
echo Moving to ComfyUI folder...
cd /d "C:\Users\PC_User\Desktop\画像生成\StabilityMatrix-win-x64\Data\Packages\ComfyUI"
if errorlevel 1 (
    echo Failed to move to ComfyUI folder.
    echo Please check the folder path.
    echo.
    pause
    exit /b 1
)

echo.
echo Launching ComfyUI...
"C:\Users\PC_User\Desktop\画像生成\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe" -u "C:\Users\PC_User\Desktop\画像生成\StabilityMatrix-win-x64\Data\Packages\ComfyUI\main.py" --preview-method auto --use-pytorch-cross-attention --enable-manager
echo.
echo ComfyUI process ended.
pause
