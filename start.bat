@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

set "CONDA=%USERPROFILE%\miniconda3"
if not exist "%CONDA%\Scripts\activate.bat" (
    echo Miniconda not found: %CONDA%
    echo Edit the CONDA line in this file.
    pause
    exit /b 1
)
call "%CONDA%\Scripts\activate.bat"

set HF_HUB_OFFLINE=1
python -m src.agent.graph %*
pause