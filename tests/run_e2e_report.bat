@echo off
rem Double-click. Keep next to e2e_report.py, no arguments needed.
chcp 65001 >nul
set PYTHONUTF8=1

set "CONDA=%USERPROFILE%\miniconda3"
if not exist "%CONDA%\Scripts\activate.bat" (
    echo Miniconda not found: %CONDA%
    echo Edit the CONDA line in this file.
    pause
    exit /b 1
)
call "%CONDA%\Scripts\activate.bat"

cd /d "%~dp0"
python e2e_report.py
pause
