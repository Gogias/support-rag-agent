@echo off
rem Run triage eval. Keep this file next to triage_eval.py and double-click it.
rem Optional argument: dataset path relative to this folder, e.g. golden_sets\other.jsonl
chcp 65001 >nul
set PYTHONUTF8=1

for %%I in ("%~dp0.") do set "PKG=%%~nxI"

set "CONDA=%USERPROFILE%\miniconda3"
if not exist "%CONDA%\Scripts\activate.bat" (
    echo Miniconda not found: %CONDA%
    echo Edit the CONDA line in this file.
    pause
    exit /b 1
)
call "%CONDA%\Scripts\activate.bat"

set "ARG="
if not "%~1"=="" set "ARG=%~dp0%~1"

cd /d "%~dp0.."
echo Python: & where python | findstr /n "^" | findstr "^1:"
python -m %PKG%.triage_eval %ARG%
pause