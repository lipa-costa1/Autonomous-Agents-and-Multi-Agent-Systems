@echo off
REM Using sets inside of ifs annoys windows, this Setlocal fixes that
Setlocal EnableDelayedExpansion

for /L %%i in (1, 1, 1) do (
    start cmd.exe /c "conda activate rocket && python run.py" ^& pause
    timeout 45 >nul
)