@echo off
REM Using sets inside of ifs annoys windows, this Setlocal fixes that
Setlocal EnableDelayedExpansion

REM optional argument to launch multiple workers at once
set instance_num=1
if %1.==. goto :endparams
set instance_num=%1
:endparams

set /p password=Enter password:

for /L %%i in (1, 1, !instance_num!) do (
    start cmd.exe /c "conda activate rocket && python training/worker.py sequeira !password! --compress" ^& pause
    timeout 45 >nul
)