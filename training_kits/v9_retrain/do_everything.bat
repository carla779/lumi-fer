@echo off
REM ============================================================
REM  do_everything.bat — runs verify+patch, train, eval in sequence.
REM  Total wall time: ~5 hours (mostly training).
REM
REM  Recommended for first run: use 1_, 2_, 3_ separately so you
REM  can review the output of each step before committing to the
REM  long training run.
REM ============================================================
cd /d "%~dp0"

call 1_verify_and_patch.bat
if errorlevel 1 (
    echo [ERROR] verify+patch failed. Aborting.
    exit /b 1
)

call 2_train_v9.bat
if errorlevel 1 (
    echo [ERROR] training failed. Aborting.
    exit /b 1
)

call 3_eval_v9.bat
if errorlevel 1 (
    echo [ERROR] eval failed.
    exit /b 1
)

echo.
echo ============================================================
echo  All steps complete. Send Claude the eval output and meta.json.
echo ============================================================
pause
