@echo off
REM ============================================================
REM  v9 retrain prep: verify data, patch CSV, patch trainer
REM  Idempotent: safe to run multiple times.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  Step 1a: verify corpus (initial)
echo ============================================================
python verify.py
set VERIFY_EXIT=%errorlevel%
if %VERIFY_EXIT%==0 (
    echo.
    echo Initial verification PASSED already. Patches will still run idempotently.
)

echo.
echo ============================================================
echo  Step 1b: patch unified CSV (substitute orig_image)
echo ============================================================
python patch_csv.py
if errorlevel 1 (
    echo.
    echo [ERROR] patch_csv.py reported a problem. Stopping.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 1c: patch trainer (add diagnostic prints)
echo ============================================================
python patch_trainer.py
if errorlevel 1 (
    echo.
    echo [ERROR] patch_trainer.py reported a problem. Stopping.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 1d: re-verify after patches
echo ============================================================
python verify.py
if errorlevel 1 (
    echo.
    echo [WARN] Some datasets still have under 95%% path resolution.
    echo        Review the messages above. You can still proceed to
    echo        train, but the trainer will silently drop those rows.
    echo        See [loader] prints during training to confirm counts.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Verify+patch complete. Ready to run 2_train_v9.bat.
echo ============================================================
pause
