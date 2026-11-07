@echo off
REM ============================================================
REM  Evaluate the retrained v9 on the held-out test split.
REM  Prints pooled macro-F1, per-class report, and per-dataset macro-F1.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  Evaluating fer7_v9_5class_mixup_RETRAIN on test split
echo ============================================================
echo.

python eval_v9.py
if errorlevel 1 (
    echo.
    echo [ERROR] Eval failed. Send the error message.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Eval complete. Send Claude:
echo    1. The pooled macro-F1 line.
echo    2. The per-class classification report.
echo    3. The per-dataset macro-F1 block.
echo    4. The contents of:
echo         C:\Users\super\PyCharmMiscProject\models_cnn_multi\
echo            fer7_v9_5class_mixup_RETRAIN_meta.json
echo ============================================================
pause
