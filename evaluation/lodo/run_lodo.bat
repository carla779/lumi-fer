@echo off
REM =====================================================================
REM  LUMI FER — Leave-One-Dataset-Out (LODO) cross-dataset evaluation.
REM
REM  Trains four 5-class variants of the v9 pipeline. Each variant trains
REM  on three of {RAF-DB, FERPlus, ExpW, FER2013} and tests on the held-out
REM  fourth. This reveals whether the model learns transferable emotion
REM  features or dataset-specific artefacts.
REM
REM  Each run is identical to the v9 baseline (EfficientNet-B0, soft labels,
REM  AU loss, MixUp, balanced sampler) — only the train/test split changes.
REM
REM  Each run ~1.5–2.5 h on a typical gaming GPU; total ~6–10 h.
REM =====================================================================

set SCRIPT=cnn_train_5class_fer_soft_au.py
set CSV=unified_fer7_processed_au_FRIEND.csv

set COMMON_ARGS=--arch efficientnet_b0 --pretrained --img_size 224 --batch 32 --epochs 30 --lr 1e-4 --num_workers 2 --save_dir models_lodo --lambda_au 0.05 --mixup_alpha 0.2 --balance dataset_class --smoothing 0.05

echo ============================================================
echo Step 0: build LODO CSVs
echo ============================================================
python make_lodo_csvs.py --in %CSV%
if errorlevel 1 ( echo [ERROR] failed to build LODO CSVs & exit /b 1 )

echo.
echo ============================================================
echo Run 1/4  LODO holdout = RAF-DB
echo ============================================================
python %SCRIPT% --csv lodo_HOLDOUT_rafdb.csv %COMMON_ARGS% --run_name LODO_rafdb
if errorlevel 1 ( echo [ERROR] LODO_rafdb failed & exit /b 1 )

echo.
echo ============================================================
echo Run 2/4  LODO holdout = FERPlus
echo ============================================================
python %SCRIPT% --csv lodo_HOLDOUT_ferplus.csv %COMMON_ARGS% --run_name LODO_ferplus
if errorlevel 1 ( echo [ERROR] LODO_ferplus failed & exit /b 1 )

echo.
echo ============================================================
echo Run 3/4  LODO holdout = ExpW
echo ============================================================
python %SCRIPT% --csv lodo_HOLDOUT_expw.csv %COMMON_ARGS% --run_name LODO_expw
if errorlevel 1 ( echo [ERROR] LODO_expw failed & exit /b 1 )

echo.
echo ============================================================
echo Run 4/4  LODO holdout = FER2013
echo ============================================================
python %SCRIPT% --csv lodo_HOLDOUT_fer2013.csv %COMMON_ARGS% --run_name LODO_fer2013
if errorlevel 1 ( echo [ERROR] LODO_fer2013 failed & exit /b 1 )

echo.
echo ============================================================
echo All LODO runs complete.
echo Now run:  python summarize_lodo.py
echo ============================================================
