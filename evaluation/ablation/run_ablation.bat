@echo off
REM =====================================================================
REM  LUMI FER ablation runner — Windows / gaming-PC GPU.
REM
REM  Trains 4 ablation variants of the v9 EfficientNet-B0 5-class pipeline,
REM  each different from the full pipeline by one component:
REM     ABLATE_NO_AU       --lambda_au 0          (drops Action Unit aux loss)
REM     ABLATE_NO_MIXUP    --mixup_alpha 0        (drops MixUp augmentation)
REM     ABLATE_NO_BALANCE  --balance none         (drops dataset-class sampler)
REM     ABLATE_HARD_LABEL  hard-label CSV         (drops soft FERPlus targets)
REM
REM  Each run uses 30 epochs (instead of 50) to keep wall-clock under ~2 h
REM  per run on a single mid-range GPU. The original v9 50-epoch checkpoint
REM  is used as the *baseline* row in the ablation table — DO NOT retrain.
REM
REM  Adjust paths at the top if your folder layout differs.
REM =====================================================================

REM ---------- EDIT THESE THREE PATHS IF NEEDED ----------
set SCRIPT=cnn_train_multidataset_fer7_soft_au.py
set CSV=unified_fer7_processed_au_FRIEND.csv
set HARD_CSV=unified_fer7_processed_au_HARDLABELS.csv
REM ------------------------------------------------------

set COMMON=--csv %CSV% --arch efficientnet_b0 --pretrained --img_size 224 --batch 32 --epochs 30 --lr 1e-4 --num_workers 2 --save_dir models_ablation

echo ============================================================
echo Step 0: build hard-label CSV (strips FERPlus soft probabilities)
echo ============================================================
python make_hardlabel_csv.py --in %CSV% --out %HARD_CSV%
if errorlevel 1 ( echo [ERROR] failed to build hard-label CSV & exit /b 1 )

echo.
echo ============================================================
echo Run 1/4  ABLATE_NO_AU       (no Action Unit auxiliary loss)
echo ============================================================
python %SCRIPT% %COMMON% --lambda_au 0    --mixup_alpha 0.2 --balance dataset_class --run_name ABLATE_NO_AU
if errorlevel 1 ( echo [ERROR] ABLATE_NO_AU failed & exit /b 1 )

echo.
echo ============================================================
echo Run 2/4  ABLATE_NO_MIXUP    (no MixUp augmentation)
echo ============================================================
python %SCRIPT% %COMMON% --lambda_au 0.05 --mixup_alpha 0   --balance dataset_class --run_name ABLATE_NO_MIXUP
if errorlevel 1 ( echo [ERROR] ABLATE_NO_MIXUP failed & exit /b 1 )

echo.
echo ============================================================
echo Run 3/4  ABLATE_NO_BALANCE  (no dataset-class balanced sampler)
echo ============================================================
python %SCRIPT% %COMMON% --lambda_au 0.05 --mixup_alpha 0.2 --balance none --run_name ABLATE_NO_BALANCE
if errorlevel 1 ( echo [ERROR] ABLATE_NO_BALANCE failed & exit /b 1 )

echo.
echo ============================================================
echo Run 4/4  ABLATE_HARD_LABEL  (hard one-hot labels, no FERPlus soft)
echo ============================================================
python %SCRIPT% --csv %HARD_CSV% --arch efficientnet_b0 --pretrained --img_size 224 --batch 32 --epochs 30 --lr 1e-4 --num_workers 2 --save_dir models_ablation --lambda_au 0.05 --mixup_alpha 0.2 --balance dataset_class --run_name ABLATE_HARD_LABEL
if errorlevel 1 ( echo [ERROR] ABLATE_HARD_LABEL failed & exit /b 1 )

echo.
echo ============================================================
echo All ablation runs complete.
echo Now run:  python summarize_ablation.py
echo ============================================================
