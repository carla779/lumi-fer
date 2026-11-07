@echo off
REM ============================================================
REM  Retrain v9 from scratch on the corrected 4-source corpus.
REM  Config matches the existing v9 meta plus 50 epochs.
REM  Expect 4-6 hours on a modern GPU.
REM
REM  Output: models_cnn_multi\fer7_v9_5class_mixup_RETRAIN_*
REM  A copy of stdout is also written next to the checkpoint.
REM ============================================================
setlocal
cd /d C:\Users\super\PyCharmMiscProject

set RUN_NAME=fer7_v9_5class_mixup_RETRAIN

if not exist models_cnn_multi mkdir models_cnn_multi

echo.
echo Starting v9 retrain: %DATE% %TIME%
echo Run name: %RUN_NAME%
echo.
echo Hyperparameters (matches existing v9 meta + 50 epochs):
echo   arch=efficientnet_b0  img_size=224  batch=32  epochs=50
echo   lr=1e-4  lambda_au=0.05  mixup_alpha=0.2  smoothing=0.05
echo   balance=dataset_class  freeze_epochs=5  dropout=0.2
echo.
echo Watching for [loader] lines at start of training to confirm
echo zero rows are silently dropped due to broken image paths.
echo.

python cnn_train_multidataset_fer7_soft_au.py ^
    --csv unified_fer7_processed_au_FRIEND.csv ^
    --arch efficientnet_b0 ^
    --pretrained ^
    --img_size 224 ^
    --batch 32 ^
    --epochs 50 ^
    --lr 1e-4 ^
    --num_workers 2 ^
    --save_dir models_cnn_multi ^
    --lambda_au 0.05 ^
    --mixup_alpha 0.2 ^
    --balance dataset_class ^
    --smoothing 0.05 ^
    --freeze_epochs 5 ^
    --dropout 0.2 ^
    --run_name %RUN_NAME%

if errorlevel 1 (
    echo.
    echo ============================================================
    echo  [ERROR] Training failed. Capture the error and send it.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Training complete: %DATE% %TIME%
echo  Run 3_eval_v9.bat next to compute test-split macro-F1.
echo ============================================================
pause
