v9 Retraining Kit
=================

This kit retrains v9 on the corrected 4-source FER corpus (RAF-DB +
FERPlus + ExpW + FER2013) and reports test-split macro-F1.


PREREQUISITES (gaming PC)
-------------------------
- Python 3.x with: torch (with CUDA), torchvision, pandas, scikit-learn,
  Pillow, numpy. (You already have these installed for your existing
  training pipeline.)
- Path layout assumed:
    C:\Users\super\PyCharmMiscProject\
        unified_fer7_processed_au_FRIEND.csv
        cnn_train_multidataset_fer7_soft_au.py
        processed_faces\
            rafdb\
            ferplus\
            expw\
            fer2013\
        models_cnn_multi\          <- created if missing


HOW TO USE
----------
Step-by-step is recommended for the first run; do_everything.bat is for
unattended re-runs once you trust the pipeline.

  1. Copy this entire v9_retrain_kit\ folder to a location on the gaming
     PC you can run scripts from. Anywhere works; e.g.:
         C:\Users\super\PyCharmMiscProject\v9_retrain_kit\

  2. Open a Command Prompt (cmd.exe) in the v9_retrain_kit\ folder.

  3. Run the steps in order:

     a) 1_verify_and_patch.bat
        - Confirms the four datasets are on disk and in the CSV.
        - Patches the CSV's broken nan.jpg paths (substitutes
          orig_image, which actually exists for rafdb/ferplus rows).
        - Patches the trainer to print a [loader] line so silent row
          drops become visible.
        - Re-verifies. Pauses; review output before proceeding.

     b) 2_train_v9.bat
        - Trains EfficientNet-B0 with AU loss + soft labels + MixUp +
          dataset-class balanced sampler for 50 epochs.
        - Run name: fer7_v9_5class_mixup_RETRAIN.
        - Walk away. Expect 4-6 hours on a modern GPU.

     c) 3_eval_v9.bat
        - Loads the retrained best.pt and evaluates on the held-out
          test split. Prints pooled macro-F1, per-class F1, and
          per-dataset macro-F1.

  4. Send Claude:
     - Final eval output (pooled macro-F1 + per-class report + per-
       dataset breakdown).
     - Contents of:
         models_cnn_multi\fer7_v9_5class_mixup_RETRAIN_meta.json
     - The [loader] lines printed at the start of training (these
       confirm zero rows were silently dropped).


WHAT TO EXPECT (sanity checks)
------------------------------
After 1_verify_and_patch.bat:
  - All four datasets show >=95% path resolution.
  - "[OK] All datasets have >=95% resolution after patch."

At the start of 2_train_v9.bat:
  - "[loader] 0 rows dropped..." (or a small number; not 17,000+).
  - "[loader] per-dataset rows kept: {'expw': ~88K, 'rafdb': ~12K,
    'ferplus': ~30K, 'fer2013': ~XK}" — exact counts depend on your
    splits.
  - First epoch's val macro-F1 should be > 0.20 within minutes.

After 3_eval_v9.bat:
  - Pooled macro-F1 in the 0.55-0.65 range is the most likely outcome.
  - 0.65-0.72 is optimistic but possible.
  - <0.50 indicates a problem; send the output to Claude before
    interpreting.


FILES IN THIS KIT
-----------------
  README.txt                this file
  do_everything.bat         runs 1, 2, 3 in sequence
  1_verify_and_patch.bat    verify + patch CSV + patch trainer
  2_train_v9.bat            launches the retraining run
  3_eval_v9.bat             evaluates the retrained checkpoint
  verify.py                 corpus sanity checks
  patch_csv.py              substitutes orig_image for image=nan.jpg
  patch_trainer.py          adds [loader] diagnostic prints
  eval_v9.py                test-split macro-F1 computation


WHAT GETS BACKED UP
-------------------
  unified_fer7_processed_au_FRIEND_BACKUP.csv     (created by patch_csv.py)
  cnn_train_multidataset_fer7_soft_au.py.bak     (created by patch_trainer.py)

Both are created on first run only; subsequent runs are no-ops.


THINGS THAT CAN GO WRONG
------------------------
- "CUDA out of memory" during training: edit 2_train_v9.bat and reduce
  --batch from 32 to 16.
- "fer2013 not in CSV": the unified CSV needs to be rebuilt to include
  the fer2013 rows (run your build_multidataset_fer7_csv_v2.py with the
  fer2013 source, then re-run this kit).
- "<95% resolution" warning after patch: some processed_faces files are
  genuinely missing on disk. The trainer will silently drop those rows;
  you'll see a non-zero count in the [loader] line at training start.
- Training runs but per-class F1 is very imbalanced (e.g., happy 0.85,
  angry 0.05): class-balanced sampler may not be doing its job. Send
  the [loader] line and the per-class report; Claude can advise.

Send any error mid-run; do not let it run for 4 hours producing junk.
