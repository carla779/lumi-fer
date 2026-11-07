LUMI FER ablation study — README
=================================

This zip contains FOUR scripts.  It does NOT contain the training data,
the image datasets, or the trained checkpoints — those need to already
be on the GPU machine you run this on.

What needs to ALREADY be on the GPU machine
-------------------------------------------
1) A Python environment with:
       torch, torchvision, timm, numpy, pandas, scikit-learn, pillow
   (a CUDA-capable PyTorch build for the GPU to be used)

2) The unified training CSV used to train v9, in the same folder where
   you will run the .bat — by default the script expects:
       unified_fer7_processed_au_FRIEND.csv
   If yours is named differently, open run_ablation.bat in Notepad and
   edit the line:    set CSV=...

3) The image files referenced by the "image" column of that CSV
   (RAF-DB, FERPlus, ExpW etc.).  The CSV uses absolute Windows paths,
   so the dataset folder layout must match what was used to build the
   CSV.  If the paths point to a different drive, you may need to either
   (a) copy the dataset folder to that drive, or (b) edit the CSV to
   rewrite the image paths.

4) The v9 baseline checkpoint and its meta in:
       models_cnn_multi/fer7_v9_5class_mixup_best.pt
       models_cnn_multi/fer7_v9_5class_mixup_meta.json
   The summarizer needs these to compute the baseline row.

Files in this zip
-----------------
  cnn_train_multidataset_fer7_soft_au.py   the training script (same as on Carla's
                                           original machine; reproduced here so the
                                           run_ablation.bat can call it)
  make_hardlabel_csv.py                    builds a hard-label version of the CSV
  run_ablation.bat                         runs all four ablation variants
                                           sequentially
  summarize_ablation.py                    re-evaluates v9 + every ablation on the
                                           held-out test split and writes a
                                           markdown table
  ablation_README.txt                      this file

How to run
----------
1) Drop EVERY file in this zip into your PyCharm project folder
   (the same folder that already contains the cnn_webcam_*.py scripts and
   the models_cnn_multi/ subfolder).

2) Open Anaconda Prompt (or any shell with the right Python env active),
   cd into that folder.

3) Kick off the four training runs:
       run_ablation.bat
   This trains four variants (each ~1.5–2.5 h on a typical mid-range
   gaming GPU; ~6–10 h total).  Leave the PC running overnight.
   Disable Sleep mode in Power Settings while it runs.

4) When the .bat ends with "All ablation runs complete.", run:
       python summarize_ablation.py

   This re-evaluates everything on the same held-out test split and
   produces:
       ablation_results.csv     <- machine-readable
       ablation_results.md      <- markdown table to send to Carla

5) Send ablation_results.md back to Carla; she'll drop it into the
   report (Section 4 / Implementation).

If something errors out
-----------------------
* "FileNotFoundError" on the CSV  --> open run_ablation.bat in Notepad,
  change "set CSV=..." to your actual CSV filename.

* "CUDA out of memory"  --> in run_ablation.bat, change every "--batch 32"
  to "--batch 16" and re-run.

* One of the runs crashes mid-way  --> the others still ran.  Re-run
  just the failed one by copying its single python ... line out of the
  .bat file and pasting it into the shell directly.

* GPU not detected  --> torch.cuda.is_available() must be True.  If your
  PyTorch is CPU-only, install the CUDA build: visit pytorch.org for the
  exact command for your CUDA version.
