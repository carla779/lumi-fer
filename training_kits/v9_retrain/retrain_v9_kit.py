"""
retrain_v9_kit.py — orchestrator for v9 retraining on the gaming PC.

Subcommands
-----------
  rebuild   Full Path C: fix preprocessor, rebuild unified CSV with all 4 datasets,
            re-run preprocessing (with --resume to keep existing ExpW), re-extract
            AU columns. End-to-end ~2-3 hours.
  verify    Sanity-check disk + CSV.
  patch     Light patch: substitute orig_image for image=nan.jpg (Path A only;
            do not use after rebuild).
  train     Launch the v9 retraining run (~4-6 h on a modern GPU).
  eval      Compute test-split macro-F1 on the retrained best.pt.
  all       rebuild -> verify -> train -> eval (Path C end-to-end, ~7-9 hours).

Examples
--------
  python retrain_v9_kit.py rebuild
  python retrain_v9_kit.py verify
  python retrain_v9_kit.py train
  python retrain_v9_kit.py eval
  python retrain_v9_kit.py all
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# -------------------- paths (edit if your layout differs) --------------------
PROJ = r"C:\Users\super\PyCharmMiscProject"
# On this gaming PC, EVERYTHING for the four datasets lives under processed_faces/.
# Each dataset folder contains BOTH the source data (label CSVs, original images)
# AND the cropped 224x224 outputs from preprocessing:
#   processed_faces/rafdb/      train_labels.csv, test_labels.csv, DATASET/, train/, test/, val/
#   processed_faces/ferplus/    FERPlusData/FER2013Train/label.csv (etc.) + cropped outputs
#   processed_faces/expw/       label.lst + origin/ + cropped outputs
#   processed_faces/fer2013/    train/<emotion>/ + test/<emotion>/ + cropped outputs
DATASETS_ROOT = os.path.join(PROJ, "processed_faces")
PROC = os.path.join(PROJ, "processed_faces")
CSV = os.path.join(PROJ, "unified_fer7_processed_au_FRIEND.csv")
TRAINER = os.path.join(PROJ, "cnn_train_multidataset_fer7_soft_au.py")
MODEL_DIR = os.path.join(PROJ, "models_cnn_multi")
RUN_NAME = "fer7_v9_5class_mixup_RETRAIN"
DATASETS = ["rafdb", "ferplus", "expw", "fer2013"]

# Kit-local files (this folder)
KIT_DIR = os.path.dirname(os.path.abspath(__file__))
FER2013_APPENDER = os.path.join(KIT_DIR, "fer2013_appender.py")
PATCH_PREPROCESS = os.path.join(KIT_DIR, "patch_preprocess.py")


def _resolve(name):
    """Return the path to a pipeline script, preferring PROJ and falling
    back to the kit folder if PROJ doesn't have it."""
    proj_path = os.path.join(PROJ, name)
    kit_path = os.path.join(KIT_DIR, name)
    if os.path.isfile(proj_path):
        return proj_path
    if os.path.isfile(kit_path):
        return kit_path
    # Last resort: return the proj path so the eventual error is clear
    return proj_path


BUILDER = _resolve("build_multidataset_fer7_csv_v2.py")
PREPROCESSOR = _resolve("preprocess_faces_from_csv_v3.py")
AU_EXTRACTOR = _resolve("extract_auproxy_facemesh_from_csv.py")


def banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70)


def run(cmd, cwd=None, label=None):
    """Run a subprocess command, surface return code, stop on error."""
    if label:
        print(f"\n>>> {label}")
    print(">>> " + " ".join(repr(c) if " " in str(c) else str(c) for c in cmd))
    rc = subprocess.call(cmd, cwd=cwd or PROJ)
    if rc != 0:
        print(f"\n[ERROR] command exited with code {rc}")
        sys.exit(rc)


# =============================================================================
# verify
# =============================================================================
def cmd_verify():
    import pandas as pd

    banner("Step: processed_faces/ on disk")
    ok_disk = True
    for d in DATASETS:
        p = os.path.join(PROC, d)
        if not os.path.isdir(p):
            print(f"  [MISSING] folder: {p}")
            ok_disk = False
            continue
        n = sum(len(files) for _, _, files in os.walk(p))
        flag = "[OK]   " if n >= 100 else "[BAD]  "
        print(f"  {flag} {d:10s} {n:>8d} files in {p}")
        if n < 100:
            ok_disk = False

    banner("Step: unified CSV")
    if not os.path.isfile(CSV):
        print(f"  [MISSING] {CSV}")
        return 1
    df = pd.read_csv(CSV, low_memory=False)
    df["dataset"] = df["dataset"].astype(str).str.lower()
    print(f"  Rows: {len(df)}")
    print("  Per-dataset row count:")
    counts = df["dataset"].value_counts().to_dict()
    for d in DATASETS:
        flag = "[OK]   " if counts.get(d, 0) >= 100 else "[BAD]  "
        print(f"    {flag} {d:10s}: {counts.get(d, 0)}")

    banner("Step: per-dataset image-path resolution")
    exists = df["image"].apply(lambda p: os.path.isfile(str(p)))
    res = df.assign(ok=exists).groupby("dataset")["ok"].agg(["sum", "count"])
    print(res.to_string())
    print()

    ok_paths = True
    for d in DATASETS:
        if d not in res.index:
            print(f"  [MISSING] {d}: not present in CSV")
            ok_paths = False
            continue
        s, c = int(res.loc[d, "sum"]), int(res.loc[d, "count"])
        pct = 100 * s / max(c, 1)
        if c == 0:
            print(f"  [MISSING] {d}: zero rows in CSV")
            ok_paths = False
        elif pct < 95:
            print(f"  [BROKEN]  {d}: only {s}/{c} ({pct:.1f}%) image paths resolve")
            ok_paths = False
        else:
            print(f"  [OK]      {d}: {s}/{c} ({pct:.1f}%) resolve")

    banner("Summary")
    if ok_disk and ok_paths:
        print("  [SUCCESS] Corpus ready for training.")
        return 0
    print("  [FAIL] Corpus has issues.")
    if not ok_disk:
        print("    - Some processed_faces folders are missing or near-empty.")
    if not ok_paths:
        print("    - Some image paths in the CSV do not resolve.")
        print("    - Run `python retrain_v9_kit.py rebuild` to do a clean rebuild")
        print("      OR `python retrain_v9_kit.py patch` for the lightweight fix.")
    return 1


# =============================================================================
# patch (light: substitute orig_image for image=nan.jpg)
# =============================================================================
def cmd_patch():
    import pandas as pd

    banner("Patching CSV: nan.jpg -> orig_image")
    if not os.path.isfile(CSV):
        print(f"[ERROR] CSV not found: {CSV}")
        return 1
    backup = CSV.replace(".csv", "_BACKUP.csv")
    if not os.path.isfile(backup):
        shutil.copy(CSV, backup)
        print(f"[OK] Backup written to {backup}")
    else:
        print(f"[OK] Backup already exists; not overwriting")

    df = pd.read_csv(CSV, low_memory=False)
    n_before = len(df)
    mask = df["image"].astype(str).str.contains(r"nan\.jpg$", regex=True, na=False)
    n_broken = int(mask.sum())
    print(f"[INFO] Rows with broken image path: {n_broken} ({100*n_broken/max(n_before,1):.1f}% of {n_before})")
    if n_broken > 0:
        if "orig_image" not in df.columns:
            print("[ERROR] orig_image column missing")
            return 1
        df.loc[mask, "image"] = df.loc[mask, "orig_image"]
        df.to_csv(CSV, index=False)
        print(f"[OK] Substituted on {n_broken} rows")
    else:
        print("[OK] Nothing to substitute")
    return 0


# =============================================================================
# rebuild — Path C: fix preprocessor, rebuild CSV with all 4 datasets,
#                  re-run preprocessing, re-extract AU
# =============================================================================
def cmd_rebuild():
    banner("PATH C STEP 1/5  Patch preprocessor (uid NaN guard)")
    if not os.path.isfile(PATCH_PREPROCESS):
        print(f"[ERROR] patch_preprocess.py not found in kit: {PATCH_PREPROCESS}")
        return 1
    run([sys.executable, PATCH_PREPROCESS], cwd=KIT_DIR, label="apply uid-NaN fix")

    # Backup the existing FRIEND.csv
    if os.path.isfile(CSV):
        bk = CSV.replace(".csv", "_BACKUP_PathC.csv")
        if not os.path.isfile(bk):
            shutil.copy(CSV, bk)
            print(f"[OK] Backed up old CSV to {bk}")

    banner("PATH C STEP 2/5  Build CSV from RAF-DB + FERPlus + ExpW (~5 min)")
    intermediate = os.path.join(PROJ, "_unified_3srcs.csv")
    expw_images_root = os.path.join(DATASETS_ROOT, "expw", "origin")
    run([
        sys.executable, BUILDER,
        "--datasets_root", DATASETS_ROOT,
        "--out_csv", intermediate,
        "--ferplus_min_votes7", "3",      # lower from 6 to recover ferplus rows
        "--expw_images_root", expw_images_root,  # tell build_*.py where ExpW images live
    ], label="build 3-source unified CSV (with relaxed FERPlus filter)")

    banner("PATH C STEP 3/5  Append FER2013 rows (~30 sec)")
    intermediate2 = os.path.join(PROJ, "_unified_4srcs.csv")
    run([
        sys.executable, FER2013_APPENDER,
        "--in_csv", intermediate,
        "--out_csv", intermediate2,
        "--data_root", os.path.join(DATASETS_ROOT, "fer2013"),
    ], label="append FER2013 rows from train/<emotion>/ + test/<emotion>/")

    banner("PATH C STEP 4/5  Preprocess faces (--resume; ~1-2 hours)")
    intermediate3 = os.path.join(PROJ, "_unified_4srcs_processed.csv")
    run([
        sys.executable, PREPROCESSOR,
        "--in_csv", intermediate2,
        "--out_dir", PROC,
        "--out_csv", intermediate3,
        "--save_size", "224",
        "--face_margin", "0.30",
        "--resume",
    ], label="preprocess (skips existing files)")

    banner("PATH C STEP 5/5  Extract AU proxy columns (~30-60 min)")
    run([
        sys.executable, AU_EXTRACTOR,
        "--in_csv", intermediate3,
        "--out_csv", CSV,                 # overwrite the FRIEND.csv with the fresh one
        "--splits", "train", "val", "test",
    ], label="extract MediaPipe AU proxies")

    banner("Rebuild complete")
    print(f"[OK] Final CSV at: {CSV}")
    print(f"     Old CSV preserved at: {CSV.replace('.csv', '_BACKUP_PathC.csv')}")
    print()
    print("Next: python retrain_v9_kit.py verify   (sanity-check the rebuilt CSV)")
    print("      python retrain_v9_kit.py train    (kick off the 50-epoch run)")
    return 0


# =============================================================================
# train
# =============================================================================
def cmd_train():
    banner(f"Retraining v9: {RUN_NAME}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Trainer: {TRAINER}")
    print(f"CSV    : {CSV}")
    print(f"Output : {MODEL_DIR}")
    print()
    print("Hyperparameters (matches existing v9 meta + 50 epochs):")
    print("  arch=efficientnet_b0  img_size=224  batch=32  epochs=50")
    print("  lr=1e-4  lambda_au=0.05  mixup_alpha=0.2  smoothing=0.05")
    print("  balance=dataset_class  freeze_epochs=5  dropout=0.2")
    print()

    cmd = [
        sys.executable, TRAINER,
        "--csv", CSV,
        "--arch", "efficientnet_b0",
        "--pretrained",
        "--img_size", "224",
        "--batch", "32",
        "--epochs", "50",
        "--lr", "1e-4",
        "--num_workers", "2",
        "--save_dir", MODEL_DIR,
        "--lambda_au", "0.05",
        "--mixup_alpha", "0.2",
        "--balance", "dataset_class",
        "--smoothing", "0.05",
        "--freeze_epochs", "5",
        "--dropout", "0.2",
        "--run_name", RUN_NAME,
    ]
    print("Command:")
    print("  " + " ".join(cmd))
    print()
    try:
        rc = subprocess.call(cmd, cwd=PROJ)
    except KeyboardInterrupt:
        print("\n[ABORT] Training interrupted by user.")
        return 1
    print(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}")
    if rc != 0:
        print(f"[ERROR] Trainer exited with code {rc}")
        return rc
    print("[OK] Training complete. Run `python retrain_v9_kit.py eval` next.")
    return 0


# =============================================================================
# eval
# =============================================================================
def cmd_eval():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image, ImageFile
    from sklearn.metrics import classification_report, f1_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    meta_path = os.path.join(MODEL_DIR, RUN_NAME + "_meta.json")
    ckpt_path = os.path.join(MODEL_DIR, RUN_NAME + "_best.pt")
    if not (os.path.isfile(meta_path) and os.path.isfile(ckpt_path)):
        print("[ERROR] Missing meta or checkpoint:")
        print(f"  meta: {meta_path}  exists={os.path.isfile(meta_path)}")
        print(f"  ckpt: {ckpt_path}  exists={os.path.isfile(ckpt_path)}")
        return 1
    meta = json.load(open(meta_path))
    classes = meta["classes"]
    img_size = int(meta["img_size"])
    print(f"[META] arch={meta.get('arch')} classes={classes} img_size={img_size}")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            b = models.efficientnet_b0(weights=None)
            feat = b.classifier[-1].in_features
            b.classifier[-1] = nn.Identity()
            self.backbone = b
            self.dropout = nn.Dropout(p=float(meta.get("dropout", 0.2)))
            self.cls_head = nn.Linear(feat, len(classes))

        def forward(self, x):
            return self.cls_head(self.dropout(self.backbone(x)))

    class DS(Dataset):
        def __init__(self, df, sz):
            self.df = df.reset_index(drop=True)
            self.lab2id = {c: i for i, c in enumerate(classes)}
            self.tf = transforms.Compose([
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ])

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            r = self.df.iloc[i]
            return self.tf(Image.open(str(r["image"])).convert("RGB")), self.lab2id[str(r["label"]).lower()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    if device == "cpu":
        print("[WARN] No CUDA detected; eval on CPU will be slow")

    model = M().to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    df = pd.read_csv(CSV, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    df["label"] = df["label"].astype(str).str.lower()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df = df[df["label"].isin(classes)]
    df = df[df["image"].apply(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)
    print(f"[EVAL] {len(df)} test rows after filtering")
    if len(df) == 0:
        print("[ERROR] Zero test rows")
        return 1

    dl = DataLoader(DS(df, img_size), batch_size=64, shuffle=False, num_workers=2)
    ys, yp = [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(device))
            ys.extend(y.numpy().tolist())
            yp.extend(torch.argmax(out, dim=1).cpu().numpy().tolist())

    ys_arr = np.array(ys)
    yp_arr = np.array(yp)
    pooled = f1_score(ys_arr, yp_arr, average="macro", zero_division=0)
    print(f"\n[RESULT] Pooled macro-F1 across all test rows: {pooled:.4f}\n")
    print(classification_report(ys_arr, yp_arr, target_names=classes, zero_division=0, digits=3))

    print("\n[PER-DATASET] macro-F1 on each dataset's test rows:")
    for d in sorted(df["dataset"].unique()):
        mask = (df["dataset"] == d).values
        if mask.sum() == 0:
            continue
        f1 = f1_score(ys_arr[mask], yp_arr[mask], average="macro", zero_division=0)
        print(f"  {d:10s}  N={int(mask.sum()):>5d}   macro-F1={f1:.4f}")
    return 0


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["rebuild", "verify", "patch", "train", "eval", "all"])
    args = ap.parse_args()

    if args.step == "rebuild":
        sys.exit(cmd_rebuild())
    if args.step == "verify":
        sys.exit(cmd_verify())
    if args.step == "patch":
        sys.exit(cmd_patch())
    if args.step == "train":
        sys.exit(cmd_train())
    if args.step == "eval":
        sys.exit(cmd_eval())
    if args.step == "all":
        for fn in (cmd_rebuild, cmd_verify, cmd_train, cmd_eval):
            rc = fn()
            if rc != 0:
                print(f"\n[ABORT] step `{fn.__name__}` returned {rc}")
                sys.exit(rc)
        print("\n[OK] All steps complete.")


if __name__ == "__main__":
    main()
