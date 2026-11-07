"""
fer2013_appender.py — append FER2013 rows to a unified CSV produced by
build_multidataset_fer7_csv_v2.py.

The existing build script does not load FER2013. This script walks
DATASETS/fer-2013/train/<emotion>/*.jpg and DATASETS/fer-2013/test/<emotion>/*.jpg,
infers labels from the parent folder, carves a stratified val split out of train,
and appends rows to the unified CSV with all the expected columns.

Usage:
    python fer2013_appender.py --in_csv <existing> --out_csv <new>

Schema written for FER2013 rows:
    image, label, split, dataset='fer2013', uid='fer2013_<stem>',
    p_*=NaN (no soft labels), bbox_*=NaN (will be filled by preprocess),
    orig_image=image (placeholder), proc_ok=NaN, proc_method=NaN,
    au_*=NaN (will be filled by extract_auproxy), au_ok=0
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATA_ROOT = r"C:\Users\super\PyCharmMiscProject\DATASETS\fer2013"
FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def build_fer2013_rows(data_root: str, val_ratio: float = 0.10, seed: int = 13) -> pd.DataFrame:
    rows = []
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"FER2013 data root not found: {root}")

    # Walk train/<emotion>/* and test/<emotion>/*
    for split_dir, split_label in [("train", "train"), ("test", "test")]:
        sd = root / split_dir
        if not sd.exists():
            print(f"[WARN] {sd} missing; skipping")
            continue
        for emotion_dir in sd.iterdir():
            if not emotion_dir.is_dir():
                continue
            label = emotion_dir.name.lower()
            if label not in FER7:
                # Allow common renames
                label = label.replace("happiness", "happy").replace("sadness", "sad").replace("surprised", "surprise").replace("anger", "angry")
                if label not in FER7:
                    print(f"[WARN] unknown FER2013 label folder: {emotion_dir.name}; skipping")
                    continue
            for f in emotion_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    rows.append({
                        "image": str(f.resolve()),
                        "label": label,
                        "split": split_label,
                        "dataset": "fer2013",
                        "uid": f"fer2013_{f.stem}",
                    })
    df = pd.DataFrame(rows)
    print(f"[INFO] FER2013 rows discovered: train={int((df['split']=='train').sum())} "
          f"test={int((df['split']=='test').sum())}")

    # Carve a stratified val split out of train
    if val_ratio > 0 and (df["split"] == "train").any():
        train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
        test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        for tr_idx, va_idx in sss.split(train_df["image"].values, train_df["label"].values):
            train_df.loc[va_idx, "split"] = "val"
        df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"[INFO] After val carve: train={int((df['split']=='train').sum())} "
              f"val={int((df['split']=='val').sum())} test={int((df['split']=='test').sum())}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Existing unified CSV (without fer2013)")
    ap.add_argument("--out_csv", required=True, help="Where to write the augmented CSV")
    ap.add_argument("--data_root", default=DATA_ROOT, help="Path to DATASETS/fer-2013/")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if not os.path.isfile(args.in_csv):
        print(f"[ERROR] in_csv not found: {args.in_csv}")
        sys.exit(1)

    print(f"[INFO] Loading existing CSV: {args.in_csv}")
    base = pd.read_csv(args.in_csv, low_memory=False)
    print(f"[INFO] Base rows: {len(base)}")

    print(f"[INFO] Walking FER2013: {args.data_root}")
    fer = build_fer2013_rows(args.data_root, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[INFO] FER2013 rows: {len(fer)}")

    # Align columns
    for col in base.columns:
        if col not in fer.columns:
            fer[col] = np.nan
    # uid column may not exist in base; ensure presence
    if "uid" not in base.columns:
        base["uid"] = ""
    fer["orig_image"] = fer["image"]
    if "au_ok" in base.columns:
        fer["au_ok"] = 0
    if "proc_ok" in base.columns:
        fer["proc_ok"] = np.nan

    fer = fer[base.columns]  # column order match

    out = pd.concat([base, fer], ignore_index=True)
    print(f"[INFO] Augmented total: {len(out)}")
    print(f"[INFO] Per-dataset row count after append:")
    print(out["dataset"].astype(str).str.lower().value_counts().to_string())

    out.to_csv(args.out_csv, index=False)
    print(f"\n[OK] Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
