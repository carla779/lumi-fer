"""
eval_v9.py — evaluate the retrained v9 on the held-out test split.
Prints pooled macro-F1, per-class F1, and per-dataset macro-F1.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJ = r"C:\Users\super\PyCharmMiscProject"
MODEL_DIR = os.path.join(PROJ, "models_cnn_multi")
RUN_NAME = "fer7_v9_5class_mixup_RETRAIN"
CSV = os.path.join(PROJ, "unified_fer7_processed_au_FRIEND.csv")


def main():
    meta_path = os.path.join(MODEL_DIR, RUN_NAME + "_meta.json")
    ckpt_path = os.path.join(MODEL_DIR, RUN_NAME + "_best.pt")

    if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
        print("[ERROR] Missing meta or checkpoint:")
        print(f"  meta: {meta_path}  exists={os.path.isfile(meta_path)}")
        print(f"  ckpt: {ckpt_path}  exists={os.path.isfile(ckpt_path)}")
        sys.exit(1)

    meta = json.load(open(meta_path))
    classes = meta["classes"]
    img_size = int(meta["img_size"])
    print(f"[META] arch={meta.get('arch')} classes={classes} img_size={img_size}")

    # Build model matching the retrained architecture
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
            img = Image.open(str(r["image"])).convert("RGB")
            return self.tf(img), self.lab2id[str(r["label"]).lower()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    if device == "cpu":
        print("[WARN] No CUDA detected; eval on CPU will be slow (still works)")

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
        print("[ERROR] Zero test rows after filtering — check label space and paths.")
        sys.exit(1)

    dl = DataLoader(DS(df, img_size), batch_size=64, shuffle=False, num_workers=2)
    ys, yp = [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(device))
            ys.extend(y.numpy().tolist())
            yp.extend(torch.argmax(out, dim=1).cpu().numpy().tolist())

    ys_arr = np.array(ys)
    yp_arr = np.array(yp)

    # ---------------- Pooled ----------------
    pooled = f1_score(ys_arr, yp_arr, average="macro", zero_division=0)
    print(f"\n[RESULT] Pooled macro-F1 across all test rows: {pooled:.4f}\n")
    print(classification_report(ys_arr, yp_arr, target_names=classes, zero_division=0, digits=3))

    # ---------------- Per-dataset ----------------
    print("\n[PER-DATASET] macro-F1 on each dataset's test rows:")
    for d in sorted(df["dataset"].unique()):
        mask = (df["dataset"] == d).values
        if mask.sum() == 0:
            continue
        f1 = f1_score(ys_arr[mask], yp_arr[mask], average="macro", zero_division=0)
        print(f"  {d:10s}  N={int(mask.sum()):>5d}   macro-F1={f1:.4f}")


if __name__ == "__main__":
    main()
