import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ---------- helpers ----------
def make_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unsupported arch: {arch}")


def build_image_index(images_root: str) -> Dict[str, str]:
    idx = {}
    for dirpath, _, files in os.walk(images_root):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                if fn not in idx:
                    idx[fn] = os.path.join(dirpath, fn)
    return idx


def auto_detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}
    path_candidates = [c for c in cols if any(k in lower[c] for k in ["path", "file", "image", "img", "name"])]
    label_candidates = [c for c in cols if any(k in lower[c] for k in ["label", "emotion", "class", "category"])]

    if not path_candidates:
        raise RuntimeError(f"Could not detect image column. Columns: {cols}")
    path_col = path_candidates[0]

    if not label_candidates:
        if len(cols) >= 2:
            label_col = cols[1]
        else:
            raise RuntimeError(f"Could not detect label column. Columns: {cols}")
    else:
        label_col = label_candidates[0]

    return path_col, label_col


# mappings used in your RAF-DB CSV approach
NUM0_KAGGLE = {0:"angry",1:"disgust",2:"fear",3:"happy",4:"sad",5:"surprise",6:"neutral"}
NUM1_OFFICIAL = {1:"surprise",2:"fear",3:"disgust",4:"happy",5:"sad",6:"angry",7:"neutral"}


def normalize_label(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip().lower()
    if s == "fearful":
        s = "fear"
    if s == "surprised":
        s = "surprise"
    return s


def map_label_to_index(v, classes: List[str], numeric_scheme: str) -> int:
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    s = normalize_label(v)

    if s in cls_to_idx:
        return cls_to_idx[s]

    try:
        iv = int(v)
    except Exception:
        return -1

    if numeric_scheme == "kaggle0":
        name = NUM0_KAGGLE.get(iv)
    else:
        name = NUM1_OFFICIAL.get(iv)

    return cls_to_idx.get(name, -1)


def resolve_image_path(raw: str, images_root: str, img_index: Dict[str, str]) -> Optional[str]:
    raw = str(raw).strip()

    if os.path.isfile(raw):
        return raw

    p = os.path.join(images_root, raw)
    if os.path.isfile(p):
        return p

    base = os.path.basename(raw)
    if base in img_index:
        return img_index[base]

    # try aligned suffix if dataset uses it
    stem, ext = os.path.splitext(base)
    for ext2 in [ext, ".jpg", ".png", ".jpeg"]:
        cand = f"{stem}_aligned{ext2}"
        if cand in img_index:
            return img_index[cand]

    return None


@dataclass
class ImgRow:
    idx: int
    path: str
    y: int


class ImgDataset(Dataset):
    def __init__(self, rows: List[ImgRow], tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r.path).convert("RGB")
        x = self.tf(img)
        return x, r.y, r.idx


@torch.no_grad()
def predict_all(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_y = []
    all_idx = []

    for x, y, idx in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.numpy())
        all_idx.append(idx.numpy())

    return np.concatenate(all_probs, 0), np.concatenate(all_y, 0), np.concatenate(all_idx, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=r"models_cnn_raf7\raf7_csv_meta.json")
    ap.add_argument("--ckpt", default=r"models_cnn_raf7\raf7_csv_best.pt")
    ap.add_argument("--out_clean", default=r"RAF-DB\train_labels_clean.csv")
    ap.add_argument("--out_dropped", default=r"RAF-DB\train_labels_dropped_noise.csv")

    ap.add_argument("--focus", nargs="+", default=["angry", "disgust", "fear", "sad"],
                    help="Which classes to clean most aggressively (usually negative emotions).")
    ap.add_argument("--drop_frac", type=float, default=0.02,
                    help="Max fraction to drop PER class from focus set (start small: 0.01–0.03).")

    ap.add_argument("--conf_min", type=float, default=0.90,
                    help="Only flag if model is very confident in a different class.")
    ap.add_argument("--margin_min", type=float, default=0.20,
                    help="Only flag if top1-top2 margin is large (high certainty).")

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)  # safest on Windows
    args = ap.parse_args()

    meta = json.load(open(args.meta, "r", encoding="utf-8"))
    classes = meta["classes"]
    arch = meta["arch"]
    img_size = int(meta["img_size"])

    dataset_root = meta["dataset_root"]
    images_root = meta["images_root"]
    train_labels_path = meta["train_labels"]
    numeric_scheme = meta.get("numeric_scheme", "official1")

    print("[PATH] train_labels:", train_labels_path)
    print("[PATH] images_root :", images_root)
    print("[META] arch:", arch, "img_size:", img_size, "classes:", classes)
    print("[CFG] focus:", args.focus, "drop_frac:", args.drop_frac, "conf_min:", args.conf_min, "margin_min:", args.margin_min)

    df = pd.read_csv(train_labels_path) if train_labels_path.lower().endswith(".csv") else pd.read_excel(train_labels_path)
    path_col, label_col = auto_detect_columns(df)

    img_index = build_image_index(images_root)

    rows: List[ImgRow] = []
    bad = 0
    for i, row in df.iterrows():
        y = map_label_to_index(row[label_col], classes, numeric_scheme)
        if y < 0:
            bad += 1
            continue
        p = resolve_image_path(row[path_col], images_root, img_index)
        if not p:
            bad += 1
            continue
        rows.append(ImgRow(idx=i, path=p, y=y))

    print(f"[DATA] usable rows={len(rows)}  dropped_unreadable={bad}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(arch, num_classes=len(classes)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    loader = DataLoader(ImgDataset(rows, tf), batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers)

    probs, y_true, idxs = predict_all(model, loader, device)

    # compute suspicion score
    pred = probs.argmax(axis=1)
    p_pred = probs[np.arange(len(probs)), pred]
    p_true = probs[np.arange(len(probs)), y_true]

    # top2 margin
    top2 = np.partition(probs, -2, axis=1)[:, -2]
    margin = p_pred - top2

    mismatch = (pred != y_true)
    suspicious = mismatch & (p_pred >= args.conf_min) & (margin >= args.margin_min)

    # a simple score: more confident + larger disagreement
    score = np.zeros(len(probs), dtype=np.float32)
    score[suspicious] = (p_pred[suspicious] - p_true[suspicious]) + margin[suspicious]

    # build a table
    cls_names = np.array(classes)
    report = pd.DataFrame({
        "row_idx": idxs,
        "true_label": cls_names[y_true],
        "pred_label": cls_names[pred],
        "p_true": p_true,
        "p_pred": p_pred,
        "margin": margin,
        "score": score,
        "flagged": suspicious,
    })

    # drop capped amount per focus class
    focus = set(args.focus)
    drop_rows = []

    for c in focus:
        sub = report[(report["true_label"] == c) & (report["flagged"])]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("score", ascending=False)
        k = int(np.ceil(args.drop_frac * (report["true_label"] == c).sum()))
        k = max(1, k)
        drop_rows.extend(sub.head(k)["row_idx"].tolist())

    drop_rows = sorted(set(drop_rows))
    print("[CLEAN] dropping rows:", len(drop_rows))

    # write dropped report
    dropped_df = report[report["row_idx"].isin(drop_rows)].sort_values("score", ascending=False)
    os.makedirs(os.path.dirname(args.out_dropped), exist_ok=True)
    dropped_df.to_csv(args.out_dropped, index=False)
    print("[OK] wrote dropped list ->", args.out_dropped)

    # write cleaned train labels CSV (same columns as original df)
    clean_df = df.drop(index=drop_rows)
    os.makedirs(os.path.dirname(args.out_clean), exist_ok=True)
    clean_df.to_csv(args.out_clean, index=False)
    print("[OK] wrote cleaned train labels ->", args.out_clean)

    # summary
    print("\n[DROPPED SUMMARY]")
    if len(dropped_df) == 0:
        print("No rows were flagged with current thresholds. Try lower conf/margin or increase drop_frac slightly.")
    else:
        print(dropped_df["true_label"].value_counts())
        print("\nTop label->pred confusions (dropped only):")
        print(dropped_df.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()