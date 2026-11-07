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
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.model_selection import StratifiedKFold


# -------------------------
# Model builder (match your training choices)
# -------------------------
def make_model(arch: str, num_classes: int, pretrained: bool):
    arch = arch.lower()

    if arch == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            m = models.resnet18(weights=weights)
        except Exception:
            m = models.resnet18(pretrained=pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch == "mobilenet_v3_small":
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            m = models.mobilenet_v3_small(weights=weights)
        except Exception:
            m = models.mobilenet_v3_small(pretrained=pretrained)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m

    if arch == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            m = models.efficientnet_b0(weights=weights)
        except Exception:
            m = models.efficientnet_b0(pretrained=pretrained)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError("arch must be resnet18, mobilenet_v3_small, or efficientnet_b0")


# -------------------------
# CSV reading + label mapping (same as your project)
# -------------------------
NUM0_KAGGLE = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
NUM1_OFFICIAL = {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"}


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
        name = NUM0_KAGGLE.get(iv, None)
    elif numeric_scheme == "official1":
        name = NUM1_OFFICIAL.get(iv, None)
    else:
        raise ValueError("numeric_scheme must be kaggle0 or official1")

    return cls_to_idx.get(name, -1)


def auto_detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    path_candidates = [c for c in cols if any(k in lower[c] for k in ["path", "file", "image", "img", "name"])]
    label_candidates = [c for c in cols if any(k in lower[c] for k in ["label", "emotion", "class", "category"])]

    if not path_candidates:
        raise RuntimeError(f"Could not detect image column. Columns={cols}")
    path_col = path_candidates[0]

    if not label_candidates:
        if len(cols) >= 2:
            label_col = cols[1]
        else:
            raise RuntimeError(f"Could not detect label column. Columns={cols}")
    else:
        label_col = label_candidates[0]

    return path_col, label_col


def build_image_index(images_root: str) -> Dict[str, str]:
    idx = {}
    for dirpath, _, files in os.walk(images_root):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                if fn not in idx:
                    idx[fn] = os.path.join(dirpath, fn)
    return idx


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

    # common alternative
    stem, ext = os.path.splitext(base)
    for ext2 in [ext, ".jpg", ".jpeg", ".png"]:
        cand = f"{stem}_aligned{ext2}"
        if cand in img_index:
            return img_index[cand]

    return None


@dataclass
class Row:
    row_idx: int  # index in original df
    path: str
    y: int


class ImgDataset(Dataset):
    def __init__(self, rows: List[Row], tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r.path).convert("RGB")
        x = self.tf(img)
        return x, r.y, r.row_idx


# -------------------------
# Train + predict helpers
# -------------------------
def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = (counts.sum() / (counts + 1e-6))
    w = w / w.mean()
    return w


def make_weighted_sampler(y: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    cls_w = 1.0 / (counts + 1e-6)
    sample_w = cls_w[y]
    return WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.float32),
                                 num_samples=len(sample_w),
                                 replacement=True)


@torch.no_grad()
def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_all, y_all, idx_all = [], [], []

    for x, y, row_idx in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.numpy())
        idx_all.append(row_idx.numpy())

    return np.concatenate(probs_all, 0), np.concatenate(y_all, 0), np.concatenate(idx_all, 0)


def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    class_weights: Optional[np.ndarray] = None,
):
    model = model.to(device)
    model.train()

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=float(label_smoothing),
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

        # quick val accuracy to keep best
        model.eval()
        correct = 0
        total = 0
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = torch.argmax(model(x), dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
        val_acc = correct / max(1, total)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# -------------------------
# Noise flagging logic using OOF probs
# -------------------------
def compute_noise_report(classes: List[str], y_true: np.ndarray, probs: np.ndarray, row_idx: np.ndarray):
    cls_names = np.array(classes)

    pred = probs.argmax(axis=1)
    p_pred = probs[np.arange(len(probs)), pred]
    p_true = probs[np.arange(len(probs)), y_true]

    # top2 margin
    top2 = np.partition(probs, -2, axis=1)[:, -2]
    margin = p_pred - top2

    report = pd.DataFrame({
        "row_idx": row_idx,
        "true_label": cls_names[y_true],
        "pred_label": cls_names[pred],
        "p_true": p_true,
        "p_pred": p_pred,
        "margin": margin,
        "mismatch": (pred != y_true),
    })
    return report


def select_drops(
    report: pd.DataFrame,
    focus: List[str],
    *,
    conf_min: float,
    margin_min: float,
    drop_frac: float,
):
    """
    Drop top-scoring suspicious points per true class among focus set.
    Suspicious = mismatch & p_pred >= conf_min & margin >= margin_min
    Score = (p_pred - p_true) + margin
    """
    focus = set(focus)
    rep = report.copy()

    suspicious = rep["mismatch"] & (rep["p_pred"] >= conf_min) & (rep["margin"] >= margin_min)
    rep["flagged"] = suspicious
    rep["score"] = 0.0
    rep.loc[suspicious, "score"] = (rep.loc[suspicious, "p_pred"] - rep.loc[suspicious, "p_true"]) + rep.loc[suspicious, "margin"]

    drop_rows = []

    for c in sorted(focus):
        sub = rep[(rep["true_label"] == c) & (rep["flagged"])]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("score", ascending=False)

        total_c = int((rep["true_label"] == c).sum())
        k = int(np.ceil(drop_frac * total_c))
        k = max(1, k)

        drop_rows.extend(sub.head(k)["row_idx"].tolist())

    drop_rows = sorted(set(drop_rows))
    dropped = rep[rep["row_idx"].isin(drop_rows)].sort_values("score", ascending=False)
    return drop_rows, dropped, rep


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=r"models_cnn_raf7\raf7_csv_meta.json")
    ap.add_argument("--train_labels", default=None, help="Override train labels CSV path (else uses meta train_labels)")
    ap.add_argument("--images_root", default=None, help="Override images root (else uses meta images_root)")
    ap.add_argument("--numeric_scheme", default=None, choices=["kaggle0", "official1"], help="Override label scheme")

    ap.add_argument("--arch", default=None, choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"],
                    help="Override arch (else uses meta)")
    ap.add_argument("--pretrained", action="store_true", help="Use pretrained weights in fold models")
    ap.add_argument("--img_size", type=int, default=None, help="Override image size (else uses meta)")

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3, help="epochs per fold (start 2–4; CPU-friendly)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)

    # noise detection params
    ap.add_argument("--focus", nargs="+", default=["angry", "disgust", "fear", "sad"])
    ap.add_argument("--drop_frac", type=float, default=0.02)
    ap.add_argument("--conf_min", type=float, default=0.90)
    ap.add_argument("--margin_min", type=float, default=0.20)

    ap.add_argument("--out_clean", default=r"RAF-DB\train_labels_clean_kfold.csv")
    ap.add_argument("--out_dropped", default=r"RAF-DB\train_labels_dropped_noise_kfold.csv")
    ap.add_argument("--out_oof", default=r"RAF-DB\train_labels_oof_probs.csv")
    args = ap.parse_args()

    # load meta from your trained run (so classes/paths match your project)
    meta = json.load(open(args.meta, "r", encoding="utf-8"))
    classes = meta["classes"]

    train_labels_path = args.train_labels or meta["train_labels"]
    images_root = args.images_root or meta["images_root"]
    numeric_scheme = args.numeric_scheme or meta.get("numeric_scheme", "official1")
    arch = args.arch or meta["arch"]
    img_size = int(args.img_size or meta["img_size"])
    pretrained = bool(args.pretrained)  # explicit flag so you can choose

    print("[PATH] train_labels:", train_labels_path)
    print("[PATH] images_root :", images_root)
    print("[CFG] classes:", classes)
    print("[CFG] arch:", arch, "img_size:", img_size, "pretrained:", pretrained)
    print("[CFG] k:", args.k, "epochs/fold:", args.epochs, "batch:", args.batch)
    print("[NOISE] focus:", args.focus, "drop_frac:", args.drop_frac, "conf_min:", args.conf_min, "margin_min:", args.margin_min)

    df = pd.read_csv(train_labels_path) if train_labels_path.lower().endswith(".csv") else pd.read_excel(train_labels_path)
    path_col, label_col = auto_detect_columns(df)

    img_index = build_image_index(images_root)

    rows: List[Row] = []
    bad = 0
    for i, r in df.iterrows():
        y = map_label_to_index(r[label_col], classes, numeric_scheme)
        if y < 0:
            bad += 1
            continue
        p = resolve_image_path(r[path_col], images_root, img_index)
        if not p:
            bad += 1
            continue
        rows.append(Row(row_idx=i, path=p, y=y))

    print(f"[DATA] usable rows={len(rows)} dropped_unreadable={bad}")

    y_all = np.array([r.y for r in rows], dtype=np.int64)
    num_classes = len(classes)

    # transforms (lighter than full training; keep consistent)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device)

    # We will store OOF probs for each row in "rows" order
    oof_probs = np.full((len(rows), num_classes), np.nan, dtype=np.float32)
    oof_idx = np.array([r.row_idx for r in rows], dtype=np.int64)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(rows)), y_all), start=1):
        print(f"\n[FOLD {fold}/{args.k}] train={len(tr_idx)} val={len(va_idx)}")

        train_ds = ImgDataset([rows[i] for i in tr_idx], tf_train)
        val_ds = ImgDataset([rows[i] for i in va_idx], tf_eval)

        y_tr = np.array([rows[i].y for i in tr_idx], dtype=np.int64)

        if args.use_weighted_sampler:
            sampler = make_weighted_sampler(y_tr, num_classes)
            train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                      num_workers=args.num_workers, pin_memory=False)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=False)

        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.num_workers, pin_memory=False)

        # build fold model
        model = make_model(arch, num_classes=num_classes, pretrained=pretrained)
        class_w = compute_class_weights(y_tr, num_classes)

        model = train_fold(
            model, train_loader, val_loader, device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            class_weights=class_w,
        )

        # predict OOF on val
        probs, yv, ridx = predict_probs(model, val_loader, device)

        # store
        # ridx are original df row indices, but our oof_probs index is "rows order"
        # map val sample indices by matching ridx back to oof_idx
        # Use a dict for speed
        pos = {int(oof_idx[i]): i for i in range(len(oof_idx))}
        for j in range(len(ridx)):
            oof_probs[pos[int(ridx[j])], :] = probs[j, :]

        # quick fold accuracy (sanity)
        pred = probs.argmax(axis=1)
        acc = float((pred == yv).mean())
        print(f"[FOLD {fold}] OOF val acc={acc:.3f}")

        # free memory
        del model
        torch.cuda.empty_cache()

    # sanity: all filled?
    if np.isnan(oof_probs).any():
        missing = int(np.isnan(oof_probs).any(axis=1).sum())
        raise RuntimeError(f"OOF probabilities missing for {missing} samples. Something went wrong in mapping.")

    # Build report
    report = compute_noise_report(classes, y_all, oof_probs, oof_idx)

    # Save OOF probs for inspection
    oof_df = report.copy()
    for ci, c in enumerate(classes):
        oof_df[f"p_{c}"] = oof_probs[:, ci]
    os.makedirs(os.path.dirname(args.out_oof), exist_ok=True)
    oof_df.to_csv(args.out_oof, index=False)
    print("[OK] wrote OOF report ->", args.out_oof)

    # Select drops
    drop_rows, dropped_df, full_df = select_drops(
        report,
        args.focus,
        conf_min=args.conf_min,
        margin_min=args.margin_min,
        drop_frac=args.drop_frac,
    )

    os.makedirs(os.path.dirname(args.out_dropped), exist_ok=True)
    dropped_df.to_csv(args.out_dropped, index=False)
    print("[OK] wrote dropped list ->", args.out_dropped)

    clean_df = df.drop(index=drop_rows)
    os.makedirs(os.path.dirname(args.out_clean), exist_ok=True)
    clean_df.to_csv(args.out_clean, index=False)
    print("[OK] wrote cleaned labels ->", args.out_clean)

    print("\n[DROPPED SUMMARY]")
    if len(dropped_df) == 0:
        print("No rows were flagged. Try: lower conf_min (0.85), lower margin_min (0.15), or increase drop_frac (0.03).")
    else:
        print(dropped_df["true_label"].value_counts())
        print("\nTop confusions among dropped:")
        print(dropped_df.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False).head(12))


if __name__ == "__main__":
    main()