import os
import glob
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Class names (7 emotions)
# -------------------------
CLASSES_7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_7)}

# Common numeric encodings in Kaggle mirrors:
# 0: angry, 1: disgust, 2: fear, 3: happy, 4: sad, 5: surprise, 6: neutral
NUM0_KAGGLE = {0:"angry",1:"disgust",2:"fear",3:"happy",4:"sad",5:"surprise",6:"neutral"}

# Another possible encoding (rare in Kaggle mirrors):
# 1: surprise, 2: fear, 3: disgust, 4: happy, 5: sad, 6: angry, 7: neutral (official-style)
NUM1_OFFICIAL = {1:"surprise",2:"fear",3:"disgust",4:"happy",5:"sad",6:"angry",7:"neutral"}


@dataclass
class Sample:
    path: str
    y: int


class CSVDataset(Dataset):
    def __init__(self, samples: List[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, s.y


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


def auto_find_labels(dataset_root: str, stem: str) -> str:
    # finds train_labels.* or test_labels.*
    candidates = glob.glob(os.path.join(dataset_root, f"{stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {stem}.* in {dataset_root}")
    # pick the most common first
    candidates.sort()
    return candidates[0]


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    # fallback
    return pd.read_csv(path)


def auto_detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [c for c in df.columns]
    lower = {c: c.lower() for c in cols}

    # image/path column
    path_candidates = [c for c in cols if any(k in lower[c] for k in ["path", "file", "image", "img", "name"])]
    if not path_candidates:
        raise RuntimeError(f"Could not auto-detect image column. Columns={cols}")
    path_col = path_candidates[0]

    # label column
    label_candidates = [c for c in cols if any(k in lower[c] for k in ["label", "emotion", "class", "category"])]
    if not label_candidates:
        # sometimes label is second column
        if len(cols) >= 2:
            label_col = cols[1]
        else:
            raise RuntimeError(f"Could not auto-detect label column. Columns={cols}")
    else:
        label_col = label_candidates[0]

    return path_col, label_col


def build_image_index(images_root: str) -> Dict[str, str]:
    """
    Build {basename -> fullpath} so CSV can contain just 'xxxx.jpg' while files are nested.
    """
    idx = {}
    for dirpath, _, files in os.walk(images_root):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                if fn not in idx:
                    idx[fn] = os.path.join(dirpath, fn)
    return idx


def normalize_label(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    # common spelling variants
    if s == "fearful":
        s = "fear"
    if s == "surprised":
        s = "surprise"
    if s in CLASS_TO_IDX:
        return s
    return None


def map_labels(series: pd.Series, numeric_scheme: str) -> List[int]:
    """
    numeric_scheme:
      - kaggle0 (0..6 angry..neutral)  [default]
      - official1 (1..7 surprise..neutral)
    """
    numeric_scheme = numeric_scheme.lower()
    ys = []
    for v in series.tolist():
        # try string
        s = normalize_label(v)
        if s is not None:
            ys.append(CLASS_TO_IDX[s])
            continue

        # try numeric
        try:
            iv = int(v)
        except Exception:
            ys.append(-1)
            continue

        if numeric_scheme == "kaggle0":
            name = NUM0_KAGGLE.get(iv, None)
        elif numeric_scheme == "official1":
            name = NUM1_OFFICIAL.get(iv, None)
        else:
            raise ValueError("numeric_scheme must be kaggle0 or official1")

        ys.append(CLASS_TO_IDX[name] if name else -1)

    return ys


def stratified_split(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = np.random.default_rng(seed)
    y = np.array([s.y for s in samples], dtype=np.int64)
    idx = np.arange(len(samples))

    tr_idx, va_idx = [], []
    for c in np.unique(y):
        c_idx = idx[y == c]
        rng.shuffle(c_idx)
        n_val = max(1, int(round(len(c_idx) * val_ratio)))
        va_idx.extend(c_idx[:n_val].tolist())
        tr_idx.extend(c_idx[n_val:].tolist())

    train = [samples[i] for i in tr_idx]
    val = [samples[i] for i in va_idx]
    return train, val


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = (counts.sum() / (counts + 1e-6))
    w = w / w.mean()
    return w


@torch.no_grad()
def eval_multiclass(model, loader, device) -> Dict:
    model.eval()
    y_true, y_pred, probs_all = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)

        y_true.append(y.numpy())
        y_pred.append(pred)
        probs_all.append(probs)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    probs_all = np.concatenate(probs_all, axis=0)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(probs_all.shape[1])))

    return {"acc": acc, "macro_f1": macro_f1, "cm": cm, "y_true": y_true, "y_pred": y_pred, "probs": probs_all}


def thresholds_per_class(y_true: np.ndarray, probs: np.ndarray, precision_target: float, min_pred_pos: int) -> Dict:
    out = {}
    n_classes = probs.shape[1]

    for ci, cname in enumerate(CLASSES_7):
        p = probs[:, ci]
        t = (y_true == ci).astype(np.int64)

        best_f05 = None
        best_prec = None
        max_prec = None

        for theta in np.linspace(0.0, 1.0, 501):
            pred = (p >= theta).astype(np.int64)
            pos = int(pred.sum())
            if pos < min_pred_pos:
                continue

            tp = int(((pred == 1) & (t == 1)).sum())
            fp = int(((pred == 1) & (t == 0)).sum())
            fn = int(((pred == 0) & (t == 1)).sum())

            prec = tp / (tp + fp + 1e-12)
            rec  = tp / (tp + fn + 1e-12)

            f05 = (1.25 * prec * rec) / (0.25 * prec + rec + 1e-12)  # F0.5 favors precision

            if max_prec is None or prec > max_prec:
                max_prec = float(prec)

            if best_f05 is None or f05 > best_f05["f05"]:
                best_f05 = {"theta": float(theta), "prec": float(prec), "rec": float(rec), "f05": float(f05), "pos": pos}

            if prec >= precision_target:
                # among those reaching precision target, keep highest recall
                if best_prec is None or rec > best_prec["rec"]:
                    best_prec = {"theta": float(theta), "prec": float(prec), "rec": float(rec), "f05": float(f05), "pos": pos}

        out[cname] = {
            "max_precision": max_prec,
            "theta_precision": best_prec,   # precision-first (may be None if unreachable)
            "theta_f05": best_f05,          # good default for "avoid false"
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help=r'Folder like C:\Users\carla\PyCharmMiscProject\RAF-DB')
    ap.add_argument("--images_root", default=None, help="Where images are. Default: dataset_root/DATASET")
    ap.add_argument("--train_labels", default=None, help="Default: dataset_root/train_labels.*")
    ap.add_argument("--test_labels", default=None, help="Default: dataset_root/test_labels.*")

    ap.add_argument("--numeric_scheme", default="kaggle0", choices=["kaggle0", "official1"],
                    help="How numeric labels map to emotions. If your labels are strings, this doesn't matter.")

    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--img_size", type=int, default=112)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    ap.add_argument("--precision_target", type=float, default=0.80)
    ap.add_argument("--min_pred_pos", type=int, default=50)

    ap.add_argument("--save_dir", default="models_cnn_raf7")
    ap.add_argument("--run_name", default="raf7_csv")

    args = ap.parse_args()
    set_seed(args.seed)

    dataset_root = args.dataset_root
    images_root = args.images_root or os.path.join(dataset_root, "DATASET")

    train_labels = args.train_labels or auto_find_labels(dataset_root, "train_labels")
    test_labels  = args.test_labels  or auto_find_labels(dataset_root, "test_labels")

    print("[PATH] dataset_root:", dataset_root)
    print("[PATH] images_root :", images_root)
    print("[PATH] train_labels:", train_labels)
    print("[PATH] test_labels :", test_labels)

    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"images_root not found: {images_root}")

    # build index for robust filename resolving
    img_index = build_image_index(images_root)
    if len(img_index) < 1000:
        print("[WARN] Very small image index. Is images_root correct?")

    def load_samples(label_path: str) -> List[Sample]:
        df = read_table(label_path)
        path_col, label_col = auto_detect_columns(df)
        print(f"[CSV] {os.path.basename(label_path)} path_col={path_col} label_col={label_col} rows={len(df)}")

        ys = map_labels(df[label_col], numeric_scheme=args.numeric_scheme)
        samples = []
        bad = 0

        for i, row in df.iterrows():
            y = ys[i]
            if y < 0:
                bad += 1
                continue

            raw = str(row[path_col]).strip()
            # if raw is already an existing path
            if os.path.isfile(raw):
                p = raw
            else:
                # try join with images_root
                p = os.path.join(images_root, raw)
                if not os.path.isfile(p):
                    # try by basename lookup
                    base = os.path.basename(raw)
                    p = img_index.get(base, None)
            if not p or not os.path.isfile(p):
                bad += 1
                continue

            samples.append(Sample(path=p, y=y))

        print(f"[CSV] kept={len(samples)}  dropped={bad}")
        return samples

    train_all = load_samples(train_labels)
    test_samples = load_samples(test_labels)

    if not train_all:
        raise RuntimeError("No training samples were loaded. Check images_root and label mapping (numeric_scheme).")

    train_samples, val_samples = stratified_split(train_all, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[DATA] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    # Transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=12, translate=(0.04, 0.04), scale=(0.90, 1.10)),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
        transforms.RandomGrayscale(p=0.10),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = CSVDataset(train_samples, transform=train_tf)
    val_ds   = CSVDataset(val_samples, transform=eval_tf)
    test_ds  = CSVDataset(test_samples, transform=eval_tf) if test_samples else None

    # Loaders
    ytr = np.array([s.y for s in train_samples], dtype=np.int64)
    num_classes = len(CLASSES_7)

    if args.use_weighted_sampler:
        counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
        cls_w = 1.0 / (counts + 1e-6)
        sample_w = cls_w[ytr]
        sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.float32),
                                        num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    # Model + loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(args.arch, num_classes=num_classes, pretrained=args.pretrained).to(device)

    class_w = compute_class_weights(ytr, num_classes)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_w, dtype=torch.float32, device=device),
        label_smoothing=float(args.label_smoothing),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Train
    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")
    meta_path = os.path.join(args.save_dir, f"{args.run_name}_meta.json")
    thr_path  = os.path.join(args.save_dir, f"{args.run_name}_thresholds.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_res = eval_multiclass(model, val_loader, device)
        scheduler.step(val_res["macro_f1"])

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[EPOCH {epoch:03d}/{args.epochs}] lr={lr_now:.2e}")
        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.3f}")
        print(f"  val_acc={val_res['acc']:.3f} val_macro_f1={val_res['macro_f1']:.3f}")

        if val_res["macro_f1"] > best_f1 + 1e-6:
            best_f1 = float(val_res["macro_f1"])
            ckpt = {
                "arch": args.arch,
                "pretrained": bool(args.pretrained),
                "img_size": int(args.img_size),
                "classes": CLASSES_7,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_f1,
            }
            torch.save(ckpt, best_path)
            print(f"  [SAVE] best -> {best_path}")

    # Final: thresholds from VAL of best model
    print("\n[FINAL] Loading best checkpoint...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_res = eval_multiclass(model, val_loader, device)
    print(f"[VAL] acc={val_res['acc']:.3f} macro_f1={val_res['macro_f1']:.3f}")
    print("[VAL] confusion:\n", val_res["cm"])
    print("[VAL] report:\n", classification_report(val_res["y_true"], val_res["y_pred"], target_names=CLASSES_7, digits=3, zero_division=0))

    th = thresholds_per_class(val_res["y_true"], val_res["probs"], precision_target=args.precision_target, min_pred_pos=args.min_pred_pos)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "arch": ckpt["arch"],
            "pretrained": ckpt["pretrained"],
            "img_size": ckpt["img_size"],
            "classes": ckpt["classes"],
            "best_checkpoint": os.path.abspath(best_path),
            "val_macro_f1": ckpt["val_macro_f1"],
            "dataset_root": os.path.abspath(dataset_root),
            "images_root": os.path.abspath(images_root),
            "train_labels": os.path.abspath(train_labels),
            "test_labels": os.path.abspath(test_labels),
            "numeric_scheme": args.numeric_scheme,
        }, f, indent=2)

    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump({
            "precision_target": float(args.precision_target),
            "min_pred_pos": int(args.min_pred_pos),
            "thresholds": th,
        }, f, indent=2)

    print("[OK] wrote", meta_path)
    print("[OK] wrote", thr_path)

    if test_loader is not None:
        test_res = eval_multiclass(model, test_loader, device)
        print(f"[TEST] acc={test_res['acc']:.3f} macro_f1={test_res['macro_f1']:.3f}")
        print("[TEST] report:\n", classification_report(test_res["y_true"], test_res["y_pred"], target_names=CLASSES_7, digits=3, zero_division=0))


if __name__ == "__main__":
    main()