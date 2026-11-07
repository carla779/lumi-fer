import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


@dataclass
class Sample:
    path: str
    y: int


class ImgDataset(Dataset):
    def __init__(self, samples: List[Sample], tf):
        self.samples = samples
        self.tf = tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.tf(img)
        return x, s.y


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
    raise ValueError("Unsupported arch")


def build_image_index(images_root: str) -> Dict[str, str]:
    idx = {}
    for dirpath, _, files in os.walk(images_root):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                if fn not in idx:
                    idx[fn] = os.path.join(dirpath, fn)
    return idx


# mappings (same as your training script)
CLASSES_7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_7)}
NUM0_KAGGLE = {0:"angry",1:"disgust",2:"fear",3:"happy",4:"sad",5:"surprise",6:"neutral"}
NUM1_OFFICIAL = {1:"surprise",2:"fear",3:"disgust",4:"happy",5:"sad",6:"angry",7:"neutral"}


def normalize_label(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s == "fearful":
        s = "fear"
    if s == "surprised":
        s = "surprise"
    if s in CLASS_TO_IDX:
        return s
    return None


def map_label(v, scheme: str) -> int:
    s = normalize_label(v)
    if s is not None:
        return CLASS_TO_IDX[s]
    try:
        iv = int(v)
    except Exception:
        return -1
    if scheme == "kaggle0":
        name = NUM0_KAGGLE.get(iv)
    else:
        name = NUM1_OFFICIAL.get(iv)
    return CLASS_TO_IDX[name] if name else -1


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


@torch.no_grad()
def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    ps = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(probs)
    return np.concatenate(ys, 0), np.concatenate(ps, 0)


def calibrate_top1_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: List[str],
    precision_target: float,
    min_pred_pos: int,
    margin: float
) -> Dict:
    n = len(y_true)
    order = np.argsort(-probs, axis=1)
    top_idx = order[:, 0]
    second_idx = order[:, 1]
    p_top = probs[np.arange(n), top_idx]
    p_second = probs[np.arange(n), second_idx]
    p_margin = p_top - p_second

    out = {}
    for ci, cname in enumerate(classes):
        mask = (top_idx == ci)
        if mask.sum() == 0:
            out[cname] = {"theta": None, "prec": None, "rec": None, "pos": 0, "note": "never top1"}
            continue

        p1 = p_top[mask]
        m1 = p_margin[mask]
        correct = (y_true[mask] == ci)

        total_true = int((y_true == ci).sum())

        best = None
        best_any = None  # best precision overall (in case target not reachable)

        for theta in np.linspace(0.0, 1.0, 501):
            accept = (p1 >= theta) & (m1 >= margin)
            pos = int(accept.sum())
            if pos < min_pred_pos:
                continue

            tp = int((accept & correct).sum())
            fp = int((accept & (~correct)).sum())
            fn = total_true - tp

            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)

            # keep best-any by precision, tie-break by recall
            if (best_any is None) or (prec > best_any["prec"] + 1e-12) or (abs(prec - best_any["prec"]) < 1e-12 and rec > best_any["rec"]):
                best_any = {"theta": float(theta), "prec": float(prec), "rec": float(rec), "pos": pos}

            # keep best that meets target (maximize recall under constraint)
            if prec >= precision_target:
                if (best is None) or (rec > best["rec"]):
                    best = {"theta": float(theta), "prec": float(prec), "rec": float(rec), "pos": pos}

        out[cname] = {
            "theta": (best["theta"] if best else (best_any["theta"] if best_any else None)),
            "prec":  (best["prec"] if best else (best_any["prec"] if best_any else None)),
            "rec":   (best["rec"] if best else (best_any["rec"] if best_any else None)),
            "pos":   (best["pos"] if best else (best_any["pos"] if best_any else 0)),
            "met_target": bool(best is not None),
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=r"models_cnn_raf7\raf7_csv_meta.json")
    ap.add_argument("--ckpt", default=r"models_cnn_raf7\raf7_csv_best.pt")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--precision_target", type=float, default=0.85)
    ap.add_argument("--min_pred_pos", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.12)

    ap.add_argument("--out", default=r"models_cnn_raf7\raf7_csv_thresholds_top1.json")
    args = ap.parse_args()

    meta = json.load(open(args.meta, "r", encoding="utf-8"))
    arch = meta["arch"]
    img_size = int(meta["img_size"])
    scheme = meta.get("numeric_scheme", "official1")

    dataset_root = meta["dataset_root"]
    images_root = meta["images_root"]
    train_labels = meta["train_labels"]

    # build samples from train_labels
    df = pd.read_csv(train_labels)
    path_col = "image"
    label_col = "label"

    img_index = build_image_index(images_root)

    samples = []
    for _, row in df.iterrows():
        y = map_label(row[label_col], scheme)
        if y < 0:
            continue
        raw = str(row[path_col]).strip()
        p = raw if os.path.isfile(raw) else os.path.join(images_root, raw)
        if not os.path.isfile(p):
            base = os.path.basename(raw)
            p = img_index.get(base)
        if not p or not os.path.isfile(p):
            continue
        samples.append(Sample(path=p, y=y))

    _, val_samples = stratified_split(samples, val_ratio=args.val_ratio, seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(arch, num_classes=len(CLASSES_7)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    loader = DataLoader(ImgDataset(val_samples, tf), batch_size=64, shuffle=False, num_workers=0)
    y_true, probs = predict_probs(model, loader, device)

    th = calibrate_top1_thresholds(
        y_true=y_true,
        probs=probs,
        classes=CLASSES_7,
        precision_target=args.precision_target,
        min_pred_pos=args.min_pred_pos,
        margin=args.margin,
    )

    out = {
        "precision_target": args.precision_target,
        "min_pred_pos": args.min_pred_pos,
        "margin": args.margin,
        "thresholds_top1": th,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("[OK] wrote", args.out)
    for c, v in th.items():
        print(f"{c:9s} theta={v['theta']}  prec={v['prec']} rec={v['rec']} pos={v['pos']} met_target={v['met_target']}")


if __name__ == "__main__":
    main()