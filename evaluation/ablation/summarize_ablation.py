"""
summarize_ablation.py — Re-evaluate the v9 baseline + every ABLATE_* run
on the same held-out test split, then write a clean markdown table.

Run this AFTER run_ablation.bat finishes.

Outputs:
    ablation_results.csv         (machine-readable)
    ablation_results.md          (drop-in markdown table for the report)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, classification_report, accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----- config -----
RUNS = [
    # (display name, run id, model dir, csv path used)
    ("Full pipeline (v9 baseline)",   "fer7_v9_5class_mixup", "models_cnn_multi",
     "unified_fer7_processed_au_FRIEND.csv"),
    ("Without AU auxiliary loss",     "ABLATE_NO_AU",         "models_ablation",
     "unified_fer7_processed_au_FRIEND.csv"),
    ("Without MixUp augmentation",    "ABLATE_NO_MIXUP",      "models_ablation",
     "unified_fer7_processed_au_FRIEND.csv"),
    ("Without balanced sampler",      "ABLATE_NO_BALANCE",    "models_ablation",
     "unified_fer7_processed_au_FRIEND.csv"),
    ("Hard one-hot labels (no soft)", "ABLATE_HARD_LABEL",    "models_ablation",
     "unified_fer7_processed_au_HARDLABELS.csv"),
]

# 5-class set used by v9; if any older 7-class run is included, the script
# will detect the class list from each run's meta and report what it can.
P_COLS = ["p_angry", "p_disgust", "p_fear", "p_happy", "p_sad", "p_surprise", "p_neutral"]


def make_backbone(arch):
    arch = arch.lower()
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim
    if arch == "swin_tiny":
        import timm
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        return m, m.num_features
    raise ValueError(arch)


class FERModel(nn.Module):
    def __init__(self, arch, num_classes, au_dim=9, use_au=True, dropout=0.2):
        super().__init__()
        self.backbone, feat_dim = make_backbone(arch)
        self.dropout  = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.use_au   = use_au
        if use_au:
            self.au_head = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                nn.Dropout(p=0.1), nn.Linear(256, au_dim),
            )
    def forward(self, x):
        feats = self.backbone(x)
        return self.cls_head(self.dropout(feats))


class TestDataset(Dataset):
    def __init__(self, df, img_size, classes):
        self.df = df.reset_index(drop=True)
        self.lab2id = {c: i for i, c in enumerate(classes)}
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        y = self.lab2id[str(r["label"]).lower()]
        im = Image.open(str(r["image"])).convert("RGB")
        return self.tf(im), y


def evaluate_run(display, run_name, model_dir, csv_path, device):
    meta_path = os.path.join(model_dir, f"{run_name}_meta.json")
    ckpt_path = os.path.join(model_dir, f"{run_name}_best.pt")
    if not (os.path.isfile(meta_path) and os.path.isfile(ckpt_path)):
        print(f"[SKIP] {run_name}: missing meta or checkpoint")
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    classes  = meta["classes"]
    arch     = meta["arch"]
    img_size = int(meta["img_size"])
    use_au   = bool(meta.get("use_au", False))

    # Build model and load checkpoint
    model = FERModel(arch, num_classes=len(classes), use_au=use_au,
                     dropout=float(meta.get("dropout", 0.2))).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load test split
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    df["label"] = df["label"].astype(str).str.lower()
    df = df[df["label"].isin(classes)].reset_index(drop=True)
    df = df[df["image"].apply(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)
    ds = TestDataset(df, img_size, classes)
    dl = DataLoader(ds, batch_size=32, shuffle=False,
                    num_workers=2, pin_memory=torch.cuda.is_available())

    ys, yp = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            ys.extend(y.cpu().numpy().tolist())
            yp.extend(pred.cpu().numpy().tolist())

    macro_f1 = float(f1_score(ys, yp, average="macro", zero_division=0))
    acc      = float(accuracy_score(ys, yp))
    per_class_f1 = f1_score(ys, yp, average=None, zero_division=0).tolist()

    return {
        "display": display,
        "run":     run_name,
        "n_test":  len(ys),
        "macro_f1": macro_f1,
        "accuracy": acc,
        "per_class_f1": dict(zip(classes, per_class_f1)),
        "classes": classes,
        "config":  {
            "arch":         arch,
            "img_size":     img_size,
            "lambda_au":    meta.get("lambda_au"),
            "mixup_alpha":  meta.get("mixup_alpha"),
            "smoothing":    meta.get("smoothing"),
            "dropout":      meta.get("dropout"),
            "freeze_epochs": meta.get("freeze_epochs"),
        },
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    results = []
    for display, run_name, model_dir, csv_path in RUNS:
        r = evaluate_run(display, run_name, model_dir, csv_path, device)
        if r is None:
            continue
        results.append(r)
        print(f"  {display:38s}  macro-F1={r['macro_f1']:.3f}  acc={r['accuracy']:.3f}  N={r['n_test']}")

    if not results:
        print("[ERROR] no runs evaluated.")
        return

    # CSV
    rows = []
    for r in results:
        row = {
            "display":     r["display"],
            "run":         r["run"],
            "n_test":      r["n_test"],
            "macro_f1":    f"{r['macro_f1']:.4f}",
            "accuracy":    f"{r['accuracy']:.4f}",
        }
        for cls, v in r["per_class_f1"].items():
            row[f"f1_{cls}"] = f"{v:.4f}"
        for k, v in r["config"].items():
            row[k] = v
        rows.append(row)
    out_csv = "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[OK] saved {out_csv}")

    # Markdown table — use baseline (first row) to compute deltas
    base = results[0]
    classes = base["classes"]

    lines = []
    lines.append("# Ablation results — drop into Section 4.X / 6.X of the report\n")
    lines.append(f"Test set: held-out test split of the unified multi-dataset CSV "
                 f"(N = {base['n_test']} examples). Baseline = full v9 pipeline "
                 f"(EfficientNet-B0, soft labels, AU auxiliary loss \u03bb=0.05, "
                 f"MixUp \u03b1=0.2, dataset-class balanced sampler).\n")
    header = ["Variant", "Macro-F1", "\u0394 vs baseline", "Accuracy"] + [f"F1 ({c})" for c in classes]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in results:
        delta = r["macro_f1"] - base["macro_f1"]
        delta_str = "—" if r is base else (f"{delta:+.3f}")
        row = [r["display"], f"{r['macro_f1']:.3f}", delta_str, f"{r['accuracy']:.3f}"]
        for c in classes:
            row.append(f"{r['per_class_f1'].get(c, 0):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    out_md = "ablation_results.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] saved {out_md}")
    print("\n--- Markdown preview ---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
