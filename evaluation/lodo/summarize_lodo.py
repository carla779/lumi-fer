"""
summarize_lodo.py — Re-evaluate the four LODO checkpoints on their respective
held-out test splits and produce a clean markdown table.

Run after run_lodo.bat finishes.
Outputs:
    lodo_results.csv   (machine-readable)
    lodo_results.md    (markdown drop-in for the report)
"""
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

RUNS = [
    ("Holdout: RAF-DB",   "LODO_rafdb",   "lodo_HOLDOUT_rafdb.csv"),
    ("Holdout: FERPlus",  "LODO_ferplus", "lodo_HOLDOUT_ferplus.csv"),
    ("Holdout: ExpW",     "LODO_expw",    "lodo_HOLDOUT_expw.csv"),
    ("Holdout: FER2013",  "LODO_fer2013", "lodo_HOLDOUT_fer2013.csv"),
]
MODEL_DIR = "models_lodo"


def make_backbone(arch):
    arch = arch.lower()
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim
    raise ValueError(arch)


class FERModel(nn.Module):
    def __init__(self, arch, num_classes, au_dim=9, use_au=True, dropout=0.2):
        super().__init__()
        self.backbone, feat_dim = make_backbone(arch)
        self.dropout  = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.use_au = use_au
        if use_au:
            self.au_head = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                nn.Dropout(p=0.1), nn.Linear(256, au_dim),
            )
    def forward(self, x):
        feats = self.backbone(x)
        return self.cls_head(self.dropout(feats))


class TestDS(Dataset):
    def __init__(self, df, img_size, classes):
        self.df = df.reset_index(drop=True)
        self.lab2id = {c: i for i, c in enumerate(classes)}
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        y = self.lab2id[str(r["label"]).lower()]
        im = Image.open(str(r["image"])).convert("RGB")
        return self.tf(im), y


def evaluate(display, run_name, csv, device):
    meta_path = os.path.join(MODEL_DIR, f"{run_name}_meta.json")
    ckpt_path = os.path.join(MODEL_DIR, f"{run_name}_best.pt")
    if not (os.path.isfile(meta_path) and os.path.isfile(ckpt_path)):
        print(f"[SKIP] {run_name}: missing checkpoint or meta")
        return None
    meta = json.load(open(meta_path))
    classes  = meta["classes"]
    img_size = int(meta["img_size"])
    model = FERModel(meta["arch"], num_classes=len(classes),
                     use_au=bool(meta.get("use_au", False)),
                     dropout=float(meta.get("dropout", 0.2))).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    df = pd.read_csv(csv, low_memory=False)
    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    df["label"] = df["label"].astype(str).str.lower()
    df = df[df["label"].isin(classes)].reset_index(drop=True)
    df = df[df["image"].apply(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)
    dl = DataLoader(TestDS(df, img_size, classes), batch_size=32, shuffle=False,
                    num_workers=2, pin_memory=torch.cuda.is_available())

    ys, yp = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(device))
            ys.extend(y.cpu().numpy().tolist())
            yp.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    macro_f1 = f1_score(ys, yp, average="macro", zero_division=0)
    acc      = accuracy_score(ys, yp)
    pc_f1    = f1_score(ys, yp, average=None, zero_division=0).tolist()
    return {"display": display, "run": run_name, "n_test": len(ys),
            "macro_f1": macro_f1, "accuracy": acc,
            "per_class_f1": dict(zip(classes, pc_f1)),
            "classes": classes}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    results = []
    for d, rn, csv in RUNS:
        r = evaluate(d, rn, csv, device)
        if r is None: continue
        results.append(r)
        print(f"  {d:25s}  N={r['n_test']:6d}  macro-F1={r['macro_f1']:.3f}  acc={r['accuracy']:.3f}")

    if not results:
        print("[ERROR] no LODO runs evaluated"); return

    # Markdown
    classes = results[0]["classes"]
    lines = []
    lines.append("# LODO results — drop into Section 4.5.2 of the report\n")
    lines.append("Each row trains the 5-class v9 pipeline on three of the four FER datasets "
                 "and tests on the held-out fourth, isolating cross-dataset generalisation.\n")
    header = ["Holdout dataset", "N (test)", "Macro-F1", "Accuracy"] + [f"F1 ({c})" for c in classes]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in results:
        row = [r["display"], str(r["n_test"]), f"{r['macro_f1']:.3f}", f"{r['accuracy']:.3f}"]
        for c in classes: row.append(f"{r['per_class_f1'].get(c, 0):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    # Mean row
    mean_f1 = np.mean([r["macro_f1"] for r in results])
    mean_acc = np.mean([r["accuracy"] for r in results])
    lines.append(f"| **Mean (across LODO splits)** | — | **{mean_f1:.3f}** | **{mean_acc:.3f}** | " +
                 " | ".join(["—"] * len(classes)) + " |")
    out = "\n".join(lines)
    open("lodo_results.md", "w").write(out)
    pd.DataFrame([{"run": r["run"], "n": r["n_test"], "macro_f1": r["macro_f1"],
                   "accuracy": r["accuracy"]} for r in results]).to_csv("lodo_results.csv", index=False)
    print("\n[OK] saved lodo_results.csv and lodo_results.md")
    print()
    print(out)


if __name__ == "__main__":
    main()
