import os
import glob
import argparse
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_label(name: str) -> str:
    s = (name or "").strip().lower()
    if s == "fearful":
        return "fear"
    return s


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(dir_path: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(dir_path, f"**/*{ext}"), recursive=True))
    return files


@dataclass
class Sample:
    path: str
    y: int


class FERFolderDataset(Dataset):
    """
    Expects:
      dataset_root/
        train/<class>/*.jpg
        validation/<class>/*.jpg   (optional)
        test/<class>/*.jpg         (optional)

    Only keeps classes listed in `classes` (after normalization).
    """
    def __init__(self, samples: List[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # PIL is safest for torchvision transforms
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.y


def scan_split(
    dataset_root: str,
    split: str,
    classes: List[str],
) -> List[Sample]:
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        return []

    classes_norm = [norm_label(c) for c in classes]
    class_to_idx = {c: i for i, c in enumerate(classes_norm)}

    samples: List[Sample] = []
    for folder in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        label = norm_label(folder)
        if label not in class_to_idx:
            continue

        files = list_images(folder_path)
        for p in files:
            samples.append(Sample(path=p, y=class_to_idx[label]))

    return samples


def stratified_train_val_split(
    train_samples: List[Sample],
    val_ratio: float = 0.15,
    seed: int = 13,
) -> Tuple[List[Sample], List[Sample]]:
    y = np.array([s.y for s in train_samples], dtype=np.int64)
    idx = np.arange(len(train_samples))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(sss.split(idx, y))

    tr = [train_samples[i] for i in tr_idx]
    va = [train_samples[i] for i in va_idx]
    return tr, va


def class_counts(samples: List[Sample], num_classes: int) -> np.ndarray:
    y = np.array([s.y for s in samples], dtype=np.int64)
    counts = np.bincount(y, minlength=num_classes)
    return counts


def make_model(arch: str, num_classes: int, pretrained: bool):
    arch = arch.lower()

    if arch == "resnet18":
        # torchvision API changed across versions; handle both
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "mobilenet_v3_small":
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
        except Exception:
            model = models.mobilenet_v3_small(pretrained=pretrained)
        # classifier is Sequential; last is Linear
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model

    if arch == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=pretrained)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model

    raise ValueError(f"Unknown arch: {arch}. Use resnet18, mobilenet_v3_small, efficientnet_b0")


@torch.no_grad()
def evaluate(model, loader, device, class_names: List[str]):
    model.eval()
    ys, preds = [], []
    total_loss = 0.0
    n = 0

    # If you want to compute loss in eval too, you can pass criterion; here we just do metrics.
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        p = torch.argmax(logits, dim=1)

        ys.append(y.cpu().numpy())
        preds.append(p.cpu().numpy())
        n += y.numel()

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return acc, macro_f1, report, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help=r'Folder like C:\Users\carla\PyCharmMiscProject\fer-2013 (must contain train/...)')

    ap.add_argument("--classes", nargs="*", default=["happy", "sad", "neutral", "fear"],
                    help="Which emotion folders to use (case-insensitive). Example: happy sad neutral fear")

    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])

    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    ap.add_argument("--img_size", type=int, default=96, help="Input size (FER is 48x48; 96 is a good start)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--val_ratio", type=float, default=0.15,
                    help="Used only if dataset_root/validation does not exist")

    ap.add_argument("--use_weighted_sampler", action="store_true",
                    help="Use WeightedRandomSampler to balance classes in training batches")

    ap.add_argument("--save_dir", default="models_cnn")
    ap.add_argument("--run_name", default="fer4_cnn")

    ap.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")

    args = ap.parse_args()
    set_seed(args.seed)

    classes_norm = [norm_label(c) for c in args.classes]
    num_classes = len(classes_norm)

    # -----------------------------
    # Load samples
    # -----------------------------
    train_samples = scan_split(args.dataset_root, "train", classes_norm)
    val_samples = scan_split(args.dataset_root, "validation", classes_norm)
    test_samples = scan_split(args.dataset_root, "test", classes_norm)

    if not train_samples:
        raise RuntimeError("No training samples found. Check your folder structure and --classes.")

    if not val_samples:
        # create val from train
        train_samples, val_samples = stratified_train_val_split(
            train_samples, val_ratio=args.val_ratio, seed=args.seed
        )
        print(f"[DATA] No validation/ folder found. Created stratified val split: val_ratio={args.val_ratio:.2f}")

    print(f"[DATA] train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    tr_counts = class_counts(train_samples, num_classes)
    va_counts = class_counts(val_samples, num_classes)
    print("[DATA] train class counts:", {classes_norm[i]: int(tr_counts[i]) for i in range(num_classes)})
    print("[DATA] val   class counts:", {classes_norm[i]: int(va_counts[i]) for i in range(num_classes)})

    # -----------------------------
    # Transforms
    # -----------------------------
    # ImageNet normalization (good when using pretrained CNNs)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = FERFolderDataset(train_samples, transform=train_tf)
    val_ds   = FERFolderDataset(val_samples, transform=eval_tf)
    test_ds  = FERFolderDataset(test_samples, transform=eval_tf) if test_samples else None

    # -----------------------------
    # DataLoaders
    # -----------------------------
    if args.use_weighted_sampler:
        ytr = np.array([s.y for s in train_samples], dtype=np.int64)
        counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
        cls_w = 1.0 / (counts + 1e-6)
        sample_w = cls_w[ytr]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.float32),
            num_samples=len(sample_w),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    # -----------------------------
    # Model
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(args.arch, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    # Class weights for loss (helps imbalance)
    ytr = np.array([s.y for s in train_samples], dtype=np.int64)
    counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    use_amp = bool(args.amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -----------------------------
    # Train loop
    # -----------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")
    last_path = os.path.join(args.save_dir, f"{args.run_name}_last.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            seen += int(y.size(0))

        train_loss = total_loss / max(1, seen)
        train_acc = correct / max(1, seen)

        val_acc, val_f1, val_report, val_cm = evaluate(model, val_loader, device, classes_norm)
        scheduler.step(val_f1)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[EPOCH {epoch:03d}/{args.epochs}] lr={lr_now:.2e}")
        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}")
        print(f"  val_acc={val_acc:.3f}  val_macro_f1={val_f1:.3f}")
        print("  val_confusion_matrix:\n", val_cm)

        # Save best by macro-F1
        ckpt = {
            "arch": args.arch,
            "pretrained": args.pretrained,
            "img_size": args.img_size,
            "classes": classes_norm,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_macro_f1": val_f1,
        }
        torch.save(ckpt, last_path)

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            torch.save(ckpt, best_path)
            print(f"  [SAVE] new best macro-F1={best_f1:.3f} -> {best_path}")

    # Final evaluation with best model
    print("\n[FINAL] Loading best checkpoint for evaluation...")
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["state_dict"])

    val_acc, val_f1, val_report, val_cm = evaluate(model, val_loader, device, classes_norm)
    print(f"[VAL] acc={val_acc:.3f}  macro_f1={val_f1:.3f}")
    print("[VAL] report:\n", val_report)

    if test_loader is not None:
        test_acc, test_f1, test_report, test_cm = evaluate(model, test_loader, device, classes_norm)
        print(f"[TEST] acc={test_acc:.3f}  macro_f1={test_f1:.3f}")
        print("[TEST] report:\n", test_report)

    # Save metadata for inference scripts
    meta_path = os.path.join(args.save_dir, f"{args.run_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "arch": args.arch,
            "pretrained": args.pretrained,
            "img_size": args.img_size,
            "classes": classes_norm,
            "best_macro_f1": best_f1,
            "best_checkpoint": os.path.abspath(best_path),
        }, f, indent=2)
    print("[OK] wrote", meta_path)


if __name__ == "__main__":
    main()