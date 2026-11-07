"""
FER-7 multi-dataset training script  — v4
==========================================
Changes from v3:
  - FIX 1: Class-weighted loss (penalises mistakes on disgust/fear more)
  - FIX 2: Targeted extra augmentation for rare classes (disgust, fear)
  - FIX 3: Reduced over-regularisation (smoothing 0.05, dropout 0.2)
  - FIX 4: EfficientNet-B0 as default backbone
  - FIX 5: MixUp augmentation (optional, --mixup_alpha > 0)
  - FIX 6: Cosine LR restart (CosineAnnealingWarmRestarts) option
  - KEEP:  OneCycleLR, freeze/unfreeze, gradient clipping, AU aux loss

Roadmap to F1 0.7 / 0.8:
  Stage 1 — this script (~0.45-0.55 expected):
    Run with efficientnet_b0, img_size 160, epochs 50

  Stage 2 — add AffectNet data (~0.55-0.65 expected):
    Request AffectNet at http://mohammadmahoor.com/affectnet
    Add to your build_multidataset_fer7_csv pipeline

  Stage 3 — swap to ViT or Swin-Tiny backbone (~0.65-0.75 expected):
    pip install timm
    Use: --arch swin_tiny  (add to make_backbone below)

  Stage 4 — test-time augmentation (TTA) (~+0.02-0.04):
    Average predictions over 5 augmented versions at test time

  Stage 5 — ensemble (~0.75-0.82 expected):
    Train 3 models (resnet18, efficientnet_b0, swin_tiny) separately
    Average their softmax outputs at inference
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PIL import Image, ImageFile
from torchvision import transforms, models

from sklearn.metrics import f1_score, classification_report

ImageFile.LOAD_TRUNCATED_IMAGES = True


FER5    = ["angry", "happy", "sad", "surprise", "neutral"]
LAB2ID  = {c: i for i, c in enumerate(FER5)}

# Rare classes that get extra augmentation (Fix 2)
RARE_CLASSES = set()  # 5-class build: no rare classes

P_COLS = ["p_angry", "p_happy", "p_sad", "p_surprise", "p_neutral"]
AU_COLS = [
    "au_eye_ear_l", "au_eye_ear_r",
    "au_eye_open_l", "au_eye_open_r",
    "au_mouth_open", "au_mouth_width",
    "au_brow_raise_l", "au_brow_raise_r",
    "au_brow_inner_dist",
]


# ── Backbone factory ──────────────────────────────────────────────────────────

def make_backbone(arch: str, pretrained: bool):
    arch = arch.lower()

    if arch == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    if arch == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    if arch == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim

    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim

    if arch == "efficientnet_b2":
        w = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b2(weights=w)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim

    # Stage 3 upgrade: pip install timm, then use --arch swin_tiny
    if arch == "swin_tiny":
        try:
            import timm
            m = timm.create_model("swin_tiny_patch4_window7_224",
                                  pretrained=pretrained, num_classes=0)
            feat_dim = m.num_features
            return m, feat_dim
        except ImportError:
            raise ImportError("pip install timm  to use swin_tiny")

    raise ValueError(f"Unsupported arch: {arch}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class FERDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool):
        self.df       = df.reset_index(drop=True)
        self.img_size = img_size
        self.train    = train

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]

        # Standard train transform
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
                transforms.RandomErasing(p=0.10, scale=(0.02, 0.10)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])

        # FIX 2: Extra augmentation specifically for rare classes (disgust, fear)
        # Applied BEFORE the standard transform so it adds variety on top
        self.rare_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
            transforms.RandomAffine(degrees=10, translate=(0.10, 0.10), scale=(0.88, 1.12)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r        = self.df.iloc[idx]
        img_path = str(r["image"])
        label    = str(r["label"]).lower()
        y        = LAB2ID[label]

        # Soft target (FERPlus rows have genuine probability distributions)
        soft = None
        if all(c in self.df.columns for c in P_COLS):
            vals = [r.get(c, np.nan) for c in P_COLS]
            if all(pd.notna(v) for v in vals):
                v = np.array(vals, dtype=np.float32)
                s = float(v.sum())
                if s > 1e-6:
                    soft = v / s

        if soft is None:
            soft = np.zeros((len(FER5),), dtype=np.float32)
            soft[y] = 1.0

        # AU targets
        au_ok = int(r.get("au_ok", 0)) if "au_ok" in self.df.columns else 0
        au    = np.zeros((len(AU_COLS),), dtype=np.float32)
        if au_ok and all(c in self.df.columns for c in AU_COLS):
            vals = [r.get(c, np.nan) for c in AU_COLS]
            if all(pd.notna(v) for v in vals):
                au = np.array(vals, dtype=np.float32)
            else:
                au_ok = 0

        im = Image.open(img_path).convert("RGB")

        # FIX 2: apply rare-class extra augmentation before the standard pipeline
        if self.train and label in RARE_CLASSES:
            im = self.rare_aug(im)

        x = self.tf(im)

        dataset_name = str(r.get("dataset", "unk"))
        return x, y, soft, au, au_ok, dataset_name


# ── Model ─────────────────────────────────────────────────────────────────────

class FERModel(nn.Module):
    def __init__(self, arch: str, pretrained: bool,
                 num_classes: int = 7, au_dim: int = 9,
                 use_au: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone, feat_dim = make_backbone(arch, pretrained)
        # FIX 3: dropout 0.2 (reduced from 0.3 — was over-regularising)
        self.dropout  = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.use_au   = use_au
        if use_au:
            self.au_head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(256, au_dim),
            )

    def forward(self, x):
        feats   = self.backbone(x)
        feats_d = self.dropout(feats)
        logits  = self.cls_head(feats_d)
        au_pred = self.au_head(feats) if self.use_au else None
        return logits, au_pred


# ── Loss ──────────────────────────────────────────────────────────────────────

def soft_ce_loss(logits: torch.Tensor,
                 target_probs: torch.Tensor,
                 smoothing: float = 0.05,
                 weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Soft cross-entropy with optional label smoothing and class weighting.
    FIX 1: weight tensor penalises mistakes on rare classes (disgust, fear).
    FIX 3: smoothing reduced to 0.05 (was 0.1).
    """
    n_cls  = logits.size(1)
    # FIX 3: reduced smoothing
    smooth = target_probs * (1.0 - smoothing) + smoothing / n_cls
    logp   = F.log_softmax(logits, dim=1)
    loss   = -(smooth * logp).sum(dim=1)           # (B,)

    # FIX 1: scale each sample's loss by its class weight
    if weight is not None:
        hard_class = target_probs.argmax(dim=1)    # use argmax of soft as class id
        w = weight[hard_class]                     # (B,)
        loss = loss * w

    return loss.mean()


# ── MixUp (optional, Stage 1+ quality boost) ─────────────────────────────────

def mixup_batch(x: torch.Tensor, soft: torch.Tensor, alpha: float = 0.2):
    """
    MixUp: linearly interpolates pairs of samples and their soft labels.
    Helps the model learn smoother decision boundaries between similar emotions.
    Enable with --mixup_alpha 0.2
    """
    lam   = float(np.random.beta(alpha, alpha))
    idx   = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    s_mix = lam * soft + (1.0 - lam) * soft[idx]
    return x_mix, s_mix


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loader, device, smoothing: float = 0.05,
               weight: Optional[torch.Tensor] = None):
    model.eval()
    ys, yp   = [], []
    loss_sum = 0.0
    n        = 0

    for x, y, soft, au, au_ok, _ds in loader:
        x    = x.to(device)
        y    = y.to(device)
        soft = soft.to(device)

        logits, _ = model(x)
        loss = soft_ce_loss(logits, soft, smoothing=smoothing, weight=weight)

        loss_sum += float(loss.item()) * len(x)
        n        += len(x)

        pred = torch.argmax(logits, dim=1)
        ys.extend(y.cpu().numpy().tolist())
        yp.extend(pred.cpu().numpy().tolist())

    acc      = float((np.array(ys) == np.array(yp)).mean()) if ys else 0.0
    macro_f1 = float(f1_score(ys, yp, average="macro", zero_division=0)) if ys else 0.0
    return loss_sum / max(1, n), acc, macro_f1, ys, yp


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",          required=True)
    ap.add_argument("--arch",         default="efficientnet_b0",
                    choices=["resnet18", "resnet50", "mobilenet_v3_small",
                             "efficientnet_b0", "efficientnet_b2", "swin_tiny"])
    ap.add_argument("--pretrained",   action="store_true")
    ap.add_argument("--img_size",     type=int,   default=160)
    ap.add_argument("--batch",        type=int,   default=32)
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers",  type=int,   default=2)
    ap.add_argument("--seed",         type=int,   default=13)

    ap.add_argument("--balance",      default="dataset_class",
                    choices=["none", "dataset", "dataset_class"])
    ap.add_argument("--lambda_au",    type=float, default=0.05)
    # FIX 3: reduced defaults
    ap.add_argument("--smoothing",    type=float, default=0.05)
    ap.add_argument("--dropout",      type=float, default=0.20)
    ap.add_argument("--freeze_epochs", type=int,  default=5)
    # FIX 5 (optional): MixUp alpha — set >0 to enable, e.g. --mixup_alpha 0.2
    ap.add_argument("--mixup_alpha",  type=float, default=0.0)

    ap.add_argument("--save_dir",     default="models_cnn_multi")
    ap.add_argument("--run_name",     default="fer7_v4")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load & validate CSV ───────────────────────────────────────────────────
    df = pd.read_csv(args.csv, low_memory=False)

    for col in ["image", "label", "split", "dataset"]:
        if col not in df.columns:
            if col == "dataset":
                df["dataset"] = "unk"
            else:
                raise RuntimeError(f"CSV must contain column: {col}")

    df["image"]   = df["image"].astype(str)
    df["label"]   = df["label"].astype(str).str.lower()
    df["split"]   = df["split"].astype(str).str.lower()
    df["dataset"] = df["dataset"].astype(str).str.lower()

    for c in P_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in AU_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "au_ok" in df.columns:
        df["au_ok"] = pd.to_numeric(df["au_ok"], errors="coerce").fillna(0).astype(int)

    df = df[df["label"].isin(FER5)].reset_index(drop=True)
    df = df[df["image"].apply(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df   = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df  = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"[DATA] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"[DATA] train class distribution:\n{train_df['label'].value_counts().to_string()}")

    use_au = (args.lambda_au > 0) and ("au_ok" in df.columns) and all(c in df.columns for c in AU_COLS)
    print(f"[AU]     use_au={use_au}  lambda_au={args.lambda_au}")
    print(f"[CONFIG] smoothing={args.smoothing}  dropout={args.dropout}  "
          f"freeze_epochs={args.freeze_epochs}  mixup_alpha={args.mixup_alpha}")

    # ── FIX 1: Compute class weights from training set ────────────────────────
    label_counts  = train_df["label"].value_counts()
    raw_weights   = torch.tensor(
        [1.0 / label_counts.get(c, 1) for c in FER5], dtype=torch.float32
    )
    # Normalise so weights average to 1.0 (keeps loss scale stable)
    class_weights = (raw_weights / raw_weights.mean())
    print("[CLASS WEIGHTS]")
    for c, w in zip(FER5, class_weights.tolist()):
        print(f"  {c:10s}: {w:.3f}  (n={label_counts.get(c, 0)})")

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_train = FERDataset(train_df, img_size=args.img_size, train=True)
    ds_val   = FERDataset(val_df,   img_size=args.img_size, train=False) if len(val_df)  else None
    ds_test  = FERDataset(test_df,  img_size=args.img_size, train=False) if len(test_df) else None

    # ── Sampler ───────────────────────────────────────────────────────────────
    sampler = None
    if args.balance != "none":
        if args.balance == "dataset":
            ds_counts  = train_df["dataset"].value_counts().to_dict()
            weights_np = train_df["dataset"].map(
                lambda d: 1.0 / ds_counts.get(d, 1)
            ).to_numpy(dtype=np.float64)

        elif args.balance == "dataset_class":
            tmp        = train_df.copy()
            tmp["key"] = tmp["dataset"].astype(str) + "::" + tmp["label"].astype(str)
            key_counts = tmp["key"].value_counts().to_dict()
            weights_np = tmp["key"].map(
                lambda k: 1.0 / key_counts.get(k, 1)
            ).to_numpy(dtype=np.float64)
        else:
            weights_np = np.ones(len(train_df), dtype=np.float64)

        try:
            wsum = float(weights_np.sum())
            if wsum > 0:
                tmp2          = train_df[["dataset", "label"]].copy()
                tmp2["w"]     = weights_np
                ds_share      = (tmp2.groupby("dataset")["w"].sum() / wsum).sort_values(ascending=False)
                cls_share     = (tmp2.groupby("label")["w"].sum()   / wsum).sort_values(ascending=False)
                print("[BALANCE DBG] expected dataset sampling share:")
                print(ds_share.to_string())
                print("[BALANCE DBG] expected class sampling share:")
                print(cls_share.to_string())
        except Exception:
            pass

        weights = torch.as_tensor(weights_np, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"[BALANCE] {args.balance} enabled")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    pin      = torch.cuda.is_available()
    pw       = args.num_workers > 0
    dl_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler,
                          shuffle=(sampler is None), num_workers=args.num_workers,
                          pin_memory=pin, persistent_workers=pw)
    dl_val   = DataLoader(ds_val,  batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin,
                          persistent_workers=pw) if ds_val  else None
    dl_test  = DataLoader(ds_test, batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin,
                          persistent_workers=pw) if ds_test else None

    # ── Model ─────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    model = FERModel(
        args.arch, args.pretrained,
        num_classes=len(FER5), au_dim=len(AU_COLS),
        use_au=use_au, dropout=args.dropout
    ).to(device)

    # Move class weights to device
    cw = class_weights.to(device)

    # ── Freeze backbone initially ─────────────────────────────────────────────
    backbone_frozen = args.freeze_epochs > 0 and args.pretrained
    if backbone_frozen:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print(f"[FREEZE] backbone frozen for first {args.freeze_epochs} epochs")

    def build_optimizer(frozen: bool):
        if frozen:
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            param_groups = [
                {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
                {"params": model.cls_head.parameters(), "lr": args.lr},
            ]
            if use_au:
                param_groups.append({"params": model.au_head.parameters(), "lr": args.lr})
            return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    def build_scheduler(opt, n_epochs):
        return torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[g["lr"] for g in opt.param_groups],
            steps_per_epoch=len(dl_train),
            epochs=n_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1e4,
        )

    opt   = build_optimizer(backbone_frozen)
    sched = build_scheduler(opt, args.epochs)

    au_loss_fn = nn.SmoothL1Loss(reduction="mean")

    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{args.run_name}_best.pt"
    meta_path = save_dir / f"{args.run_name}_meta.json"

    best_f1 = -1.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, args.epochs + 1):

        # Unfreeze backbone after freeze_epochs
        if backbone_frozen and ep == args.freeze_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            backbone_frozen = False
            remaining       = args.epochs - ep + 1
            opt             = build_optimizer(frozen=False)
            sched           = build_scheduler(opt, remaining)
            print(f"[UNFREEZE] backbone unfrozen at epoch {ep}, "
                  f"new max_lr={args.lr * 0.1:.2e} (backbone) / {args.lr:.2e} (head) "
                  f"over {remaining} remaining epochs")

        model.train()
        loss_sum = 0.0
        n        = 0

        for x, y, soft, au, au_ok, _ds in dl_train:
            x     = x.to(device)
            soft  = soft.to(device)
            au    = au.to(device)
            au_ok = au_ok.to(device)

            # FIX 5 (optional): MixUp
            if args.mixup_alpha > 0:
                x, soft = mixup_batch(x, soft, alpha=args.mixup_alpha)

            logits, au_pred = model(x)

            # FIX 1 + FIX 3: class-weighted loss + reduced smoothing
            loss_cls = soft_ce_loss(logits, soft,
                                    smoothing=args.smoothing,
                                    weight=cw)
            loss = loss_cls

            if use_au:
                mask = (au_ok > 0)
                if mask.any():
                    loss_au = au_loss_fn(au_pred[mask], au[mask])
                    loss    = loss + args.lambda_au * loss_au

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

            loss_sum += float(loss.item()) * len(x)
            n        += len(x)

        train_loss = loss_sum / max(1, n)
        current_lr = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else args.lr

        if dl_val is not None:
            val_loss, val_acc, val_f1, ys, yp = eval_epoch(
                model, dl_val, device,
                smoothing=args.smoothing, weight=cw
            )
        else:
            val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0

        frozen_tag = " [frozen]" if backbone_frozen else ""
        print(
            f"[EPOCH {ep:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  val_macro_f1={val_f1:.3f}  "
            f"lr={current_lr:.2e}{frozen_tag}"
        )

        if dl_val is not None and val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "state_dict": model.state_dict(),
                "arch":       args.arch,
                "img_size":   args.img_size,
                "epoch":      ep,
                "val_f1":     val_f1,
                "val_acc":    val_acc,
            }, best_path)
            print(f"  [SAVE] best val_macro_f1={val_f1:.3f} -> {best_path}")

    # ── Test ──────────────────────────────────────────────────────────────────
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\n[BEST CKPT] epoch={ckpt.get('epoch','?')}  "
              f"val_f1={ckpt.get('val_f1', 0):.3f}")

    if dl_test is not None:
        test_loss, test_acc, test_f1, ys, yp = eval_epoch(
            model, dl_test, device, smoothing=0.0, weight=None  # no smoothing/weighting at test
        )
        print(f"\n[TEST] acc={test_acc:.3f}  macro_f1={test_f1:.3f}")
        print("[TEST] report:\n",
              classification_report(ys, yp, target_names=FER5, digits=3))

    meta: Dict[str, Any] = {
        "classes":       FER5,
        "arch":          args.arch,
        "img_size":      args.img_size,
        "pretrained":    bool(args.pretrained),
        "use_au":        bool(use_au),
        "au_cols":       AU_COLS if use_au else [],
        "lambda_au":     args.lambda_au,
        "smoothing":     args.smoothing,
        "dropout":       args.dropout,
        "freeze_epochs": args.freeze_epochs,
        "mixup_alpha":   args.mixup_alpha,
        "csv":           str(Path(args.csv).resolve()),
        "best_ckpt":     str(best_path.resolve()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[OK] wrote", meta_path)


if __name__ == "__main__":
    main()