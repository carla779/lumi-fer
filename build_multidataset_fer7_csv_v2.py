from __future__ import annotations

import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLS_SET = set(FER7)

# 0..6 scheme used by FER2013 and many ExpW versions
KAGGLE0 = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
# 1..7 scheme used by some datasets
OFFICIAL1 = {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def normalize_label(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = s.replace("happiness", "happy").replace("sadness", "sad").replace("surprised", "surprise")
    s = s.replace("anger", "angry")
    return s


def build_image_index(images_root: Path) -> dict:
    """
    Map filename + stem -> absolute path (also store lowercase keys).
    This fixes cases where label.lst says .jpg but file is .png, etc.
    """
    idx = {}
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        for fp in images_root.rglob(ext):
            p = str(fp.resolve())
            keys = {fp.name, fp.name.lower(), fp.stem, fp.stem.lower()}
            for k in keys:
                idx.setdefault(k, p)
    return idx


def resolve_image(raw, images_root: Path, idx: dict) -> str | None:
    if raw is None:
        return None
    raw = str(raw).strip().strip('"').strip("'")
    if not raw:
        return None

    # absolute path
    if os.path.isfile(raw):
        return str(Path(raw).resolve())

    base = Path(raw).name

    # direct relative
    p = images_root / raw
    if p.is_file():
        return str(p.resolve())

    # index lookup (case + stem)
    if base in idx:
        return idx[base]
    b = base.lower()
    if b in idx:
        return idx[b]

    stem = Path(base).stem
    if stem in idx:
        return idx[stem]
    s = stem.lower()
    if s in idx:
        return idx[s]

    return None


def stratified_train_val_test(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    """
    Adds a 'split' column: train/val/test.
    Works even when test_ratio == 0.0 (no test split).
    """
    assert 0 <= val_ratio < 1 and 0 <= test_ratio < 1 and (val_ratio + test_ratio) < 1
    if df.empty:
        raise RuntimeError("Cannot split an empty dataframe.")

    # no test split
    if test_ratio <= 0:
        df_tv = df.copy().reset_index(drop=True)
        if val_ratio > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
            idx_all = np.arange(len(df_tv))
            y = df_tv["label"].to_numpy()
            idx_train, idx_val = next(sss.split(idx_all, y))
            tr = df_tv.iloc[idx_train].copy().reset_index(drop=True)
            va = df_tv.iloc[idx_val].copy().reset_index(drop=True)
            tr["split"] = "train"
            va["split"] = "val"
            return pd.concat([tr, va], ignore_index=True)
        df_tv["split"] = "train"
        return df_tv

    # trainval / test
    y = df["label"].to_numpy()
    idx_all = np.arange(len(df))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(idx_all, y))

    df_tv = df.iloc[idx_trainval].copy().reset_index(drop=True)
    df_te = df.iloc[idx_test].copy().reset_index(drop=True)
    df_te["split"] = "test"

    if val_ratio > 0:
        y_tv = df_tv["label"].to_numpy()
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio / (1.0 - test_ratio), random_state=seed)
        idx_tv = np.arange(len(df_tv))
        idx_tr, idx_va = next(sss2.split(idx_tv, y_tv))
        df_tr = df_tv.iloc[idx_tr].copy().reset_index(drop=True)
        df_va = df_tv.iloc[idx_va].copy().reset_index(drop=True)
        df_tr["split"] = "train"
        df_va["split"] = "val"
        return pd.concat([df_tr, df_va, df_te], ignore_index=True)

    df_tv["split"] = "train"
    return pd.concat([df_tv, df_te], ignore_index=True)


# ---------------- RAF-DB ----------------
def load_rafdb(raf_root: Path, val_ratio: float, seed: int) -> pd.DataFrame:
    """
    RAF-DB loader (robust):
      - uses train_labels_clean_kfold.csv if present, else train_labels.csv
      - uses test_labels.csv for test split
      - auto-detects image path column + label column
      - infers numeric label scheme:
          * OFFICIAL1 (1..7)  -> {1:surprise,2:fear,3:disgust,4:happy,5:sad,6:angry,7:neutral}
          * KAGGLE0  (0..6)   -> {0:angry,1:disgust,2:fear,3:happy,4:sad,5:surprise,6:neutral}
      - creates a stratified val split from train
    """
    train_csv = raf_root / "train_labels_clean_kfold.csv"
    if not train_csv.exists():
        train_csv = raf_root / "train_labels.csv"

    test_csv = raf_root / "test_labels.csv"
    images_root = raf_root / "DATASET"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"RAF-DB missing train/test label CSVs in {raf_root}")

    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    # --- auto-detect columns ---
    def pick_cols(df: pd.DataFrame):
        cols = list(df.columns)
        low = {c: c.lower() for c in cols}

        path_col = next(
            (c for c in cols if any(k in low[c] for k in ["image", "img", "path", "file", "name"])),
            cols[0],
        )
        label_col = next(
            (c for c in cols if any(k in low[c] for k in ["label", "emotion", "expression", "class"])),
            cols[1] if len(cols) > 1 else cols[0],
        )
        return path_col, label_col

    p_tr, y_tr = pick_cols(df_tr)
    p_te, y_te = pick_cols(df_te)

    # --- infer RAF scheme from TRAIN labels ---
    num_series = pd.to_numeric(df_tr[y_tr], errors="coerce").dropna()
    if not num_series.empty:
        vals = set(num_series.astype(int).unique().tolist())
        if 7 in vals:
            scheme = OFFICIAL1
            scheme_name = "official1"
        elif 0 in vals:
            scheme = KAGGLE0
            scheme_name = "kaggle0"
        else:
            # RAF is usually official1; choose that as safe default
            scheme = OFFICIAL1
            scheme_name = "official1_default"
    else:
        scheme = OFFICIAL1
        scheme_name = "official1_default"

    idx = build_image_index(images_root)

    def conv(df: pd.DataFrame, pcol: str, ycol: str) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            imgp = resolve_image(r[pcol], images_root, idx)
            if not imgp:
                continue

            lab_raw = r[ycol]
            lab = normalize_label(lab_raw)

            if lab in CLS_SET:
                lab7 = lab
            else:
                # numeric label (also handles "1.0")
                try:
                    v = int(float(lab_raw))
                except Exception:
                    continue
                lab7 = scheme.get(v)
                if lab7 is None:
                    continue

            rows.append({"image": imgp, "label": lab7})

        return pd.DataFrame(rows)

    out_tr = conv(df_tr, p_tr, y_tr)
    out_te = conv(df_te, p_te, y_te)

    if out_tr.empty:
        raise RuntimeError(f"RAF-DB: train rows parsed = 0. Check columns in {train_csv}")
    if out_te.empty:
        raise RuntimeError(f"RAF-DB: test rows parsed = 0. Check columns in {test_csv}")

    # train/val from train; test stays test
    out_tr = stratified_train_val_test(out_tr, val_ratio=val_ratio, test_ratio=0.0, seed=seed)
    out_te["split"] = "test"

    out = pd.concat([out_tr, out_te], ignore_index=True)
    out["dataset"] = "rafdb"
    out["raf_scheme"] = scheme_name
    return out


# ---------------- FERPlusData ----------------
def find_ferplus_data_root(datasets_root: Path) -> Path:
    cand = datasets_root / "FERPlusData"
    if (cand / "FER2013Train" / "label.csv").exists():
        return cand
    # fallback search
    for p in datasets_root.rglob("FER2013Train"):
        if (p / "label.csv").exists():
            return p.parent
    raise FileNotFoundError("Could not find FERPlusData with FER2013Train/label.csv")


def load_ferplus(datasets_root: Path, min_votes_7: int = 6) -> pd.DataFrame:
    base = find_ferplus_data_root(datasets_root)

    split_map = {
        "FER2013Train": "train",
        "FER2013Valid": "val",
        "FER2013Test": "test",
    }

    # expected FERPlus vote order (last 10 columns)
    vote_names = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "nf"]

    rows = []
    for folder, split in split_map.items():
        d = base / folder
        label_csv = d / "label.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"Missing {label_csv}")

        df = pd.read_csv(label_csv, header=None)
        if df.shape[1] < 11:
            raise RuntimeError(f"FERPlus label.csv has too few columns: {df.shape[1]} in {label_csv}")

        # filename = col 0 ; votes = last 10 cols
        votes = df.iloc[:, -10:].copy()
        votes.columns = vote_names
        for c in vote_names:
            votes[c] = pd.to_numeric(votes[c], errors="coerce").fillna(0.0)

        total_median = float(votes.sum(axis=1).median())

        for i in range(len(df)):
            fn = str(df.iat[i, 0]).strip()
            imgp = d / fn
            if not imgp.exists():
                continue

            v_neu = float(votes.at[i, "neutral"])
            v_hap = float(votes.at[i, "happiness"])
            v_sur = float(votes.at[i, "surprise"])
            v_sad = float(votes.at[i, "sadness"])
            v_ang = float(votes.at[i, "anger"])
            v_dis = float(votes.at[i, "disgust"])
            v_fea = float(votes.at[i, "fear"])

            v7 = np.array([v_ang, v_dis, v_fea, v_hap, v_sad, v_sur, v_neu], dtype=np.float32)
            s7 = float(v7.sum())
            extra = float(votes.at[i, "contempt"] + votes.at[i, "unknown"] + votes.at[i, "nf"])

            # clean filter
            if total_median > 2.0:
                if s7 < min_votes_7:
                    continue
            else:
                if s7 < 0.60:
                    continue

            p7 = v7 / (s7 + 1e-6)
            lab = FER7[int(np.argmax(p7))]

            rows.append({
                "image": str(imgp.resolve()),
                "label": lab,
                "split": split,
                "dataset": "ferplus",
                "p_angry": float(p7[0]),
                "p_disgust": float(p7[1]),
                "p_fear": float(p7[2]),
                "p_happy": float(p7[3]),
                "p_sad": float(p7[4]),
                "p_surprise": float(p7[5]),
                "p_neutral": float(p7[6]),
                "ferplus_votes7": float(s7),
                "ferplus_extra_votes": float(extra),
            })

    return pd.DataFrame(rows)


# ---------------- ExpW ----------------
def find_expw_images_root(expw_root: Path) -> Path:
    # choose folder with most images (among root + immediate subfolders)
    cands = [expw_root] + [p for p in expw_root.iterdir() if p.is_dir()]
    best = expw_root
    best_count = -1
    for d in cands:
        cnt = 0
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            cnt += len(list(d.rglob(ext)))
        if cnt > best_count:
            best_count = cnt
            best = d
    return best


def _to_int_like(t: str):
    t = t.strip().strip('"').strip("'").strip(",;")
    # accept 0, 1, 2 ... or 0.0, 1.0 ...
    try:
        v = int(t)
        return v
    except Exception:
        try:
            f = float(t)
            if abs(f - round(f)) < 1e-6:
                return int(round(f))
        except Exception:
            return None
    return None


def _parse_expw_line(line: str, images_root: Path, idx: dict) -> tuple[str | None, str | None]:
    """
    Returns (resolved_image_path, label_str).
    Robust to:
      - label first / image second
      - image without extension
      - numeric labels written as 0.0
      - bbox numbers present
    """
    line = line.strip().strip("\ufeff")
    if not line:
        return None, None

    # split by comma or whitespace (works for .lst/.txt)
    toks = [t for t in re.split(r"[,\s]+", line) if t]
    toks = [t.strip().strip('"').strip("'") for t in toks if t.strip()]
    if not toks:
        return None, None

    # 1) find the image token by checking what resolves to a real image
    img_path = None
    img_tok_index = None
    for j, t in enumerate(toks):
        p = resolve_image(t, images_root, idx)
        if p:
            img_path = p
            img_tok_index = j
            break

    if img_path is None:
        return None, None

    # 2) label: prefer explicit string label
    for t in toks:
        lab = normalize_label(t)
        if lab in CLS_SET:
            return img_path, lab

    # 3) numeric label: prefer neighbor of the image token
    candidates = []
    for j, t in enumerate(toks):
        v = _to_int_like(t)
        if v is None:
            continue
        if 0 <= v <= 6 or 1 <= v <= 7:
            candidates.append((j, v))

    if not candidates:
        return None, None

    # neighbor preference (label is usually near the image token)
    for off in [-2, -1, 1, 2]:
        k = img_tok_index + off
        for j, v in candidates:
            if j == k:
                lab = KAGGLE0.get(v) if 0 <= v <= 6 else OFFICIAL1.get(v)
                if lab in CLS_SET:
                    return img_path, lab

    # fallback: first valid candidate
    j, v = candidates[0]
    lab = KAGGLE0.get(v) if 0 <= v <= 6 else OFFICIAL1.get(v)
    if lab in CLS_SET:
        return img_path, lab

    return None, None


def _parse_expw_file(label_file: Path, images_root: Path, idx: dict, max_lines: int | None = None) -> list[dict]:
    rows = []
    with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            imgp, lab = _parse_expw_line(line, images_root, idx)
            if not imgp or not lab:
                continue
            rows.append({"image": imgp, "label": lab})
    return rows


def load_expw(expw_root: Path, val_ratio: float, test_ratio: float, seed: int,
             force_labels: str | None = None, force_images_root: str | None = None,
             min_conf: float = 10.0) -> pd.DataFrame:
    """
    ExpW label.lst format (whitespace-separated, 8 columns):
      image_name face_id_in_image face_box_top face_box_left face_box_right face_box_bottom face_box_confidence expression_label

    expression_label mapping (ExpW):
      0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
    """
    expw_root = Path(expw_root)

    # --- label file ---
    if force_labels:
        label_file = Path(force_labels)
    else:
        cand = expw_root / "data" / "label" / "label.lst"
        if cand.exists():
            label_file = cand
        else:
            hits = list(expw_root.rglob("label.lst"))
            if not hits:
                raise FileNotFoundError(f"ExpW: could not find label.lst under {expw_root}")
            label_file = hits[0]

    # --- where images live ---
    if force_images_root:
        images_root = Path(force_images_root)
    else:
        # common: ExpW/data contains origin/images folders
        images_root = expw_root / "data" if (expw_root / "data").exists() else expw_root

    idx = build_image_index(images_root)

    df = pd.read_csv(label_file, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 8:
        raise RuntimeError(f"ExpW label.lst expected >=8 columns, got {df.shape[1]} in {label_file}")

    df = df.iloc[:, :8].copy()
    df.columns = [
        "image_name",
        "face_id_in_image",
        "face_box_top",
        "face_box_left",
        "face_box_right",
        "face_box_bottom",
        "face_box_confidence",
        "expression_label",
    ]

    # numeric conversions
    for c in ["face_id_in_image", "face_box_top", "face_box_left", "face_box_right", "face_box_bottom", "expression_label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["face_box_confidence"] = pd.to_numeric(df["face_box_confidence"], errors="coerce")

    df = df.dropna(subset=["image_name", "expression_label", "face_box_left", "face_box_top", "face_box_right", "face_box_bottom"]).copy()
    df["expression_label"] = df["expression_label"].astype(int)

    # ExpW mapping = same as FER2013
    expw_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

    rows = []
    missing_img = 0
    low_conf = 0
    bad_label = 0

    for i, r in df.iterrows():
        conf = float(r["face_box_confidence"]) if pd.notna(r["face_box_confidence"]) else 0.0
        if conf < float(min_conf):
            low_conf += 1
            continue

        v = int(r["expression_label"])
        lab = expw_map.get(v)
        if lab is None:
            bad_label += 1
            continue

        imgp = resolve_image(r["image_name"], images_root, idx)
        if not imgp:
            missing_img += 1
            continue

        # unique id so later cropping won't overwrite
        face_id = int(r["face_id_in_image"]) if pd.notna(r["face_id_in_image"]) else 0
        stem = Path(str(r["image_name"])).stem
        uid = f"expw_{stem}_f{face_id}_{i}"

        rows.append({
            "uid": uid,
            "image": imgp,
            "label": lab,
            "dataset": "expw",

            # keep bbox so we can crop exactly the annotated face later (better than re-detecting)
            "bbox_top": int(r["face_box_top"]),
            "bbox_left": int(r["face_box_left"]),
            "bbox_right": int(r["face_box_right"]),
            "bbox_bottom": int(r["face_box_bottom"]),
            "bbox_conf": conf,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(
            f"ExpW: 0 rows after parsing.\n"
            f"label_file={label_file}\nimages_root={images_root}\n"
            f"missing_img={missing_img} low_conf={low_conf} bad_label={bad_label}\n"
            f"Tip: pass --expw_images_root to the folder that actually contains the images."
        )

    # Create splits (ExpW has no official split here)
    out = stratified_train_val_test(out, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--raf_val_ratio", type=float, default=0.15)
    ap.add_argument("--expw_val_ratio", type=float, default=0.10)
    ap.add_argument("--expw_test_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ferplus_min_votes7", type=int, default=6)
    ap.add_argument("--expw_labels", default=None)
    ap.add_argument("--expw_images_root", default=None)
    args = ap.parse_args()

    root = Path(args.datasets_root)

    raf = load_rafdb(root / "rafdb", val_ratio=args.raf_val_ratio, seed=args.seed)
    ferp = load_ferplus(root, min_votes_7=args.ferplus_min_votes7)
    expw = load_expw(root / "expw",
                    val_ratio=args.expw_val_ratio, test_ratio=args.expw_test_ratio, seed=args.seed,
                    force_labels=args.expw_labels, force_images_root=args.expw_images_root)

    # Ensure soft-prob columns exist
    for df in (raf, expw):
        for c in ["p_angry","p_disgust","p_fear","p_happy","p_sad","p_surprise","p_neutral"]:
            if c not in df.columns:
                df[c] = np.nan

    all_df = pd.concat([raf, ferp, expw], ignore_index=True)
    all_df = all_df[all_df["label"].isin(FER7)].reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)

    print("[OK] wrote:", out_csv)
    print("\n[COUNTS by dataset/split]")
    print(all_df.groupby(["dataset", "split"])["label"].count())
    print("\n[COUNTS by dataset/label]")
    print(all_df.groupby(["dataset", "label"])["image"].count())


if __name__ == "__main__":
    main()