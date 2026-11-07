import os
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit


FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

KAGGLE0 = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
OFFICIAL1 = {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_label(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = s.replace("happiness", "happy").replace("sadness", "sad").replace("surprised", "surprise").replace("fearful", "fear")
    return s


def infer_numeric_scheme(vals) -> dict | None:
    vals = pd.Series(vals).dropna()
    try:
        v = vals.astype(int)
    except Exception:
        return None
    mn, mx = int(v.min()), int(v.max())
    if mn == 0 and mx == 6:
        return KAGGLE0
    if mn == 1 and mx == 7:
        return OFFICIAL1
    return None


def auto_cols(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}
    path_candidates = [c for c in cols if any(k in lower[c] for k in ["image", "file", "path", "name", "img"])]
    label_candidates = [c for c in cols if any(k in lower[c] for k in ["label", "emotion", "expression", "class"])]

    if not path_candidates:
        path_col = cols[0]
    else:
        path_col = path_candidates[0]

    if not label_candidates:
        label_col = cols[1] if len(cols) > 1 else cols[0]
    else:
        label_col = label_candidates[0]

    return path_col, label_col


def build_image_index(images_root: Path) -> dict:
    idx = {}
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        for fp in images_root.rglob(ext):
            idx.setdefault(fp.name, str(fp.resolve()))
    return idx


def resolve_image(raw, images_root: Path, idx: dict) -> str | None:
    raw = str(raw).strip()

    # absolute path
    if os.path.isfile(raw):
        return str(Path(raw).resolve())

    # relative to images_root
    p = images_root / raw
    if p.is_file():
        return str(p.resolve())

    # by basename
    base = Path(raw).name
    if base in idx:
        return idx[base]

    return None


def stratified_train_val_test(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    """
    Returns df with a 'split' column: train / val / test.
    Works even when test_ratio == 0.0 (no test split).
    """
    assert 0 <= val_ratio < 1 and 0 <= test_ratio < 1 and (val_ratio + test_ratio) < 1

    y = df["label"].to_numpy()
    n = len(df)
    idx_all = np.arange(n)

    # ---------- case 1: no test split ----------
    if test_ratio <= 0:
        df_tv = df.copy().reset_index(drop=True)

        if val_ratio > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
            idx_train, idx_val = next(sss.split(np.arange(len(df_tv)), df_tv["label"].to_numpy()))
            df_train = df_tv.iloc[idx_train].copy().reset_index(drop=True)
            df_val = df_tv.iloc[idx_val].copy().reset_index(drop=True)
            df_train["split"] = "train"
            df_val["split"] = "val"
            return pd.concat([df_train, df_val], ignore_index=True)

        df_tv["split"] = "train"
        return df_tv

    # ---------- case 2: normal train/val/test ----------
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(idx_all, y))

    df_tv = df.iloc[idx_trainval].copy().reset_index(drop=True)
    df_test = df.iloc[idx_test].copy().reset_index(drop=True)
    df_test["split"] = "test"

    if val_ratio > 0:
        y_tv = df_tv["label"].to_numpy()
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio / (1.0 - test_ratio), random_state=seed)
        idx_tv = np.arange(len(df_tv))
        idx_train, idx_val = next(sss2.split(idx_tv, y_tv))

        df_train = df_tv.iloc[idx_train].copy().reset_index(drop=True)
        df_val = df_tv.iloc[idx_val].copy().reset_index(drop=True)
        df_train["split"] = "train"
        df_val["split"] = "val"
        return pd.concat([df_train, df_val, df_test], ignore_index=True)

    df_tv["split"] = "train"
    return pd.concat([df_tv, df_test], ignore_index=True)


def load_rafdb(raf_root: Path, val_ratio: float, seed: int) -> pd.DataFrame:
    train_csv = raf_root / "train_labels_clean_kfold.csv"
    if not train_csv.exists():
        train_csv = raf_root / "train_labels.csv"
    test_csv = raf_root / "test_labels.csv"
    images_root = raf_root / "DATASET"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("RAF-DB expects train_labels.csv and test_labels.csv in RAF-DB root")

    df_tr = read_table(train_csv)
    df_te = read_table(test_csv)
    path_col_tr, label_col_tr = auto_cols(df_tr)
    path_col_te, label_col_te = auto_cols(df_te)

    idx = build_image_index(images_root)

    def conv(df, path_col, label_col):
        scheme = infer_numeric_scheme(df[label_col])
        rows = []
        for _, r in df.iterrows():
            imgp = resolve_image(r[path_col], images_root, idx)
            if not imgp:
                continue
            lab_raw = r[label_col]
            lab = normalize_label(lab_raw)
            if lab in FER7:
                lab7 = lab
            elif scheme is not None:
                try:
                    lab7 = scheme[int(lab_raw)]
                except Exception:
                    continue
            else:
                continue
            rows.append({"image": imgp, "label": lab7})
        return pd.DataFrame(rows)

    out_tr = conv(df_tr, path_col_tr, label_col_tr)
    out_te = conv(df_te, path_col_te, label_col_te)

    # create val split from train
    out_tr = stratified_train_val_test(out_tr, val_ratio=val_ratio, test_ratio=0.0, seed=seed)
    out_te["split"] = "test"

    out = pd.concat([out_tr, out_te], ignore_index=True)
    out["dataset"] = "rafdb"
    return out


def find_ferplus_data_root(datasets_root: Path) -> Path:
    # common place: DATASETS\FERPlusData
    cand = datasets_root / "FERPlusData"
    if (cand / "FER2013Train" / "label.csv").exists():
        return cand

    # otherwise search
    for p in datasets_root.rglob("FER2013Train"):
        if (p / "label.csv").exists():
            return p.parent

    raise FileNotFoundError("Could not find FERPlusData with FER2013Train/label.csv. Run generate_training_data.py first.")


def load_ferplus(datasets_root: Path, min_votes_7: int = 6) -> pd.DataFrame:
    """
    Robust FERPlus loader.

    Many FERPlusData/FER2013Train/label.csv files have NO HEADER.
    In that case pandas will interpret the first data row as header unless header=None.

    We handle:
    - 12 cols: file, rect, [10 vote cols]
    - 13 cols: file, rect, emotion, [10 vote cols]
    - >13 cols: rect might be split; we just take LAST 10 cols as votes, col0 as filename.
    """
    base = find_ferplus_data_root(datasets_root)

    split_map = {
        "FER2013Train": "train",
        "FER2013Valid": "val",
        "FER2013Test": "test",
    }

    # Vote column order used by FERPlus toolkits is commonly:
    # neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, nf
    vote_names = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "nf"]

    all_rows = []

    for folder, split in split_map.items():
        d = base / folder
        label_csv = d / "label.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"Missing {label_csv}")

        # IMPORTANT: header=None because many FERPlus label.csv have no header row
        df = pd.read_csv(label_csv, header=None)

        if df.shape[1] < 11:
            raise RuntimeError(
                f"FERPlus label.csv has too few columns ({df.shape[1]}). "
                f"Expected at least filename + 10 vote columns. File: {label_csv}"
            )

        # Filename is always first column in FERPlus output
        file_col = 0

        # Last 10 columns are the vote columns (robust even if rect/emotion columns vary)
        votes = df.iloc[:, -10:].copy()
        votes.columns = vote_names

        # Convert votes to numeric safely
        for c in vote_names:
            votes[c] = pd.to_numeric(votes[c], errors="coerce").fillna(0.0)

        # Decide if these look like "counts" (~10 total) or "probabilities" (~1 total)
        total_vote_median = float(votes.sum(axis=1).median())

        # Iterate rows
        for i, row in df.iterrows():
            fn = str(row[file_col]).strip()
            imgp = d / fn
            if not imgp.exists():
                continue

            # 7-class votes
            v_neu = float(votes.at[i, "neutral"])
            v_hap = float(votes.at[i, "happiness"])
            v_sur = float(votes.at[i, "surprise"])
            v_sad = float(votes.at[i, "sadness"])
            v_ang = float(votes.at[i, "anger"])
            v_dis = float(votes.at[i, "disgust"])
            v_fea = float(votes.at[i, "fear"])

            v7 = np.array([v_ang, v_dis, v_fea, v_hap, v_sad, v_sur, v_neu], dtype=np.float32)
            s7 = float(v7.sum())

            # extra votes (noise indicators)
            extra = float(votes.at[i, "contempt"] + votes.at[i, "unknown"] + votes.at[i, "nf"])

            # Filtering (cleaning)
            # If votes are counts (median total ~10): require enough mass on the 7 basic emotions
            # If votes are probs (median total ~1): require enough mass too (lower threshold)
            if total_vote_median > 2.0:
                if s7 < min_votes_7:
                    continue
            else:
                # probabilities-like
                if s7 < 0.60:
                    continue

            p7 = v7 / (s7 + 1e-6)
            lab = FER7[int(np.argmax(p7))]

            all_rows.append({
                "image": str(imgp.resolve()),
                "label": lab,
                "split": split,
                "dataset": "ferplus",

                # soft targets
                "p_angry": float(p7[0]),
                "p_disgust": float(p7[1]),
                "p_fear": float(p7[2]),
                "p_happy": float(p7[3]),
                "p_sad": float(p7[4]),
                "p_surprise": float(p7[5]),
                "p_neutral": float(p7[6]),

                # diagnostics
                "ferplus_votes7": float(s7),
                "ferplus_extra_votes": float(extra),
                "ferplus_total_vote_median": total_vote_median,
            })

    return pd.DataFrame(all_rows)


def find_expw_labels(expw_root: Path) -> Path:
    cands = []
    for ext in ["*.csv", "*.txt"]:
        for p in expw_root.rglob(ext):
            name = p.name.lower()
            if any(k in name for k in ["label", "labels", "annot", "annotation", "emotion", "expw"]):
                cands.append(p)

    if not cands:
        # fallback: take largest csv/txt file
        all_files = []
        for ext in ["*.csv", "*.txt"]:
            all_files += list(expw_root.rglob(ext))
        if not all_files:
            raise FileNotFoundError("ExpW: no csv/txt labels found")
        return max(all_files, key=lambda p: p.stat().st_size)

    return max(cands, key=lambda p: p.stat().st_size)


def find_expw_images_root(expw_root: Path) -> Path:
    # pick folder with most images
    best = expw_root
    best_count = 0
    for d in [expw_root] + [p for p in expw_root.iterdir() if p.is_dir()]:
        cnt = 0
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            cnt += len(list(d.rglob(ext)))
        if cnt > best_count:
            best = d
            best_count = cnt
    return best


def load_expw(expw_root: Path, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    label_file = find_expw_labels(expw_root)
    images_root = find_expw_images_root(expw_root)
    idx = build_image_index(images_root)

    if label_file.suffix.lower() == ".csv":
        df = pd.read_csv(label_file)
        path_col, label_col = auto_cols(df)
        scheme = infer_numeric_scheme(df[label_col])

        rows = []
        for _, r in df.iterrows():
            imgp = resolve_image(r[path_col], images_root, idx)
            if not imgp:
                continue
            lab_raw = r[label_col]
            lab = normalize_label(lab_raw)
            if lab in FER7:
                lab7 = lab
            elif scheme is not None:
                try:
                    lab7 = scheme[int(lab_raw)]
                except Exception:
                    continue
            else:
                continue
            rows.append({"image": imgp, "label": lab7})

        out = pd.DataFrame(rows)
    else:
        # txt: expect lines like "filename label"
        rows = []
        scheme = None
        with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]
                # guess which is label
                lab_guess = normalize_label(a)
                if lab_guess in FER7 or lab_guess.isdigit():
                    lab_raw, img_raw = a, b
                else:
                    lab_raw, img_raw = b, a

                lab = normalize_label(lab_raw)
                if lab in FER7:
                    lab7 = lab
                else:
                    if scheme is None and lab_raw.isdigit():
                        # infer from seen numeric ranges later; default kaggle0
                        scheme = KAGGLE0
                    if scheme is None:
                        continue
                    try:
                        lab7 = scheme[int(lab_raw)]
                    except Exception:
                        continue

                imgp = resolve_image(img_raw, images_root, idx)
                if not imgp:
                    continue
                rows.append({"image": imgp, "label": lab7})
        out = pd.DataFrame(rows)

    out = stratified_train_val_test(out, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    out["dataset"] = "expw"
    out["expw_labels_file"] = str(label_file)
    out["expw_images_root"] = str(images_root)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", required=True)
    ap.add_argument("--out_csv", default="dataset_csv/unified_fer7_raw.csv")
    ap.add_argument("--raf_val_ratio", type=float, default=0.15)
    ap.add_argument("--expw_val_ratio", type=float, default=0.10)
    ap.add_argument("--expw_test_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)

    raf = load_rafdb(datasets_root / "RAF-DB", val_ratio=args.raf_val_ratio, seed=args.seed)
    ferp = load_ferplus(datasets_root)
    expw = load_expw(datasets_root / "ExpW", val_ratio=args.expw_val_ratio, test_ratio=args.expw_test_ratio, seed=args.seed)

    # Ensure columns exist (soft probs empty for non-ferplus)
    for c in ["p_angry","p_disgust","p_fear","p_happy","p_sad","p_surprise","p_neutral"]:
        if c not in raf.columns:
            raf[c] = np.nan
        if c not in expw.columns:
            expw[c] = np.nan

    all_df = pd.concat([raf, ferp, expw], ignore_index=True)

    # keep only FER7 labels
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