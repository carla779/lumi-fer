"""
patch_csv.py — substitute orig_image for image where image ends in 'nan.jpg'.

Idempotent: a backup at *_BACKUP.csv is created on first run only. Running
again on an already-patched CSV is a no-op.
"""
import os
import shutil
import sys

import pandas as pd

CSV = r"C:\Users\super\PyCharmMiscProject\unified_fer7_processed_au_FRIEND.csv"
BACKUP = CSV.replace(".csv", "_BACKUP.csv")


def main():
    if not os.path.isfile(CSV):
        print(f"[ERROR] CSV not found: {CSV}")
        sys.exit(1)

    if not os.path.isfile(BACKUP):
        shutil.copy(CSV, BACKUP)
        print(f"[OK] Backup written to {BACKUP}")
    else:
        print(f"[OK] Backup already exists at {BACKUP}; not overwriting")

    df = pd.read_csv(CSV, low_memory=False)
    n_before = len(df)
    mask = df["image"].astype(str).str.contains(r"nan\.jpg$", regex=True, na=False)
    n_broken = int(mask.sum())
    pct = 100 * n_broken / max(n_before, 1)
    print(f"[INFO] Rows with broken image path: {n_broken} ({pct:.1f}% of {n_before})")

    if n_broken > 0:
        if "orig_image" not in df.columns:
            print("[ERROR] orig_image column missing from CSV; cannot substitute.")
            sys.exit(1)
        df.loc[mask, "image"] = df.loc[mask, "orig_image"]
        df.to_csv(CSV, index=False)
        print(f"[OK] Substituted orig_image -> image on {n_broken} rows; saved.")
    else:
        print("[OK] No nan.jpg rows found; CSV unchanged.")

    # Re-verify resolution per dataset
    print("\n[VERIFY] Post-patch path resolution (sum / count per dataset):")
    exists = df["image"].apply(lambda p: os.path.isfile(str(p)))
    res = df.assign(ok=exists).groupby(df["dataset"].astype(str).str.lower())["ok"].agg(["sum", "count"])
    print(res.to_string())

    bad = []
    for ds, row in res.iterrows():
        if int(row["count"]) == 0:
            continue
        if int(row["sum"]) / int(row["count"]) < 0.95:
            bad.append(ds)
    if bad:
        print(f"\n[WARN] These datasets still have <95% path resolution: {bad}")
        print("       Some files may genuinely be missing on disk.")
        sys.exit(1)
    print("\n[OK] All datasets have >=95% resolution after patch.")


if __name__ == "__main__":
    main()
