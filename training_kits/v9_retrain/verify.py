"""
verify.py — sanity-check the unified FER training corpus before retraining v9.

Checks:
  1. processed_faces/<dataset>/ exists and has a non-trivial file count for
     all four datasets (rafdb, ferplus, expw, fer2013).
  2. The unified CSV exists, parses, and has rows for all four datasets.
  3. Per-dataset, the `image` column resolves to actual files on disk.

Exit codes:
  0  Corpus looks ready for training.
  1  At least one dataset is missing, broken, or has <95% path resolution.
"""
import os
import sys

import pandas as pd

PROJ = r"C:\Users\super\PyCharmMiscProject"
PROC = os.path.join(PROJ, "processed_faces")
CSV = os.path.join(PROJ, "unified_fer7_processed_au_FRIEND.csv")
DATASETS = ["rafdb", "ferplus", "expw", "fer2013"]


def banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70)


# ---------------------------------------------------------------- 1a: disk
banner("Step 1a: processed_faces/ on disk")
ok_disk = True
for d in DATASETS:
    p = os.path.join(PROC, d)
    if not os.path.isdir(p):
        print(f"  [MISSING] folder: {p}")
        ok_disk = False
        continue
    n = sum(len(files) for _, _, files in os.walk(p))
    flag = "[OK]   " if n >= 100 else "[BAD]  "
    print(f"  {flag} {d:10s} {n:>8d} files in {p}")
    if n < 100:
        ok_disk = False

# ---------------------------------------------------------------- 1b: CSV
banner("Step 1b: unified CSV")
if not os.path.isfile(CSV):
    print(f"  [MISSING] {CSV}")
    sys.exit(1)
df = pd.read_csv(CSV, low_memory=False)
df["dataset"] = df["dataset"].astype(str).str.lower()
print(f"  Rows: {len(df)}")
print(f"  Per-dataset row count:")
counts = df["dataset"].value_counts().to_dict()
for d in DATASETS:
    flag = "[OK]   " if counts.get(d, 0) >= 100 else "[BAD]  "
    print(f"    {flag} {d:10s}: {counts.get(d, 0)}")

# ---------------------------------------------------------------- 1c: paths
banner("Step 1c: per-dataset image-path resolution")
exists = df["image"].apply(lambda p: os.path.isfile(str(p)))
res = df.assign(ok=exists).groupby("dataset")["ok"].agg(["sum", "count"])
print(res.to_string())
print()

ok_paths = True
for d in DATASETS:
    if d not in res.index:
        print(f"  [MISSING] {d}: not present in CSV")
        ok_paths = False
        continue
    s, c = int(res.loc[d, "sum"]), int(res.loc[d, "count"])
    pct = 100 * s / max(c, 1)
    if c == 0:
        print(f"  [MISSING] {d}: zero rows in CSV")
        ok_paths = False
    elif pct < 95:
        print(f"  [BROKEN]  {d}: only {s}/{c} ({pct:.1f}%) image paths resolve")
        ok_paths = False
    else:
        print(f"  [OK]      {d}: {s}/{c} ({pct:.1f}%) resolve")

# ---------------------------------------------------------------- summary
banner("Summary")
if ok_disk and ok_paths:
    print("  [SUCCESS] Corpus looks ready for training.")
    sys.exit(0)
else:
    print("  [FAIL] Corpus has issues.")
    if not ok_disk:
        print("    - Some processed_faces folders are missing or near-empty.")
    if not ok_paths:
        print("    - Some image paths in the CSV do not resolve.")
        print("    - Run patch_csv.py to substitute orig_image for the broken rows.")
    sys.exit(1)
