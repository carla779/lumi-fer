"""
Strip the soft-label probability columns from the unified training CSV so that
the same training script falls back to one-hot labels.

Usage:
    python make_hardlabel_csv.py --in unified_fer7_processed_au_FRIEND.csv \
                                 --out unified_fer7_processed_au_HARDLABELS.csv
"""
import argparse
import pandas as pd

P_COLS = ["p_angry", "p_disgust", "p_fear",
          "p_happy", "p_sad", "p_surprise", "p_neutral"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_csv",  required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, low_memory=False)
    drop = [c for c in P_COLS if c in df.columns]
    print(f"Dropping {len(drop)} soft-label columns: {drop}")
    df = df.drop(columns=drop)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] saved {args.out_csv}  ({len(df)} rows, {len(df.columns)} columns)")

if __name__ == "__main__":
    main()
