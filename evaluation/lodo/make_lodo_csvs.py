"""
Build four leave-one-dataset-out (LODO) training CSVs from the unified CSV.
For each held-out dataset, the rows belonging to that dataset become the
test split; rows from the other three datasets are kept with their original
train/val splits but their original test rows are folded into train.

Usage on the gaming PC:
    python make_lodo_csvs.py --in unified_fer7_processed_au_FRIEND.csv

Outputs (next to the input CSV):
    lodo_HOLDOUT_rafdb.csv
    lodo_HOLDOUT_ferplus.csv
    lodo_HOLDOUT_expw.csv
    lodo_HOLDOUT_fer2013.csv
"""
import argparse
import os
import pandas as pd

DATASETS = ["rafdb", "ferplus", "expw", "fer2013"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, low_memory=False)
    if "dataset" not in df.columns:
        raise RuntimeError("input CSV is missing a 'dataset' column")
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["split"]   = df["split"].astype(str).str.lower()

    out_dir = os.path.dirname(os.path.abspath(args.in_csv)) or "."
    print(f"Loaded {len(df)} rows from {args.in_csv}")
    print(f"Datasets present: {sorted(df['dataset'].unique())}")
    print(f"Original split counts: {df['split'].value_counts().to_dict()}")

    for held_out in DATASETS:
        if held_out not in df["dataset"].unique():
            print(f"[SKIP] {held_out} not in CSV")
            continue
        new = df.copy()
        # Held-out rows: all become test
        held_mask = new["dataset"] == held_out
        new.loc[held_mask, "split"] = "test"
        # Other datasets: their existing test rows fold into train so the model
        # gets the maximum possible training data from those datasets
        other_mask = ~held_mask
        new.loc[other_mask & (new["split"] == "test"), "split"] = "train"

        out_path = os.path.join(out_dir, f"lodo_HOLDOUT_{held_out}.csv")
        new.to_csv(out_path, index=False)
        n_train = int((new["split"] == "train").sum())
        n_val   = int((new["split"] == "val").sum())
        n_test  = int((new["split"] == "test").sum())
        print(f"[OK] {out_path}  (train={n_train}, val={n_val}, test={n_test})")


if __name__ == "__main__":
    main()
