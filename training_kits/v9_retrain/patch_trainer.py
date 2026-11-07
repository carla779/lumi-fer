"""
patch_trainer.py — add diagnostic prints to cnn_train_multidataset_fer7_soft_au.py
so that silent row-drops at the os.path.isfile filter become visible during training.

Idempotent: detects a marker phrase to avoid double-patching.
"""
import os
import sys

TRAINER = r"C:\Users\super\PyCharmMiscProject\cnn_train_multidataset_fer7_soft_au.py"
MARKER = "[loader] rows dropped due to missing image files"

# Anchor: the existing line that filters rows by os.path.isfile. The exact
# 4-space indentation must match what's in the file (verified via grep).
ANCHOR = '    df = df[df["image"].apply(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)'

REPLACEMENT = (
    '    n_before_filter = len(df)\n'
    + ANCHOR + '\n'
    + '    n_dropped = n_before_filter - len(df)\n'
    + '    print(f"[loader] {n_dropped} rows dropped due to missing image files (kept {len(df)})")\n'
    + '    print(f"[loader] per-dataset rows kept: {df[\'dataset\'].astype(str).str.lower().value_counts().to_dict()}")'
)


def main():
    if not os.path.isfile(TRAINER):
        print(f"[ERROR] Trainer not found: {TRAINER}")
        sys.exit(1)

    with open(TRAINER, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("[OK] Trainer already patched (marker phrase found); no changes made.")
        sys.exit(0)

    if ANCHOR not in src:
        print("[ERROR] Could not locate the os.path.isfile filter line in the trainer.")
        print("        Expected line:")
        print(f"        {ANCHOR}")
        print("        The trainer may have been edited by hand. Skipping patch.")
        sys.exit(1)

    # Backup first
    backup = TRAINER + ".bak"
    if not os.path.isfile(backup):
        with open(backup, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[OK] Backup written to {backup}")

    new_src = src.replace(ANCHOR, REPLACEMENT, 1)
    with open(TRAINER, "w", encoding="utf-8") as f:
        f.write(new_src)
    print("[OK] Trainer patched: silent row drops will now print during training.")


if __name__ == "__main__":
    main()
