import os
import pandas as pd
import shutil

for fold in range(5):
    log_path = f"log_fold{fold}.csv"
    ckpt_dir = f"checkpoints_fold{fold}"
    best_ckpt_path = os.path.join(ckpt_dir, "best_model.ckpt")

    if not os.path.exists(log_path):
        print(f"[Fold {fold}] Log file not found: {log_path}")
        continue

    df = pd.read_csv(log_path)
    df = df.dropna()
    if "val_loss" not in df.columns:
        print(f"[Fold {fold}] No val_loss column in log.")
        continue

    best_row = df.loc[df["val_loss"].idxmin()]
    best_epoch = int(best_row["epoch"]) - 1  # zero-based index for epochN.ckpt

    src_ckpt = os.path.join(ckpt_dir, f"epoch{best_epoch}.ckpt")
    if not os.path.exists(src_ckpt):
        print(f"[Fold {fold}] Checkpoint not found: {src_ckpt}")
        continue

    shutil.copy(src_ckpt, best_ckpt_path)
    print(f"[Fold {fold}] Promoted epoch{best_epoch}.ckpt → best_model.ckpt")
