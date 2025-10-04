import pandas as pd, glob

for csv_path in glob.glob('logs/log_pl_fold*.csv'):
    df = pd.read_csv(csv_path)
    # find the max numeric epoch
    numeric_epochs = pd.to_numeric(df['epoch'], errors='coerce')
    last_epoch = int(numeric_epochs.max())
    # replace 'final' (or any non-numeric) with last_epoch
    df.loc[numeric_epochs.isna(), 'epoch'] = last_epoch
    df.to_csv(csv_path, index=False)
    print(f"Patched {csv_path}: set 'final' → {last_epoch}")
