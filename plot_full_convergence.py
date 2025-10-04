import matplotlib.pyplot as plt
import pandas as pd
import os

# ---- Paths to your logs ----
pretrain_log = 'logs/pretrain.csv'
train_log = './log_fold0.csv'
pl_log = 'logs/log_pl_fold0.csv'

# ---- Load each CSV ----
pretrain_df = pd.read_csv(pretrain_log)
train_df = pd.read_csv(train_log)
pl_df = pd.read_csv(pl_log)

# ---- Collapse Pretrain Loss ----
# Group multiple validation records per epoch
pretrain_df_grouped = pretrain_df.groupby('epoch').agg({'val_loss': 'min'}).reset_index()
pretrain_epochs = pretrain_df_grouped['epoch']
pretrain_val_loss = pretrain_df_grouped['val_loss']

# ---- Align Epochs ----
train_epochs = train_df['epoch'] + pretrain_epochs.max()
pl_epochs = pl_df['epoch'] + train_epochs.max()

# ---- Plot ----
plt.figure(figsize=(10, 6))

# Pretrain phase
plt.plot(pretrain_epochs, pretrain_val_loss, label='Pretrain Validation Loss', color='blue', linewidth=2)

# Train phase
plt.plot(train_epochs, train_df['val_loss'], label='Train Validation Loss', color='green', linewidth=2)

# PL phase
plt.plot(pl_epochs, pl_df['val_loss'], label='PL Validation Loss', color='red', linewidth=2)

# Styling
plt.xlabel('Training Epoch')
plt.ylabel('Validation Loss')
plt.title('Full Training Convergence: Pretrain ➔ Supervised ➔ Semi-Supervised')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save
os.makedirs('visualizations_best_model', exist_ok=True)
plt.savefig('visualizations_best_model/full_training_convergence_FINAL.png')
plt.close()

print("[Visualization] Full convergence plot saved as 'visualizations_best_model/full_training_convergence_FINAL.png'")
