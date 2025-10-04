import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_pretrain_loss_curve(log_file_path='runs/20250427_090322/logs/logs/pretrain.csv', save_path='pretrain_loss_curve_fixed.png'):
    """
    Load pretrain CSV log, de-duplicate by epoch, and plot training/validation loss curves.

    Args:
        log_file_path (str): Path to the pretrain CSV log.
        save_path (str): Path to save the resulting plot.
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    # Load the log file
    df = pd.read_csv(log_file_path)

    # Sanity check
    expected_cols = ['epoch', 'train_loss', 'val_loss']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in log file.")

    # Group by epoch and average if duplicate epochs exist
    df_grouped = df.groupby('epoch', as_index=False).mean()

    # Sort explicitly
    df_grouped = df_grouped.sort_values('epoch')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped['epoch'], df_grouped['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(df_grouped['epoch'], df_grouped['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretraining Loss Curve (Fixed)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save
    plt.savefig(save_path, dpi=300)
    print(f"Fixed loss curve saved to {save_path}")

    plt.show()

# Example usage:
if __name__ == '__main__':
    plot_pretrain_loss_curve()
