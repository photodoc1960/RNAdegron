import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_folds_loss_curves(log_folder='.', save_path='all_folds_loss_curves.png', n_folds=5):
    """
    Plot multipanel train/validation loss curves for multiple folds.

    Args:
        log_folder (str): Folder containing log_fold{i}.csv files.
        save_path (str): Path to save the resulting multipanel figure.
        n_folds (int): Number of folds to attempt to plot.
    """
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 5), sharey=True)

    if n_folds == 1:
        axes = [axes]  # make it iterable

    for i in range(n_folds):
        log_file = os.path.join(log_folder, f'log_fold{i}.csv')
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found, skipping fold {i}")
            continue

        df = pd.read_csv(log_file)

        # Group by epoch to handle duplicates, average losses
        df_grouped = df.groupby('epoch', as_index=False).mean()
        df_grouped = df_grouped.sort_values('epoch')

        ax = axes[i]
        ax.plot(df_grouped['epoch'], df_grouped['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in df_grouped.columns and not df_grouped['val_loss'].isnull().all():
            ax.plot(df_grouped['epoch'], df_grouped['val_loss'], label='Validation Loss', linewidth=2)

        ax.set_title(f'Fold {i}')
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()

    plt.suptitle('Training and Validation Loss Curves Across Folds', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.savefig(save_path, dpi=300)
    print(f"Multipanel loss curves saved to {save_path}")
    plt.show()

# Example usage
if __name__ == '__main__':
    plot_all_folds_loss_curves(log_folder='.', n_folds=5)
