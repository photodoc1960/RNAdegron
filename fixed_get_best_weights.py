import numpy as np
import os
import pandas as pd

def get_best_weights_from_fold(fold, csv_file, weights_path, des, top=1):
    print(f"\nProcessing fold {fold}:")
    print(f"  Reading: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"  ❌ CSV file does not exist!")
        return 0.0
        
    history = pd.read_csv(csv_file)
    print(f"  📊 Total rows in CSV: {len(history)}")
    
    # Filter for valid validation losses (positive values only, exclude placeholders)
    valid_scores = history[
        history.val_loss.notna() & 
        (history.val_loss > 0) &  # Only positive validation losses
        (history.val_loss != -1)  # Exclude -1 placeholders
    ]
    
    print(f"  📊 Valid validation scores: {len(valid_scores)}")
    
    if len(valid_scores) == 0:
        print(f"  ❌ No valid validation scores found!")
        return 0.0
        
    # Show validation loss range
    val_range = f"{valid_scores.val_loss.min():.6f} - {valid_scores.val_loss.max():.6f}"
    print(f"  📊 Validation loss range: {val_range}")
    
    # Find the minimum validation loss (best performance)
    best_val_loss = valid_scores.val_loss.min()
    best_epoch = valid_scores.loc[valid_scores.val_loss.idxmin(), 'epoch']
    
    print(f"  ✅ Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Copy the best checkpoint
    os.makedirs(des, exist_ok=True)
    checkpoint_path = f"{weights_path}/epoch{int(best_epoch)}.ckpt"
    
    if os.path.exists(checkpoint_path):
        dest_path = f"{des}/fold{fold}top1.ckpt"
        os.system(f"cp {checkpoint_path} {dest_path}")
        print(f"  📁 Copied {checkpoint_path} to {dest_path}")
    else:
        print(f"  ⚠️ Checkpoint file not found: {checkpoint_path}")
    
    # Return negative of validation loss (for compatibility with original scoring)
    return -best_val_loss

print("🚀 Finding best weights for each fold...")
print("=" * 60)

scores = []
for i in range(5):
    score = get_best_weights_from_fold(
        i, 
        csv_file=f"logs/log_pl_fold{i}.csv",
        weights_path=f"weights/checkpoints_fold{i}_pl",
        des='best_pl_weights'
    )
    scores.append(score)

print("\n" + "=" * 60)
print("📊 FINAL RESULTS:")
print(f"Individual fold scores: {[f'{s:.6f}' for s in scores]}")
print(f"Mean CV score: {-np.mean(scores):.6f}")

# Write CV score to file
with open('cv.txt', 'w+') as f:
    f.write(str(-np.mean(scores)))
    
print(f"✅ CV score written to cv.txt: {-np.mean(scores):.6f}")
