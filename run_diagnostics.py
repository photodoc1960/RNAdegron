import pandas as pd
import os
import numpy as np
import glob

print("=== CHECKING CSV LOG FILES ===")
print("=" * 50)

csv_files_found = 0
for fold in range(5):
    csv_file = f"logs/log_pl_fold{fold}.csv"
    print(f"\nFold {fold}: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"  ❌ File does not exist!")
        continue
    
    csv_files_found += 1
    try:
        # Check file size
        file_size = os.path.getsize(csv_file)
        print(f"  📁 File size: {file_size} bytes")
        
        if file_size == 0:
            print(f"  ❌ File is empty!")
            continue
            
        # Read and examine the CSV
        df = pd.read_csv(csv_file)
        print(f"  📊 Total rows: {len(df)}")
        print(f"  📊 Columns: {list(df.columns)}")
        
        if 'val_loss' not in df.columns:
            print(f"  ❌ No 'val_loss' column found!")
            continue
            
        # Check validation losses
        val_losses = df['val_loss']
        non_null_vals = val_losses.dropna()
        
        print(f"  📊 Non-null val_loss entries: {len(non_null_vals)}")
        print(f"  📊 Null val_loss entries: {len(val_losses) - len(non_null_vals)}")
        
        if len(non_null_vals) > 0:
            print(f"  📊 Val_loss range: {non_null_vals.min():.6f} - {non_null_vals.max():.6f}")
            print(f"  📊 Best val_loss: {non_null_vals.min():.6f}")
            
            # Show last few entries
            print(f"  📊 Last 5 entries:")
            for i, (idx, row) in enumerate(df.tail().iterrows()):
                epoch = row.get('epoch', 'N/A')
                train_loss = row.get('train_loss', 'N/A')
                val_loss = row.get('val_loss', 'N/A')
                print(f"    {i+1}. Epoch: {epoch}, Train: {train_loss}, Val: {val_loss}")
                
            # Test the get_best_weights logic on this file
            print(f"  🔍 Testing get_best_weights logic:")
            valid_scores = df[df.val_loss.notna() & (df.val_loss != -1)]
            if len(valid_scores) > 0:
                scores = -valid_scores.val_loss.values
                best_score = scores.max() if len(scores) > 0 else 0
                print(f"    Best score: {best_score:.6f}")
                best_idx = scores.argmax() if len(scores) > 0 else 0
                best_epoch = valid_scores.iloc[best_idx].epoch
                print(f"    Best epoch: {best_epoch}")
            else:
                print(f"    ❌ No valid scores found!")
        else:
            print(f"  ❌ No valid validation losses found!")
            
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")

print(f"\n📊 Summary: Found {csv_files_found}/5 CSV files")

print("\n=== CHECKING CHECKPOINT DIRECTORIES ===")
print("=" * 50)

checkpoint_dirs_found = 0
for fold in range(5):
    weights_path = f"weights/checkpoints_fold{fold}_pl"
    print(f"\nFold {fold}: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"  ❌ Directory does not exist!")
        continue
    
    checkpoint_dirs_found += 1
    # List all checkpoint files
    checkpoint_files = glob.glob(f"{weights_path}/*.ckpt")
    print(f"  📁 Found {len(checkpoint_files)} checkpoint files")
    
    if checkpoint_files:
        for ckpt in sorted(checkpoint_files)[:3]:  # Show first 3
            filename = os.path.basename(ckpt)
            size = os.path.getsize(ckpt)
            print(f"    - {filename} ({size:,} bytes)")
        if len(checkpoint_files) > 3:
            print(f"    ... and {len(checkpoint_files) - 3} more")
    else:
        print(f"  ❌ No checkpoint files found!")

print(f"\n📊 Summary: Found {checkpoint_dirs_found}/5 checkpoint directories")

print("\n=== TESTING get_best_weights_v6.py LOGIC ===")
print("=" * 50)

# Replicate the exact logic from get_best_weights_v6.py
def get_best_weights_from_fold_debug(fold, csv_file, weights_path, des, top=1):
    print(f"\nProcessing fold {fold}:")
    print(f"  CSV file: {csv_file}")
    print(f"  Weights path: {weights_path}")
    
    if not os.path.exists(csv_file):
        print(f"  ❌ CSV file does not exist!")
        return 0.0
        
    try:
        history = pd.read_csv(csv_file)
        print(f"  📊 CSV has {len(history)} rows")
        
        valid_scores = history[history.val_loss.notna() & (history.val_loss != -1)]
        print(f"  📊 Valid scores: {len(valid_scores)} rows")
        
        if len(valid_scores) == 0:
            print(f"  ❌ No valid scores found!")
            return 0.0
            
        scores = -valid_scores.val_loss.values
        print(f"  📊 Scores array: {scores}")
        
        if len(scores) == 0:
            print(f"  ❌ Empty scores array!")
            return 0.0
            
        top_indices = scores.argsort()[-top:][::-1]
        best_score = scores[top_indices[0]]
        print(f"  ✅ Best score: {best_score}")
        
        return best_score
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0.0

scores = []
for i in range(5):
    score = get_best_weights_from_fold_debug(
        i, 
        csv_file=f"logs/log_pl_fold{i}.csv",
        weights_path=f"weights/checkpoints_fold{i}_pl",
        des='best_pl_weights'
    )
    scores.append(score)

print(f"\nFinal scores: {scores}")
print(f"Mean CV score: {-np.mean(scores)}")
