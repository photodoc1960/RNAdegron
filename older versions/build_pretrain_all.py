import pandas as pd
import os

# Paths to known files
base_path = "../data"
train_path = os.path.join(base_path, "train.json")
print(train_path)
test_path = os.path.join(base_path, "test.json")
new_seq_path = os.path.join(base_path, "post_deadline_files", "new_sequences.csv")

# Load train.json
train_df = pd.read_json(train_path, lines=True)
print(f"Loaded train.json: {len(train_df)} sequences")

# Load test.json if present
if os.path.exists(test_path):
    test_df = pd.read_json(test_path, lines=True)
    print(f"Loaded test.json: {len(test_df)} sequences")
else:
    test_df = pd.DataFrame()

# Load new_sequences.csv if present
if os.path.exists(new_seq_path):
    new_df = pd.read_csv(new_seq_path)
    print(f"Loaded new_sequences.csv: {len(new_df)} sequences")

    # Rename to match train/test column structure
    new_df = new_df.rename(columns={
        "bpRNA_string": "predicted_loop_type"
    })
    new_df["id"] = new_df["id"].astype(str)

    # Keep only expected columns
    expected_cols = ["id", "sequence", "structure", "predicted_loop_type"]
    new_df = new_df[[c for c in expected_cols if c in new_df.columns]]
else:
    new_df = pd.DataFrame()


combined = pd.concat([train_df, test_df], ignore_index=True)

print(f"Total sequences: {len(combined)}")

# Output
out_path = os.path.join(base_path, "pretrain_all.json")
combined.to_json(out_path, orient="records", lines=True)
print(f"Saved to {out_path}")
