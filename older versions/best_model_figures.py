import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from Dataset import RNADataset, variable_length_collate_fn
from X_Network import RNADegformer
from visualization import (
    plot_all_heads_in_layer,
    plot_bpp_matrix,
    plot_attention_with_structure_overlay,
    plot_all_heads_in_layer_with_bpp,
    plot_attention_with_bpp
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='best', choices=['best', 'last'],
                    help="Which model checkpoint to visualize: 'best' or 'last'")
parser.add_argument('--head', type=int, default=0, help="Which attention head to visualize")
args = parser.parse_args()

# Configuration
if args.model == 'best':
    MODEL_PATH = "../pretrain_weights/best_model.ckpt"
else:
    checkpoint_dir = "../pretrain_weights"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch") and f.endswith(".ckpt")]
    if not checkpoints:
        raise FileNotFoundError("No epoch checkpoints found.")
    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace("epoch", "").replace(".ckpt", "")))
    MODEL_PATH = os.path.join(checkpoint_dir, checkpoints[-1])

DATA_PATH = "../data/kaggle/test.json"
BPP_PATH = "../data/kaggle/new_sequences_bpps/"
SAVE_PATH = "../visualizations_best_model_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
NINPUT = 256
NHEAD = 8
NLAYERS = 5
NHID = 1024

print("Loading fixed-length dataset for visualization...")
json = pd.read_json(DATA_PATH, lines=True)

dataset = RNADataset(
    json.sequence.tolist(),
    np.zeros(len(json)),
    json.id.tolist(),
    list(range(len(json))),
    BPP_PATH,
    pad=True,
    k=5
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=variable_length_collate_fn)

model = RNADegformer(
    ntoken=21, nclass=5, ninp=NINPUT, nhead=NHEAD, nhid=NHID, nlayers=NLAYERS,
    kmer_aggregation=True, kmers=[1], stride=1, dropout=0.1,
    pretrain=True, rinalmo_weights_path="/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_micro_pretrained.pt"
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

os.makedirs(SAVE_PATH, exist_ok=True)

print("Running best model forward pass...")
print("Scanning for structurally diverse sequence...")
max_unique = 0
best_data = None
for batch in dataloader:
    loop_col = batch['data'][:, :, 2]  # [B, L]
    for i in range(loop_col.size(0)):
        unique_labels = torch.unique(loop_col[i])
        valid_labels = unique_labels[unique_labels != 6]  # exclude X
        if len(valid_labels) > max_unique:
            max_unique = len(unique_labels)
            best_data = {k: (v[i] if isinstance(v, torch.Tensor) else v[i]) for k, v in batch.items()}
print(f"Best sequence found with {max_unique} unique loop types.")


rinalmo_emb = best_data['embedding'].unsqueeze(0).to(DEVICE)  # [1, L, D]
bpps = best_data['bpp'].unsqueeze(0).to(DEVICE)               # [1, C, L, L]
src_mask = best_data['src_mask'].unsqueeze(0).to(DEVICE)      # [1, num_layers, L]

loop_labels = best_data['data'][:, 2].unsqueeze(0).to(DEVICE)  # shape: [1, L]

print(torch.unique(loop_labels[0]))

with torch.no_grad():
    outputs, attention_maps = model(rinalmo_emb, bpps, src_mask)

print("Plotting attention maps...")

plot_all_heads_in_layer_with_bpp(
    attention_maps=attention_maps,
    bpp_targets=bpps[:, 0],
    layer_idx=0,
    seq_idx=0,
    threshold=0.5,
    save_path=SAVE_PATH
)

plot_attention_with_bpp(
    attention_maps=attention_maps,
    bpp_targets=bpps[:, 0],
    layer_idx=0,
    head_idx=args.head,
    seq_idx=0,
    threshold=0.5,
    save_path=SAVE_PATH
)

plot_attention_with_structure_overlay(
    attention_maps=attention_maps,
    loop_labels=loop_labels,
    layer_idx=0,
    head_idx=args.head,
    seq_idx=0,
    save_path=os.path.join(SAVE_PATH, "attn_struct")
)

plot_all_heads_in_layer(
    attention_maps=attention_maps,
    layer_idx=0,
    seq_idx=0,
    save_path=os.path.join(SAVE_PATH, "attn_only")
)

plot_bpp_matrix(
    bpp_targets=bpps,
    seq_idx=0,
    save_path=os.path.join(SAVE_PATH, "bpp_only")
)

print(f"[Visualization] Attention maps saved under {SAVE_PATH}/")
