import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

from rinalmo.data.constants import RNA_TOKENS
from rinalmo.data.alphabet import Alphabet
from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo

# Paths
rinalmo_weights_path = "/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_mega_pretrained.pt"
embedding_out_path = "data/precomputed_features.pt"
# Use the combined JSON of *all* sequences (train+test) for embedding serialization
seq_ids_path = "data/pretrain_all.json"

# Load sequences and structures
print(f"Loading sequence data from {seq_ids_path}...")
df = pd.read_json(seq_ids_path, lines=True)
seqs = df.sequence.tolist()
structures = df.structure.tolist()
ids = df.id.tolist()
N = len(seqs)

# Determine maximum sequence length across all entries
max_len = max(len(seq) for seq in seqs)
print(f"Max sequence length: {max_len}")


# Helpers for structure-based features

def paired_positions(structure: str):
    stack, pairs = [], []
    for i, ch in enumerate(structure):
        if ch == '(':
            stack.append(i)
        elif ch == ')' and stack:
            j = stack.pop()
            pairs.append((j, i))
    return pairs


def compute_graph_distances(structure: str) -> np.ndarray:
    L = len(structure)
    G = nx.Graph()
    G.add_nodes_from(range(L))
    # linear connections
    for i in range(L - 1):
        G.add_edge(i, i + 1)
    # base-pair connections
    for i, j in paired_positions(structure):
        G.add_edge(i, j)
    # compute shortest paths
    dist = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        lengths = nx.single_source_shortest_path_length(G, i)
        for j, d in lengths.items():
            dist[i, j] = d
    return dist


def compute_nearest_distances(structure: str):
    L = len(structure)
    pairs = paired_positions(structure)
    paired_idxs = {i for i, _ in pairs} | {j for _, j in pairs}
    unpaired = [i for i in range(L) if i not in paired_idxs]
    nearest_p = np.full(L, L, dtype=np.int32)
    nearest_up = np.full(L, L, dtype=np.int32)
    for i in range(L):
        if paired_idxs:
            nearest_p[i] = min(abs(i - j) for j in paired_idxs)
        if unpaired:
            nearest_up[i] = min(abs(i - j) for j in unpaired)
    return nearest_p, nearest_up


# Initialize RiNALMo model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alphabet = Alphabet(standard_tkns=RNA_TOKENS)
config = model_config("mega")
model = RiNALMo(config)

# Check what precision the model parameters are in
model_param_dtype = next(model.parameters()).dtype
print(f"Model parameter dtype: {model_param_dtype}")

# ENHANCED: Explicit precision handling for mega model
state_dict = torch.load(rinalmo_weights_path, map_location="cpu")
# Ensure model and state dict precision compatibility
if next(iter(state_dict.values())).dtype != model_param_dtype:
    print(f"Converting state dict from {next(iter(state_dict.values())).dtype} to {model_param_dtype}")
    state_dict = {k: v.to(model_param_dtype) if v.is_floating_point() else v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.to(device).eval()

# CRITICAL FIX: Determine embedding dimension with proper precision handling
print("Determining actual RiNALMo embedding dimensions with precision compatibility...")
with torch.no_grad():
    # Create test input tensor
    test_token_ids = torch.tensor([alphabet.tkn_to_idx.get('A', alphabet.tkn_to_idx['<unk>'])], dtype=torch.long)
    test_seq = test_token_ids.unsqueeze(0).to(device)

    # CRITICAL: Use autocast context to handle precision automatically
    # This allows RiNALMo to use its expected precision internally while we work with standard tensors
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        try:
            test_output = model(test_seq)["representation"]
            actual_embedding_dim = test_output.shape[-1]
            print(f"✅ Successfully determined RiNALMo embedding dimension: {actual_embedding_dim}")
        except Exception as e:
            print(f"❌ Error with float16 autocast: {e}")
            print("Trying with bfloat16...")
            # Fallback to bfloat16 if float16 fails
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                test_output = model(test_seq)["representation"]
                actual_embedding_dim = test_output.shape[-1]
                print(f"✅ Successfully determined RiNALMo embedding dimension with bfloat16: {actual_embedding_dim}")

# Use actual dimension instead of hardcoded 480
embedding_dim = actual_embedding_dim
dist_dim = max_len
all_embeddings = torch.zeros((N, dist_dim, embedding_dim), dtype=torch.float32)

# Allocate storage for features

all_graph_dists = np.zeros((N, dist_dim, dist_dim), dtype=np.float32)
all_nearest_p = np.zeros((N, dist_dim), dtype=np.int32)
all_nearest_up = np.zeros((N, dist_dim), dtype=np.int32)

print(f"Generating features for {N} sequences...")
for idx, (seq, struct) in enumerate(tqdm(zip(seqs, structures), total=N)):
    # --- Embedding Generation with Precision Handling ---
    # CRITICAL FIX: Standardize on T-based representation throughout pipeline
    seq_tokens = seq.replace('U', 'T')  # Standardize to T-based representation
    token_ids = torch.tensor([alphabet.tkn_to_idx.get(ch, alphabet.tkn_to_idx['<unk>']) for ch in seq_tokens],
                             dtype=torch.long)
    input_tensor = token_ids.unsqueeze(0).to(device)

    # CRITICAL: Use same autocast context as dimension detection
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            try:
                rep = model(input_tensor)["representation"].squeeze(0).cpu().float()  # Convert back to float32
            except:
                # Fallback to bfloat16 if float16 fails
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    rep = model(input_tensor)["representation"].squeeze(0).cpu().float()  # Convert back to float32

    L = rep.size(0)
    # pad or truncate to dist_dim
    if L < dist_dim:
        pad = torch.zeros((dist_dim - L, embedding_dim), dtype=rep.dtype)
        rep = torch.cat([rep, pad], dim=0)
    else:
        rep = rep[:dist_dim]
    all_embeddings[idx] = rep

    # --- Graph distances ---
    gd = compute_graph_distances(struct)
    Lg = gd.shape[0]
    if Lg < dist_dim:
        gd_padded = np.zeros((dist_dim, dist_dim), dtype=np.float32)
        gd_padded[:Lg, :Lg] = gd
    else:
        gd_padded = gd[:dist_dim, :dist_dim]
    all_graph_dists[idx] = gd_padded

    # --- Nearest distances ---
    npd, nud = compute_nearest_distances(struct)
    # pad to dist_dim
    npd_pad = np.zeros(dist_dim, dtype=np.int32)
    nud_pad = np.zeros(dist_dim, dtype=np.int32)
    npd_pad[:len(npd)] = npd
    nud_pad[:len(nud)] = nud
    all_nearest_p[idx] = npd_pad
    all_nearest_up[idx] = nud_pad

# Save features to disk
torch.save({
    'ids': ids,
    'embeddings': all_embeddings,
    'graph_dists': all_graph_dists,
    'nearest_paired': all_nearest_p,
    'nearest_unpaired': all_nearest_up,
    'embedding_dim': embedding_dim,  # Save actual dimension for validation
    'max_seq_len': max_len,
    'model_precision': str(model_param_dtype)  # Save precision info for debugging
}, embedding_out_path)
print(f"✅ Saved precomputed features to {embedding_out_path}")
print(f"📊 Embedding dimensions: {all_embeddings.shape}")
print(f"🔢 Actual embedding dim used: {embedding_dim}")
print(f"⚡ Model precision: {model_param_dtype}")