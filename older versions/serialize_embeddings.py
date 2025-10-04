# save as: scripts/serialize_embeddings.py
import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
from rinalmo.data.constants import RNA_TOKENS
from rinalmo.data.alphabet import Alphabet
from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo

# Global constants
rinalmo_weights_path = "/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_micro_pretrained.pt"
embedding_out_path = "data/precomputed_embeddings.pt"
seq_ids_path = "data/test.json"  # assumes same sequences as train.json

# Load sequences
import pandas as pd
df = pd.read_json(seq_ids_path, lines=True)
seqs = df.sequence.tolist()
ids = df.id.tolist()

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alphabet = Alphabet(standard_tkns=RNA_TOKENS)
config = model_config("micro")
model = RiNALMo(config)
model.load_state_dict(torch.load(rinalmo_weights_path, map_location="cpu"))
model.to(device).eval()

# Allocate output tensor
fixed_len = 130
embedding_dim = 480
N = len(seqs)

all_embeddings = torch.zeros((N, fixed_len, embedding_dim), dtype=torch.float32)

print(f"Generating embeddings for {N} sequences...")

for idx, seq in enumerate(tqdm(seqs)):
    seq = seq.replace('U', 'T')
    token_ids = torch.tensor([alphabet.tkn_to_idx.get(ch, alphabet.tkn_to_idx["<unk>"]) for ch in seq])
    input_tensor = token_ids.unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
        output = model(input_tensor)["representation"].squeeze(0).cpu()

    if output.size(0) < fixed_len:
        pad = torch.zeros((fixed_len - output.size(0), embedding_dim))
        output = torch.cat([output, pad], dim=0)
    elif output.size(0) > fixed_len:
        output = output[:fixed_len]

    all_embeddings[idx] = output

torch.save({"ids": ids, "embeddings": all_embeddings}, embedding_out_path)
print(f"Saved embeddings: {embedding_out_path}")
