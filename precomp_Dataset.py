import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from Functions import *
import matplotlib.pyplot as plt
import RNA

tokens = ['<cls>', '<pad>', '<eos>', '<unk>', '<mask>',
          'A', 'C', 'G', 'T', 'I', 'R', 'Y', 'K', 'M', 'S', 'W',
          'B', 'D', 'H', 'V', 'N', '-', 'E', 'X']

#eterna,'nupack','rnastructure','vienna_2','contrafold_2',

# sliding_window.py
import numpy as np
import torch

import torch

def variable_length_collate_fn(batch):
    batch_size = len(batch)
    max_seq_len = max(sample['data'].shape[0] for sample in batch)

    padded_data = torch.full((batch_size, max_seq_len, 3), fill_value=14, dtype=torch.long)
    padded_bpp = torch.zeros((batch_size, 4, max_seq_len, max_seq_len), dtype=torch.float32)
    padded_src_mask = torch.zeros((batch_size, batch[0]['src_mask'].shape[0], max_seq_len), dtype=torch.int8)

    labels = torch.zeros(batch_size, dtype=torch.float32)
    ews = torch.zeros(batch_size, dtype=torch.float32)
    ids = []

    for i, sample in enumerate(batch):
        seq_len = sample['data'].shape[0]

        padded_data[i, :seq_len, :] = torch.from_numpy(sample['data'])

        # Explicit padding/truncation for bpp matrices
        bpp_len = sample['bpp'].shape[-1]
        effective_len = min(seq_len, bpp_len, max_seq_len)
        padded_bpp[i, :, :effective_len, :effective_len] = torch.from_numpy(sample['bpp'][:, :effective_len, :effective_len])

        # src_mask explicitly padded/truncated
        mask_len = sample['src_mask'].shape[-1]
        padded_src_mask[i, :, :min(mask_len, max_seq_len)] = torch.from_numpy(sample['src_mask'][:, :min(mask_len, max_seq_len)])

        label_tensor = torch.tensor(sample['labels'], dtype=torch.float32)
        if i == 0:
            labels = torch.zeros((batch_size, *label_tensor.shape), dtype=torch.float32)
        labels[i] = label_tensor

        ew_tensor = torch.tensor(sample['ew'], dtype=torch.float32)
        if i == 0:
            ews = torch.zeros((batch_size, *ew_tensor.shape), dtype=torch.float32)
        ews[i] = ew_tensor

        ids.append(sample['id'])

    return {
        'data': padded_data,
        'bpp': padded_bpp,
        'src_mask': padded_src_mask,
        'labels': labels,
        'ew': ews,
        'id': ids
    }

def process_long_sequence(models, sequence, bpp, window_size=130, stride=100):
    """Process a long sequence using a sliding window approach."""
    seq_len = len(sequence)
    device = next(models[0].parameters()).device

    # For sequences that fit within model capacity
    if seq_len <= window_size:
        # Convert to tensors
        seq_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
        bpp_tensor = torch.tensor(bpp).unsqueeze(0).to(device)

        # Generate predictions from all models
        outputs = []
        for model in models:
            with torch.no_grad():
                output = model(seq_tensor, bpp_tensor)
            outputs.append(output.cpu().numpy())

        # Average predictions
        avg_output = np.mean(outputs, axis=0)
        return avg_output

    # For longer sequences, use sliding window
    all_outputs = []
    positions = []

    for start_idx in range(0, seq_len - window_size + 1, stride):
        end_idx = start_idx + window_size

        # Extract window
        window_seq = sequence[start_idx:end_idx]
        window_bpp = bpp[start_idx:end_idx, start_idx:end_idx]

        # Convert to tensors
        seq_tensor = torch.tensor(window_seq).unsqueeze(0).to(device)
        bpp_tensor = torch.tensor(window_bpp).unsqueeze(0).to(device)

        # Generate predictions from all models
        window_outputs = []
        for model in models:
            with torch.no_grad():
                output = model(seq_tensor, bpp_tensor)
            window_outputs.append(output.cpu().numpy())

        # Average predictions for this window
        avg_output = np.mean(window_outputs, axis=0)
        all_outputs.append(avg_output)
        positions.append((start_idx, end_idx))

    # Merge predictions from all windows
    merged_output = np.zeros((seq_len, avg_output.shape[1]))  # Assumes output dim is consistent
    counts = np.zeros(seq_len)

    for (start_idx, end_idx), output in zip(positions, all_outputs):
        merged_output[start_idx:end_idx] += output
        counts[start_idx:end_idx] += 1

    # Average overlapping regions
    merged_output = merged_output / np.maximum(counts.reshape(-1, 1), 1)

    return merged_output


def generate_bpp_matrix(sequence):
    """Generate base pairing probability matrix using ViennaRNA."""
    # Ensure sequence uses 'U' instead of 'T'
    sequence = sequence.replace('T', 'U')

    # Create fold compound and calculate partition function
    fc = RNA.fold_compound(sequence)
    fc.pf()

    # Get the complete BPP matrix - specific API depends on ViennaRNA version
    L = len(sequence)
    bpp_matrix = np.zeros((L, L))

    # Most recent ViennaRNA versions use this method
    probs = fc.bpp()

    # Convert to matrix format
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            if (i, j) in probs:
                bpp_matrix[i - 1, j - 1] = probs[(i, j)]
                bpp_matrix[j - 1, i - 1] = probs[(i, j)]  # Make symmetric

    return bpp_matrix


def generate_structure_and_loop(sequence):
    """Generate secondary structure and loop types using ViennaRNA."""
    # Ensure sequence uses 'U' instead of 'T'
    sequence = sequence.replace('T', 'U')

    # Get MFE structure
    structure, _ = RNA.fold(sequence)

    # Generate loop types (E: exterior, H: hairpin, I: internal, M: multiloop, S: stem)
    stack = []
    loop_types = ['.' for _ in range(len(sequence))]

    # First pass: mark paired positions and exterior loop
    for i, char in enumerate(structure):
        if char == '.':
            loop_types[i] = 'E'  # Exterior loop by default
        elif char == '(':
            stack.append(i)
            loop_types[i] = '('  # Mark stem
        elif char == ')':
            if stack:
                j = stack.pop()
                loop_types[i] = ')'  # Mark stem

    # Second pass: identify loop types
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                # If no nested pairs, this is a hairpin
                if structure[j + 1:i].count('(') == 0:
                    for k in range(j + 1, i):
                        loop_types[k] = 'H'  # Hairpin loop
                # If this contains other pairs, it might be internal or multi
                else:
                    # Count unpaired regions between paired regions
                    unpaired_regions = 0
                    in_unpaired = False
                    for k in range(j + 1, i):
                        if structure[k] == '.' and not in_unpaired:
                            unpaired_regions += 1
                            in_unpaired = True
                        elif structure[k] != '.' and in_unpaired:
                            in_unpaired = False

                    # If more than 2 unpaired regions, it's a multi-loop
                    if unpaired_regions > 2:
                        for k in range(j + 1, i):
                            if loop_types[k] == '.':
                                loop_types[k] = 'M'  # Multi-loop
                    # Otherwise internal loop
                    else:
                        for k in range(j + 1, i):
                            if loop_types[k] == '.':
                                loop_types[k] = 'I'  # Internal loop

    # Convert to bpRNA notation format
    bpRNA_notation = ''.join(loop_types)

    return structure, bpRNA_notation

class RNADataset(Dataset):
    def __init__(self, seqs, labels, precomputed_embeddings_path, ids, ew, bpp_path, transform=None, training=True, pad=False, k=5):
        self.transform = transform
        self.seqs = seqs
        self.data = []
        self.labels = labels.astype('float32')
        self.bpp_path = bpp_path
        self.ids = ids
        self.training = training
        self.bpps = []
        self.embeddings = torch.load(precomputed_embeddings_path)

        max_seq_len = max(len(seq) for seq in seqs)

        dm = get_distance_mask(max_seq_len)
        self.dms = np.asarray([dm for i in range(12)])

        self.lengths = []

        for i, id in tqdm(enumerate(self.ids), desc="Processing sequences"):
            seq = self.seqs[i]
            seq_len = len(seq)
            self.lengths.append(seq_len)

            # Path to BPP file
            bpp_file = os.path.join(self.bpp_path, 'new_sequences_bpps', f"{id}_bpp.npy")
            struc_file = os.path.join(self.bpp_path, 'new_sequences_bpps', f"{id}_struc.p")
            loop_file = os.path.join(self.bpp_path, 'new_sequences_bpps', f"{id}_loop.p")

            try:
                # Try to load pre-computed files
                bpps = np.load(bpp_file, allow_pickle=True)
                with open(struc_file, 'rb') as f:
                    structures = pickle.load(f)
                with open(loop_file, 'rb') as f:
                    loops = pickle.load(f)
            except (FileNotFoundError, IOError):
                # Generate on-the-fly using ViennaRNA
                print(f"Generating structures for sequence {id} (length: {seq_len})")

                # Generate BPP matrix
                bpp_matrix = generate_bpp_matrix(seq)

                # Generate structure and loop type using the detailed function
                structure, loop_annotation = generate_structure_and_loop(seq)
                structures = [structure]
                loops = [loop_annotation]

                # Create a single-element batch for consistency
                bpps = np.expand_dims(bpp_matrix, axis=0)


            # Apply padding if needed
            if pad and seq_len < 130:
                bpps = np.pad(bpps, ([0, 0], [0, 130 - seq_len], [0, 130 - seq_len]), constant_values=0)
            # Explicitly save padded (or unpadded if padding not applied) to disk
            np.save(bpp_file, bpps)
            with open(struc_file, 'wb') as f:
                pickle.dump(structures, f)
            with open(loop_file, 'wb') as f:
                pickle.dump(loops, f)

            # Process sequence, structure, and loop type
            input_data = []
            for j in range(min(bpps.shape[0], len(structures))):
                # Convert sequences to token indices
                seq = seq.replace('U', 'T')
                input_seq = np.asarray([tokens.index(s) if s in tokens else tokens.index('<unk>') for s in seq])
                input_structure = np.asarray(
                    [tokens.index(s) if s in tokens else tokens.index('<unk>') for s in structures[j]])

                # Ensure loop data matches sequence length
                valid_loop_tokens = {'B', 'E', 'H', 'I', 'M', 'S', 'X'}

                if j < len(loops) and len(loops[j]) == len(seq):
                    input_loop = np.asarray(
                        [tokens.index(s) if s in valid_loop_tokens else tokens.index('X') for s in loops[j]]
                    )
                else:
                    # Fallback: create default loop type (exterior)
                    input_loop = np.asarray([tokens.index('E') for _ in seq])

                # Stack sequence, structure, and loop data
                input_data.append(np.stack([input_seq, input_structure, input_loop], -1))
                valid_loop_token_indices = [tokens.index(t) for t in ['B', 'E', 'H', 'I', 'M', 'S', 'X']]
                assert np.all(np.isin(input_loop, valid_loop_token_indices)), \
                    f"Invalid loop type tokens detected: {np.unique(input_loop)}"

            # Convert to array and apply padding if needed
            input_data = np.asarray(input_data)
            if pad and seq_len < 130:
                input_data = np.pad(input_data, ([0, 0], [0, 130 - seq_len], [0, 0]), constant_values=tokens.index('<unk>'))

            # Store data and BPP matrices
            self.data.append(input_data)
            self.bpps.append(np.clip(bpps, 0, 1).astype('float32'))

        # Convert lengths to numpy array (safe, since it's 1D)
        self.lengths = np.asarray(self.lengths)
        self.ew = ew
        self.k = k

        # Create source masks for transformer
        self.src_masks = []
        for i in range(len(self.data)):
            seq_len = self.lengths[i]
            max_len = self.data[i].shape[1] if self.data[i].shape[1] > seq_len else seq_len
            self.src_masks.append(self.generate_src_mask(seq_len, max_len, self.k))

        self.pad = pad

    def generate_src_mask(self, L1, L2, k):
        mask = np.ones((k, L2), dtype='int8')
        for i in range(k):
            end_idx = min(L1 + i + 1 - k, L2)
            if end_idx > 0:
                mask[i, end_idx:] = 0
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Explicitly load precomputed embeddings
        embedding = self.embeddings[self.ids[idx]]  # precomputed RiNALMo embeddings

        if self.training:
            bpp_selection = np.random.randint(self.bpps[idx].shape[0])
            bpps = self.bpps[idx][bpp_selection]

            # Generate dynamic distance mask explicitly matching bpps dimensions
            seq_len = bpps.shape[0]
            dm = get_distance_mask(seq_len)

            bpps = np.concatenate([bpps.reshape(1, seq_len, seq_len), dm], axis=0).astype('float32')

            sample = {
                'embedding': torch.tensor(embedding, dtype=torch.float32),
                'labels': self.labels[idx],
                'bpp': bpps,
                'ew': self.ew[idx],
                'id': self.ids[idx],
                'src_mask': self.src_masks[idx]
            }
        else:
            bpps = self.bpps[idx]

            seq_len = bpps.shape[1]
            num_variants = bpps.shape[0]

            dm = get_distance_mask(seq_len)

            dm_batch = np.zeros((num_variants, 12, seq_len, seq_len), dtype=np.float32)
            for i in range(num_variants):
                for j in range(12):
                    dm_batch[i, j] = dm[j % 3]

            bpps_with_dm = np.concatenate([bpps[:, np.newaxis, :, :], dm_batch[:, :3, :, :]], axis=1)
            bpps_with_dm = bpps_with_dm.astype('float32')

            sample = {
                'embedding': torch.tensor(embedding, dtype=torch.float32),
                'bpp': bpps_with_dm.squeeze(),
                'src_mask': self.src_masks[idx].squeeze(),
                'labels': self.labels[idx].squeeze(),
                'id': self.ids[idx],
                'ew': self.ew[idx].squeeze()
            }

        return sample
