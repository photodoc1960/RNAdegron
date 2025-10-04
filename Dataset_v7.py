import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from Functions_v7 import *
import matplotlib.pyplot as plt
import RNA
import random
import fcntl
import time as time_module
import tempfile

# CRITICAL FIX: Separate token vocabularies to prevent collisions
tokens = ['<cls>', '<pad>', '<eos>', '<unk>', '<mask>',
          'A', 'C', 'G', 'T', 'I', 'R', 'Y', 'K', 'M', 'S', 'W',
          'B', 'D', 'H', 'V', 'N', '-', 'E', 'X']

# CRITICAL FIX: Separate loop token mapping to prevent index collision with main tokens
LOOP_TOKEN_MAP = {
    'B': 0, 'E': 1, 'H': 2, 'I': 3, 'M': 4, 'S': 5, 'X': 6
}


def safe_file_write_with_lock(filepath, write_func, max_retries=3, timeout=10.0):
    """
    Safely write to file with file locking to prevent race conditions in multi-process scenarios.

    Args:
        filepath: Path to file to write
        write_func: Function that takes file path and performs the write operation
        max_retries: Maximum number of retry attempts
        timeout: Maximum time to wait for lock (seconds)

    Returns:
        bool: True if write succeeded, False otherwise
    """
    lock_file = filepath + '.lock'

    for attempt in range(max_retries):
        try:
            # Create lock file with exclusive access
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                # Acquire exclusive lock
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform the write operation
                write_func(filepath)

                # Release lock
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

                # Remove lock file
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

                return True

            except IOError:
                # Lock acquisition failed, close and retry
                os.close(lock_fd)
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

        except FileExistsError:
            # Lock file exists, another process is writing
            # Wait and retry
            time_module.sleep(0.1 * (attempt + 1))
            continue

        except Exception as e:
            print(f"Warning: Error during locked file write to {filepath}: {e}")
            return False

    print(f"Warning: Failed to acquire lock for {filepath} after {max_retries} attempts")
    return False


def variable_length_collate_fn(batch):
    batch_size = len(batch)
    max_seq_len = max(sample['data'].shape[0] for sample in batch)

    # Use the maximum sequence length from the batch for dynamic padding
    # This handles variable-length sequences more efficiently

    # --- Allocate padded tensors for per‑base inputs ---
    padded_data = torch.full((batch_size, max_seq_len, 3),
                             fill_value=14, dtype=torch.long)

    # CRITICAL FIX: Ensure 4-channel BPP structure consistency
    padded_bpp = torch.zeros((batch_size, 4, max_seq_len, max_seq_len),
                             dtype=torch.float32)
    padded_src_mask = torch.zeros((batch_size,
                                   batch[0]['src_mask'].shape[0],
                                   max_seq_len),
                                  dtype=torch.int8)
    padded_embeddings = None  # will init after seeing first embedding

    # --- Allocate containers for scalar & 2D structural features ---
    deltaG_tensor = torch.zeros((batch_size,), dtype=torch.float32)
    padded_graph_dist = torch.zeros((batch_size, max_seq_len, max_seq_len),
                                    dtype=torch.float32)
    padded_nearest_p = torch.zeros((batch_size, max_seq_len),
                                   dtype=torch.float32)
    padded_nearest_up = torch.zeros((batch_size, max_seq_len),
                                    dtype=torch.float32)

    # --- Allocate labels, error‑weights, IDs ---
    labels = None
    ews = None
    ids = []

    for i, sample in enumerate(batch):
        seq_len = sample['data'].shape[0]

        # data
        padded_data[i, :seq_len] = torch.from_numpy(sample['data'])

        # CRITICAL FIX: Handle BPP tensor structure validation
        bpp = sample['bpp']
        if bpp.ndim == 4 and bpp.shape[0] == 1:
            bpp = bpp.squeeze(0)
        elif bpp.ndim == 3 and bpp.shape[0] != 4:
            raise ValueError(f"BPP tensor must have 4 channels, got {bpp.shape[0]}")
        elif bpp.ndim != 3:
            raise ValueError(f"BPP tensor must be 3D (4, seq_len, seq_len), got {bpp.ndim}D")

        # Ensure BPP has correct unified structure
        if bpp.shape[0] != 4:
            raise ValueError(f"BPP must have 4 channels for unified structure, got {bpp.shape[0]}")

        eff = min(seq_len, bpp.shape[-1], max_seq_len)
        padded_bpp[i, :, :eff, :eff] = torch.from_numpy(
            bpp[:, :eff, :eff]
        )

        # src_mask
        mask_len = sample['src_mask'].shape[-1]
        padded_src_mask[i, :, :min(mask_len, max_seq_len)] = torch.from_numpy(
            sample['src_mask'][:, :min(mask_len, max_seq_len)]
        )

        # labels
        lab = torch.tensor(sample['labels'], dtype=torch.float32)
        if labels is None:
            labels = torch.zeros((batch_size, *lab.shape), dtype=torch.float32)
        labels[i] = lab

        # error‑weights
        ew = torch.tensor(sample['ew'], dtype=torch.float32)
        if ews is None:
            ews = torch.zeros((batch_size, *ew.shape), dtype=torch.float32)
        ews[i] = ew

        # IDs
        ids.append(sample['id'])

        # embeddings
        emb = sample['embedding']
        if padded_embeddings is None:
            padded_embeddings = torch.zeros((batch_size,
                                             max_seq_len,
                                             emb.shape[-1]),
                                            dtype=emb.dtype)
        if emb.shape[0] <= max_seq_len:
            padded_embeddings[i, :emb.shape[0]] = emb
        else:
            padded_embeddings[i] = emb[:max_seq_len]

        # scalar ΔG
        deltaG_tensor[i] = sample['deltaG'].item()

        # ─── Slice & pad graph‑distance matrix ─────────────────────────────
        gd = sample['graph_dist']
        if gd.shape[0] > seq_len:
            gd = gd[:seq_len, :seq_len]
        padded_graph_dist[i, :seq_len, :seq_len] = torch.from_numpy(gd)

        # ─── Slice & pad nearest‑paired vector ─────────────────────────────
        npv = sample['nearest_p']
        if npv.shape[0] > seq_len:
            npv = npv[:seq_len]
        padded_nearest_p[i, :seq_len] = torch.from_numpy(npv)

        # ─── Slice & pad nearest‑unpaired vector ───────────────────────────
        nuv = sample['nearest_up']
        if nuv.shape[0] > seq_len:
            nuv = nuv[:seq_len]
        padded_nearest_up[i, :seq_len] = torch.from_numpy(nuv)

    return {
        'data': padded_data,
        'bpp': padded_bpp,
        'src_mask': padded_src_mask,
        'labels': labels,
        'ew': ews,
        'id': ids,
        'embedding': padded_embeddings,
        'deltaG': deltaG_tensor,
        'graph_dist': padded_graph_dist,
        'nearest_p': padded_nearest_p,
        'nearest_up': padded_nearest_up
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
    # FIXED: Ensure sequence uses 'U' for ViennaRNA (no double conversion)
    sequence = sequence.replace('T', 'U').upper()

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


def compute_deltaG(sequence):
    """Compute minimum free energy (ΔG) for a sequence using ViennaRNA."""
    sequence = sequence.replace('T', 'U')
    _, mfe = RNA.fold(sequence)
    return float(mfe)


class RNADataset(Dataset):
    def __init__(self, seqs, labels, ids, ew, bpp_path, transform=None, training=True, pad=False, k=5, num_layers=6,
                 deltaG_lookup=None):
        self.transform = transform
        self.seqs = seqs
        self.data = []
        self.labels = labels.astype('float32')
        self.bpp_path = bpp_path
        self.ids = ids
        if deltaG_lookup is None:
            self.deltaG_lookup = {}
            for seq_id, seq in tqdm(zip(self.ids, self.seqs), desc="Computing ΔG"):
                self.deltaG_lookup[seq_id] = compute_deltaG(seq)
        else:
            self.deltaG_lookup = deltaG_lookup
        self.training = training
        self.bpps = []
        self.num_layers = num_layers

        # CRITICAL FIX: Unified sequence length handling
        # Determine standard sequence length for consistent padding
        self.max_seq_len = max(len(seq) for seq in seqs)

        # For RNA degradation competition, sequences are typically 107bp or 130bp
        # Use the actual maximum length found in data rather than hardcoding
        self.standard_length = self.max_seq_len
        print(f"Dataset standard sequence length set to: {self.standard_length}")

        dm = get_distance_mask(self.standard_length)  # Use standard length for distance mask
        self.dms = np.asarray([dm for i in range(12)])

        self.lengths = []

        for i, id in tqdm(enumerate(self.ids), desc="Processing sequences"):
            seq = self.seqs[i]
            seq_len = len(seq)
            self.lengths.append(seq_len)

            # Path to BPP file
            bpp_file = os.path.join(self.bpp_path, f"{id}_bpp.npy")
            struc_file = os.path.join(self.bpp_path, f"{id}_struc.p")
            loop_file = os.path.join(self.bpp_path, f"{id}_loop.p")

            try:
                # Try to load pre-computed files
                bpps = np.load(bpp_file, allow_pickle=True)
                if bpps.shape[-1] != bpps.shape[-2]:
                    print(f"[AUTOFIX] Corrupt BPP for {id}: shape {bpps.shape}, regenerating.")
                    raise IOError  # triggers the next except block that regenerates
                with open(struc_file, 'rb') as f:
                    structures = pickle.load(f)
                with open(loop_file, 'rb') as f:
                    loops = pickle.load(f)

                # FIXED: Regenerate instead of corrupting via truncation
                if len(structures[0]) != len(seq) or len(loops[0]) != len(seq):
                    print(
                        f"[{id}] Length mismatch detected - Seq: {len(seq)}, Struct: {len(structures[0])}, Loop: {len(loops[0])}")
                    print(f"[{id}] Forcing regeneration via ViennaRNA to prevent data corruption")
                    # Trigger regeneration block by raising exception
                    raise IOError("Length mismatch requires regeneration")

            except (FileNotFoundError, IOError):
                print(f"[{id}] Regenerating BPP, structure, and loop via ViennaRNA (len: {len(seq)})")
                bpp_matrix = generate_bpp_matrix(seq)
                structure, loop_annotation = generate_structure_and_loop(seq)
                structures = [structure]
                loops = [loop_annotation]
                bpps = np.expand_dims(bpp_matrix, axis=0)

                # CRITICAL FIX: Use file locking to prevent race conditions
                safe_file_write_with_lock(bpp_file, lambda path: np.save(path, bpps))
                safe_file_write_with_lock(struc_file, lambda path: pickle.dump(structures, open(path, 'wb')))
                safe_file_write_with_lock(loop_file, lambda path: pickle.dump(loops, open(path, 'wb')))
            except (FileNotFoundError, IOError, ValueError):
                # 🔁 Force regeneration with ViennaRNA
                print(f"[{id}] Regenerating BPP, structure, and loop via ViennaRNA (len: {len(seq)})")

                # Generate BPP
                bpp_matrix = generate_bpp_matrix(seq)

                # Generate structure and loop
                structure, loop_annotation = generate_structure_and_loop(seq)
                structures = [structure]
                loops = [loop_annotation]
                bpps = np.expand_dims(bpp_matrix, axis=0)

                # CRITICAL FIX: Use file locking to prevent race conditions
                safe_file_write_with_lock(bpp_file, lambda path: np.save(path, bpps))
                safe_file_write_with_lock(struc_file, lambda path: pickle.dump(structures, open(path, 'wb')))
                safe_file_write_with_lock(loop_file, lambda path: pickle.dump(loops, open(path, 'wb')))

            # Apply padding if needed - use standard length consistently
            padding_applied = False
            if pad and seq_len < self.standard_length:
                current_len = bpps.shape[1]
                assert bpps.shape[1] == bpps.shape[2], f"Non-square BPP before padding: {bpps.shape}"
                pad_len = self.standard_length - current_len
                if pad_len > 0:
                    bpps = np.pad(bpps, ([0, 0], [0, pad_len], [0, pad_len]), constant_values=0)
                    padding_applied = True

            # CRITICAL FIX: Only save to disk if data was regenerated or modified (padding applied)
            # This prevents unnecessary overwrites and race conditions in multi-process loading
            # Data is already saved in the exception handlers above when regenerated
            if padding_applied:
                # Save only when padding modifies the data, using file locking
                safe_file_write_with_lock(bpp_file, lambda path: np.save(path, bpps))
                # Note: structures and loops don't need re-saving as they weren't modified

            # Process sequence, structure, and loop type
            input_data = []
            for j in range(min(bpps.shape[0], len(structures))):
                # FIXED: Complete sequence, structure, and loop processing
                seq_for_tokens = seq.replace('U', 'T')  # For model tokenization
                input_seq = np.asarray(
                    [tokens.index(s) if s in tokens else tokens.index('<unk>') for s in seq_for_tokens])
                input_structure = np.asarray(
                    [tokens.index(s) if s in tokens else tokens.index('<unk>') for s in structures[j]])

                # CRITICAL FIX: Use separate loop token mapping to prevent collision
                if j < len(loops) and len(loops[j]) == len(seq):
                    input_loop = np.asarray(
                        [LOOP_TOKEN_MAP[s] if s in LOOP_TOKEN_MAP else LOOP_TOKEN_MAP['X'] for s in loops[j]]
                    )
                else:
                    # Fallback: create default loop type (exterior)
                    input_loop = np.asarray([LOOP_TOKEN_MAP['E'] for _ in seq])

                # Stack sequence, structure, and loop data
                input_data.append(np.stack([input_seq, input_structure, input_loop], -1))
                assert np.all((input_loop >= 0) & (input_loop < 7)), \
                    f"Loop label out of bounds: {np.unique(input_loop)}"

            # Convert to array and apply consistent padding
            input_data = np.asarray(input_data)
            if pad and seq_len < self.standard_length:
                current_len = bpps.shape[1]
                assert bpps.shape[1] == bpps.shape[2], f"Non-square BPP before padding: {bpps.shape}"
                pad_len = self.standard_length - current_len
                if pad_len > 0:
                    bpps = np.pad(bpps, ([0, 0], [0, pad_len], [0, pad_len]), constant_values=0)
                # Apply consistent padding to input_data using standard_length
                input_data = np.pad(input_data, ([0, 0], [0, self.standard_length - seq_len], [0, 0]),
                                    constant_values=tokens.index('<unk>'))

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
            self.src_masks.append(np.ones((self.num_layers, seq_len), dtype='int8'))

        self.pad = pad

        # CRITICAL FIX: Remove dual embedding computation - use only precomputed embeddings
        # Load precomputed structural features including embeddings
        feat_path = os.path.join(self.bpp_path, 'precomputed_features.pt')
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Precomputed features not found at {feat_path}. Run serialize_embeddings_v7.py first.")

        feat_data = torch.load(feat_path, weights_only=False)
        ids_all = feat_data['ids']
        embs_all = feat_data['embeddings']
        graph_all = feat_data['graph_dists']
        npd_all = feat_data['nearest_paired']
        nud_all = feat_data['nearest_unpaired']

        # Validate embedding dimensions
        if 'embedding_dim' in feat_data:
            print(f"Loaded embeddings with dimension: {feat_data['embedding_dim']}")
        else:
            print(f"Warning: No embedding dimension metadata found. Using shape: {embs_all.shape}")

        id_to_idx = {str(sid): idx for idx, sid in enumerate(ids_all)}

        # Map to current dataset's IDs
        self.embeddings = []
        self.graph_dists_list = []
        self.nearest_p_list = []
        self.nearest_up_list = []

        for sid in self.ids:
            sid_str = str(sid)
            if sid_str not in id_to_idx:
                raise KeyError(f"Sequence ID {sid_str} not found in precomputed features. "
                               f"Available IDs: {list(id_to_idx.keys())[:10]}...")

            idx = id_to_idx[sid_str]
            self.embeddings.append(embs_all[idx])
            self.graph_dists_list.append(graph_all[idx])
            self.nearest_p_list.append(npd_all[idx])
            self.nearest_up_list.append(nud_all[idx])

        print(f"Loaded {len(self.embeddings)} precomputed embeddings from centralized features.")

    def __len__(self):
        return len(self.data)

    def _get_or_compute_deltaG(self, idx):
        seq_id = self.ids[idx]
        if seq_id not in self.deltaG_lookup:
            self.deltaG_lookup[seq_id] = compute_deltaG(self.seqs[idx])
        return self.deltaG_lookup[seq_id]

    def __getitem__(self, idx):
        seq_len = self.lengths[idx]

        if self.training:
            bpp_selection = np.random.randint(self.bpps[idx].shape[0])
            bpps = self.bpps[idx][bpp_selection]

            # Generate dynamic distance mask matching bpps dimensions
            seq_len = bpps.shape[0]
            dm = get_distance_mask(seq_len)

            # CRITICAL FIX: Ensure consistent BPP tensor structure for training
            # Create unified format: (4, seq_len, seq_len) matching inference expectations
            bpps_unified = np.concatenate([bpps.reshape(1, seq_len, seq_len), dm], axis=0).astype('float32')

            sample = {
                'data': np.pad(self.data[idx][bpp_selection],
                               ((0, self.standard_length - self.data[idx][bpp_selection].shape[0]), (0, 0)),
                               mode='constant', constant_values=0),
                'labels': self.labels[idx],
                'bpp': bpps_unified,  # Use unified structure
                'ew': self.ew[idx],
                'deltaG': torch.tensor(
                    self._get_or_compute_deltaG(idx),
                    dtype=torch.float32),
                'id': self.ids[idx],
                'src_mask': self.src_masks[idx],
                'embedding': self.embeddings[idx],  # Use precomputed embedding
                'graph_dist': self.graph_dists_list[idx],
                'nearest_p': self.nearest_p_list[idx],
                'nearest_up': self.nearest_up_list[idx],
            }
        else:
            bpps = self.bpps[idx]
            seq_len = bpps.shape[1]
            num_variants = bpps.shape[0]

            # FIXED: Create consistent BPP structure for inference
            dm = get_distance_mask(seq_len)

            # For inference, select FIRST variant only to match training expectations
            bpp_variant = bpps[0]  # Use first variant consistently
            bpps_unified = np.concatenate([bpp_variant.reshape(1, seq_len, seq_len), dm], axis=0).astype('float32')
            # Result: (4, seq_len, seq_len) - SAME as training

            sample = {
                'data': self.data[idx].squeeze(),
                'bpp': bpps_unified,  # Use unified structure
                'src_mask': self.src_masks[idx].squeeze(),
                'labels': self.labels[idx].squeeze(),
                'id': self.ids[idx],
                'ew': self.ew[idx].squeeze(),
                'deltaG': torch.tensor(
                    self._get_or_compute_deltaG(idx),
                    dtype=torch.float32),
                'embedding': self.embeddings[idx],  # Use precomputed embedding
                'graph_dist': self.graph_dists_list[idx],
                'nearest_p': self.nearest_p_list[idx],
                'nearest_up': self.nearest_up_list[idx],
            }

        # ─── Augmentations: sequence reversal + target noise ──────────────────
        if self.training:
            # 1) Sequence reversal — flip every per-base channel
            if random.random() < 0.5:
                # flip raw data
                sample['data'] = np.flip(sample['data'], axis=0).copy()
                # flip pairing probabilities (both spatial dimensions)
                sample['bpp'] = np.flip(
                    np.flip(sample['bpp'], axis=1),
                    axis=2
                ).copy()
                # flip source mask
                sample['src_mask'] = np.flip(sample['src_mask'], axis=-1).copy()
                # flip embedding (via numpy → torch)
                emb_np = np.flip(
                    sample['embedding'].cpu().numpy(),
                    axis=0
                ).copy()
                sample['embedding'] = torch.from_numpy(emb_np) \
                    .to(sample['embedding'].dtype)
                # flip structural features (both axes for matrices, single axis for vectors)
                sample['graph_dist'] = np.flip(np.flip(sample['graph_dist'], axis=0), axis=1).copy()
                sample['nearest_p'] = np.flip(sample['nearest_p'], axis=0).copy()
                sample['nearest_up'] = np.flip(sample['nearest_up'], axis=0).copy()

            # 2) Target noise — jitter each label by N(0, σ²), σ = experimental error
            noise = np.random.randn(*sample['labels'].shape) * sample['ew']
            sample['labels'] = sample['labels'] + noise

        return sample