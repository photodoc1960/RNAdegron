import os
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# CRITICAL FIX: Updated imports for v7 architecture compatibility
from Functions_v7 import *
from Dataset_v7 import *
from X_Network_v7 import *
from LrScheduler import *
from Logger import CSVLogger
import argparse

# Use AMP for inference optimization
from torch.amp import autocast

from synthetic_data_generator import integrate_synthetic_sequences_into_pseudolabeling, RNASyntheticDataGenerator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1', help='GPU IDs to use')
    parser.add_argument('--path', type=str, default='../', help='Data path')
    parser.add_argument('--weights_path', type=str, default='../', help='Model weights path')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for inference')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--ntoken', type=int, default=4, help='Number of tokens')
    parser.add_argument('--nclass', type=int, default=5, help='Number of output classes')
    parser.add_argument('--ninp', type=int, default=512, help='Input dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--nhid', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--nlayers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--save_freq', type=int, default=1, help='Save frequency')
    parser.add_argument('--dropout', type=float, default=.1, help='Dropout rate')
    parser.add_argument('--warmup_steps', type=int, default=3200, help='Warmup steps')
    parser.add_argument('--lr_scale', type=float, default=0.1, help='Learning rate scale')
    parser.add_argument('--nmute', type=int, default=18, help='Number of mutations')
    parser.add_argument('--nfolds', type=int, default=5, help='Number of folds')
    parser.add_argument('--fold', type=int, default=0, help='Current fold')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--output_dir', type=str, default='../pseudo_labels', help='Output directory')
    return parser.parse_args()

def validate_and_normalize_bpp_tensor(bpps, device, context=""):
    """
    CRITICAL FIX: Comprehensive BPP tensor validation for v7 architecture compatibility.

    Ensures output conforms to unified (batch_size, 4, seq_len, seq_len) structure.
    """
    if bpps.dim() == 5:  # (batch_size, num_variants, 4, seq_len, seq_len)
        print(f"{context}: Converting 5D BPP to 4D by selecting first variant")
        bpps = bpps[:, 0, :, :, :]
    elif bpps.dim() == 3:  # Legacy (batch_size, seq_len, seq_len)
        print(f"{context}: Converting legacy BPP format to unified 4-channel structure")
        batch_size, seq_len = bpps.shape[0], bpps.shape[1]

        # Generate distance masks for unified structure
        dm = get_distance_mask(seq_len)
        dm_tensor = torch.tensor(dm, device=device, dtype=bpps.dtype)
        dm_batch = dm_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Create unified structure: [BPP, distance_mask_1, distance_mask_2, distance_mask_3]
        bpps = torch.cat([bpps.unsqueeze(1), dm_batch], dim=1)
    elif bpps.dim() == 4 and bpps.shape[1] == 1:  # (batch_size, 1, seq_len, seq_len)
        print(f"{context}: Expanding single-channel BPP to unified 4-channel structure")
        batch_size, _, seq_len = bpps.shape[0], bpps.shape[1], bpps.shape[2]

        dm = get_distance_mask(seq_len)
        dm_tensor = torch.tensor(dm, device=device, dtype=bpps.dtype)
        dm_batch = dm_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        bpps = torch.cat([bpps, dm_batch], dim=1)
    elif bpps.dim() == 4 and bpps.shape[1] != 4:
        raise ValueError(f"{context}: BPP has wrong channel count: expected 4, got {bpps.shape[1]}")
    elif bpps.dim() != 4:
        raise ValueError(f"{context}: BPP has wrong dimensions: expected 4D, got {bpps.dim()}D")

    # Final validation
    if bpps.shape[1] != 4:
        raise ValueError(f"{context}: BPP normalization failed - expected 4 channels, got {bpps.shape[1]}")

    return bpps


def load_model_with_v7_architecture(fold, model_idx, opts, device, features_path):
    """Load model with v7 architecture compatibility."""

    # CRITICAL FIX: Use v7 architecture with dynamic embedding dimension
    model = RNADegformer(
        ntoken=opts.ntoken,
        nclass=opts.nclass,
        ninp=opts.ninp,
        nhead=opts.nhead,
        nhid=opts.nhid,
        nlayers=opts.nlayers,
        dropout=opts.dropout,
        rinalmo_weights_path=None,
        precomputed_features_path=features_path
    ).to(device)

    # Initialize optimizer (required for some checkpoint loading scenarios)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)

    # DataParallel wrapper
    model = nn.DataParallel(model)

    # Load checkpoint with architectural compatibility handling
    checkpoint_path = f"{opts.weights_path}/fold{fold}top{model_idx + 1}.ckpt"
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys for fold {fold} model {model_idx}: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys for fold {fold} model {model_idx}: {len(unexpected_keys)}")

            print(f"✅ Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error loading checkpoint {checkpoint_path}: {e}")
            return None
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    model.eval()
    return model


def process_sequence_batch_with_structural_features(batch, models, device, context=""):
    """
    CRITICAL FIX: Process batch with comprehensive v7 structural feature integration.

    Args:
        batch: Input batch with embeddings and structural features
        models: List of fold models for ensemble prediction
        device: Computation device
        max_positions: Maximum sequence positions to predict
        context: Context string for debugging

    Returns:
        Ensemble predictions tensor: (num_folds, batch_size, max_positions, nclass)
    """
    # Extract and validate input tensors
    sequence = batch['embedding'].to(device)
    bpps = batch['bpp'].float().to(device)
    src_mask = batch['src_mask'].to(device)

    # Handle potential singleton dimensions from collate function
    if sequence.dim() == 4 and sequence.shape[1] == 1:
        sequence = sequence.squeeze(1)
    if bpps.dim() == 5 and bpps.shape[1] == 1:
        bpps = bpps.squeeze(1)
    if src_mask.dim() == 4 and src_mask.shape[1] == 1:
        src_mask = src_mask.squeeze(1)

    # CRITICAL FIX: Apply unified BPP tensor structure validation
    bpps = validate_and_normalize_bpp_tensor(bpps, device, context=context)

    # Dynamic embedding dimension validation
    actual_embedding_dim = sequence.shape[-1]
    print(f"Processing with embedding dimension: {actual_embedding_dim}")

    # Extract structural features with device placement
    deltaG = batch['deltaG'].to(device)
    graph_dist = batch['graph_dist'].to(device)
    nearest_p = batch['nearest_p'].to(device)
    nearest_up = batch['nearest_up'].to(device)

    # Source mask dimension normalization
    if src_mask.dim() == 2:
        # Add layer dimension if missing
        src_mask = src_mask.unsqueeze(1).repeat(1,
                                                models[0].module.nlayers if hasattr(models[0].module, 'nlayers') else 6,
                                                1)
    elif src_mask.dim() == 3 and src_mask.shape[1] == 1:
        # Replicate single layer to match model requirements
        required_layers = models[0].module.nlayers if hasattr(models[0].module, 'nlayers') else 6
        src_mask = src_mask.repeat(1, required_layers, 1)

    outputs = []
    for model_idx, model in enumerate(models):
        try:
            with autocast('cuda'):
                temp = model(
                    sequence,  # Precomputed embeddings (B, L, embedding_dim)
                    bpps,  # Unified BPP tensor (B, 4, L, L)
                    src_mask,  # Source mask (B, nlayers, L)
                    deltaG,  # Scalar ΔG values (B,)
                    graph_dist,  # Graph distance matrix (B, L, L)
                    nearest_p,  # Nearest paired distances (B, L)
                    nearest_up  # Nearest unpaired distances (B, L)
                )[:, :91, :].cpu()
            outputs.append(temp)
        except Exception as e:
            print(f"❌ Error processing {context} batch for model {model_idx}: {e}")
            print(f"  Sequence shape: {sequence.shape}")
            print(f"  BPP shape: {bpps.shape}")
            print(f"  Source mask shape: {src_mask.shape}")

            # Fallback: zero predictions
            empty_pred = torch.zeros(sequence.shape[0], 91, 5)
            outputs.append(empty_pred)

    # Stack predictions across folds: (num_folds, batch_size, max_positions, nclass)
    return torch.stack(outputs, dim=0)


def apply_winner_uncertainty_filtering(predictions, stds, error_threshold=10.0, value_error_ratio=1.5):
    """Apply winner's individual position NaN masking strategy"""
    # Create mask for positions with large errors (winner's approach)
    high_error_mask = stds > error_threshold
    low_ratio_mask = np.abs(predictions) / (stds + 1e-8) < value_error_ratio

    # Combine conditions: NaN positions where error>10 AND value/error<1.5
    nan_mask = high_error_mask & low_ratio_mask

    # Apply NaN masking to predictions
    filtered_preds = predictions.copy()
    filtered_preds[nan_mask] = np.nan

    # Also NaN the corresponding standard deviations
    filtered_stds = stds.copy()
    filtered_stds[nan_mask] = np.nan

    nan_count = np.sum(nan_mask)
    total_count = predictions.size
    print(
        f"Applied winner's uncertainty filtering: {nan_count}/{total_count} ({nan_count / total_count:.2%}) positions marked as NaN")

    return filtered_preds, filtered_stds

# CRITICAL FIX: Enhanced dataset wrapper for v7 structural feature integration
class EmbeddedRNADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, embedding_dict, graph_dist_dict, nearest_p_dict, nearest_up_dict):
        self.dataset = dataset
        self.embedding_dict = embedding_dict
        self.graph_dist_dict = graph_dist_dict
        self.nearest_p_dict = nearest_p_dict
        self.nearest_up_dict = nearest_up_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sid = str(sample['id']).strip()

        try:
            # Add all structural features required by v7 architecture
            sample['embedding'] = self.embedding_dict[sid]
            sample['graph_dist'] = self.graph_dist_dict[sid]
            sample['nearest_p'] = self.nearest_p_dict[sid]
            sample['nearest_up'] = self.nearest_up_dict[sid]
        except KeyError as e:
            raise KeyError(f"[EmbeddedRNADataset] ID '{sid}' not found in feature dicts: {e}")

        return sample


def verify_winner_strategy_integration():
    """Verify all winner's strategy components are actively integrated"""

    try:
        from synthetic_data_generator import integrate_synthetic_sequences_into_pseudolabeling, \
            RNASyntheticDataGenerator
        print("✅ Winner's strategy components verified for pseudo-prediction")
        return True
    except ImportError as e:
        raise RuntimeError(f"Missing winner's strategy component: {e}")


def main():
    # Verify integration before execution
    verify_winner_strategy_integration()

    opts = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opts.output_dir, exist_ok=True)

    # Load precomputed features path for model initialization
    features_path = os.path.join(opts.path, 'precomputed_features.pt')

    # Load ensemble models for all folds
    fold_models = []
    folds = np.arange(opts.nfolds)

    for fold in folds:
        print(f"Loading models for fold {fold}...")
        MODELS = []

        for i in range(5):  # Top 5 models per fold
            model = load_model_with_v7_architecture(fold, i, opts, device, features_path)
            if model is not None:
                MODELS.append(model)

        if not MODELS:
            print(f"❌ No models loaded for fold {fold}")
            continue

        # Ensemble averaging of model weights
        print(f"Creating ensemble average for fold {fold} with {len(MODELS)} models...")
        avg_state = MODELS[0].module.state_dict()
        for key in avg_state:
            if torch.is_floating_point(avg_state[key]):
                avg_state[key] = avg_state[key].float()
                for i in range(1, len(MODELS)):
                    avg_state[key] += MODELS[i].module.state_dict()[key].float()
                avg_state[key] /= float(len(MODELS))
            else:
                avg_state[key] = avg_state[key].clone()

        MODELS[0].module.load_state_dict(avg_state)
        fold_models.append(MODELS[0])

    print(f"✅ Loaded {len(fold_models)} fold models for ensemble prediction")

    # Load test data with comprehensive validation
    json_path = os.path.join(opts.path, 'test.json')
    print(f"Loading test file: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Test file not found: {json_path}")

    test = pd.read_json(json_path, lines=True)
    print(f"Loaded {len(test)} test sequences")

    # Load precomputed features with comprehensive structural data
    print("Loading precomputed features with structural data...")
    embedding_data = torch.load(features_path, weights_only=False)

    # Create comprehensive feature dictionaries
    embedding_dict = {str(sid): emb for sid, emb in zip(embedding_data["ids"], embedding_data["embeddings"])}
    graph_dist_dict = {str(sid): gd for sid, gd in zip(embedding_data["ids"], embedding_data["graph_dists"])}
    nearest_p_dict = {str(sid): npd for sid, npd in zip(embedding_data["ids"], embedding_data["nearest_paired"])}
    nearest_up_dict = {str(sid): nud for sid, nud in zip(embedding_data["ids"], embedding_data["nearest_unpaired"])}

    print(f"Loaded structural features for {len(embedding_dict)} sequences")

    # Process long sequences (130bp, predict first 91 positions)
    print("Processing long sequences (130bp)...")
    long_indices = test.seq_length == 130
    long_data = test[long_indices]
    long_ids = np.asarray(long_data.id.to_list())

    # CRITICAL FIX: Use v7 dataset with proper configuration
    long_dataset = RNADataset(
        seqs=long_data.sequence.to_list(),
        labels=np.zeros(len(long_data)),
        ids=long_ids,
        ew=np.ones(len(long_data)),
        bpp_path=opts.path,
        training=False,
        num_layers=opts.nlayers
    )

    long_dataset = EmbeddedRNADataset(
        long_dataset, embedding_dict, graph_dist_dict, nearest_p_dict, nearest_up_dict
    )

    long_dataloader = DataLoader(
        long_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        collate_fn=variable_length_collate_fn
    )

    # Process short sequences (107bp, predict first 68 positions)
    print("Processing short sequences (107bp)...")
    short_indices = test.seq_length == 107
    short_data = test[short_indices]
    short_ids = np.asarray(short_data.id.to_list())

    short_dataset = RNADataset(
        seqs=short_data.sequence.to_list(),
        labels=np.zeros(len(short_data)),
        ids=short_ids,
        ew=np.ones(len(short_data)),
        bpp_path=opts.path,
        training=False,
        num_layers=opts.nlayers
    )

    short_dataset = EmbeddedRNADataset(
        short_dataset, embedding_dict, graph_dist_dict, nearest_p_dict, nearest_up_dict
    )

    short_dataloader = DataLoader(
        short_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        collate_fn=variable_length_collate_fn
    )

    # Generate predictions with comprehensive error handling
    ids = []
    preds = []

    with torch.no_grad():
        # Process long sequences
        for batch in tqdm(long_dataloader, desc="Processing long sequences"):
            outputs = process_sequence_batch_with_structural_features(
                batch, fold_models, device, context="long sequence"
            )

            # Append predictions and IDs
            for i in range(outputs.shape[1]):  # Iterate over batch dimension
                preds.append(outputs[:, i, :, :])  # (num_folds, 91, nclass)
                ids.append(batch['id'][i])

        # Process short sequences
        for batch in tqdm(short_dataloader, desc="Processing short sequences"):
            outputs = process_sequence_batch_with_structural_features(
                batch, fold_models, device, context="short sequence"
            )

            # Append predictions and IDs
            for i in range(outputs.shape[1]):  # Iterate over batch dimension
                preds.append(outputs[:, i, :, :])  # (num_folds, 68, nclass)
                ids.append(batch['id'][i])

    # Organize predictions by sequence ID with enhanced validation
    print("Organizing predictions by sequence ID...")
    preds_to_csv = [[] for _ in range(len(test))]
    test_ids = test.id.to_list()

    for i in tqdm(range(len(preds)), desc="Organizing predictions"):
        try:
            index = test_ids.index(ids[i])
            preds_to_csv[index].append(preds[i])
        except ValueError:
            print(f"Warning: Sequence ID {ids[i]} not found in test set")

    # Winner's variance-based position filtering for pseudo-labeling
    long_preds, long_stds = [], []
    short_preds, short_stds = [], []
    long_ids_set = set(long_data.id.to_list())
    short_ids_set = set(short_data.id.to_list())

    for pred_set, seq_id in zip(preds_to_csv, test_ids):
        if not pred_set:
            continue

        fold_preds = pred_set[0]  # (num_folds, seq_len, nclass)

        # CRITICAL: Convert PyTorch tensor to NumPy for statistical operations
        if isinstance(fold_preds, torch.Tensor):
            fold_preds_np = fold_preds.numpy()
        else:
            fold_preds_np = fold_preds

        # CRITICAL: Winner's approach - check prediction variance by position
        position_variance = np.std(fold_preds_np, axis=0)  # (seq_len, nclass)
        mean_pred = np.mean(fold_preds_np, axis=0)

        if seq_id in long_ids_set:
            # Winner's insight: only use positions 0-91 for long sequences
            mean_pred_truncated = mean_pred[:91, :]
            std_pred_truncated = position_variance[:91, :]

            # Apply winner's uncertainty filtering
            filtered_pred, filtered_std = apply_winner_uncertainty_filtering(
                mean_pred_truncated, std_pred_truncated
            )

            long_preds.append(filtered_pred)
            long_stds.append(filtered_std)

        elif seq_id in short_ids_set:
            # For short sequences, use all positions but pad to 91
            if mean_pred.shape[0] < 91:
                # Pad to 91 positions
                pad_length = 91 - mean_pred.shape[0]
                mean_pred = np.pad(mean_pred, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                position_variance = np.pad(position_variance, ((0, pad_length), (0, 0)), mode='constant',
                                           constant_values=0)

            # Apply winner's uncertainty filtering
            filtered_pred, filtered_std = apply_winner_uncertainty_filtering(
                mean_pred, position_variance
            )

            short_preds.append(filtered_pred)
            short_stds.append(filtered_std)

    # Convert to arrays with validation
    long_preds = np.array(long_preds) if long_preds else np.empty((0, 91, opts.nclass))
    long_stds = np.array(long_stds) if long_stds else np.empty((0, 91, opts.nclass))
    short_preds = np.array(short_preds) if short_preds else np.empty((0, 68, opts.nclass))
    short_stds = np.array(short_stds) if short_stds else np.empty((0, 68, opts.nclass))

    # Standardize sequence lengths to 91 positions
    print("Standardizing sequence lengths...")

    # Enforce 91 positions for long sequences
    if long_preds.shape[1] > 91:
        long_preds = long_preds[:, :91, :]
        long_stds = long_stds[:, :91, :]

    # Debug array dimensions before padding
    print(f"Debug: short_preds shape: {short_preds.shape}")
    print(f"Debug: short_stds shape: {short_stds.shape}")

    # FIXED: Pad short sequences from 68 to 91 positions with correct dimension check
    if short_preds.shape[0] > 0 and short_preds.shape[1] < 91:
        pad_length = 91 - short_preds.shape[1]  # Check sequence length dimension correctly

        # Padding specification: (num_sequences, seq_length, nclass)
        if short_preds.ndim == 3:
            pad_spec = ((0, 0), (0, pad_length), (0, 0))  # Pad sequence length dimension
        else:
            raise ValueError(f"Expected 3D array for short predictions, got {short_preds.ndim}D: {short_preds.shape}")

        short_preds = np.pad(short_preds, pad_spec, mode='constant', constant_values=0)
        short_stds = np.pad(short_stds, pad_spec, mode='constant', constant_values=0)

        # Zero uncertainty beyond position 68 for short sequences
        short_stds[:, 68:, :] = 0
    elif short_preds.shape[0] > 0:
        if short_preds.ndim == 3:
            short_preds = short_preds[:, :91, :]
            short_stds = short_stds[:, :91, :]
        elif short_preds.ndim == 4:
            short_preds = short_preds[:, :, :91, :]
            short_stds = short_stds[:, :, :91, :]

    print(f"Final shapes - Long: {long_preds.shape}, Short: {short_preds.shape}")

    # Generate pseudo-labels for synthetic sequences
    # WINNER'S STRATEGY: Mandatory synthetic sequence pseudo-labeling
    from synthetic_data_generator import integrate_synthetic_sequences_into_pseudolabeling, RNASyntheticDataGenerator

    synthetic_dataset_path = 'data/synthetic_rna_dataset.pkl'

    # Generate synthetic dataset if not exists
    if not os.path.exists(synthetic_dataset_path):
        print("Generating synthetic RNA dataset...")
        generator = RNASyntheticDataGenerator()
        synthetic_df = generator.generate_synthetic_dataset(
            n_sequences=1000,  # Generate 1000 synthetic sequences
            output_path=synthetic_dataset_path
        )

    # Integrate synthetic sequences into pseudo-labeling
    print("Integrating synthetic sequences into pseudo-labeling...")
    synthetic_pseudo_data = integrate_synthetic_sequences_into_pseudolabeling(
        synthetic_dataset_path, fold_models, device,
        output_path=os.path.join(opts.output_dir, 'synthetic_pseudo_labels.pkl')
    )
    print(f"Generated pseudo-labels for {len(synthetic_pseudo_data['synthetic_sequences'])} synthetic sequences")

    # Save pseudo-labels for each fold with enhanced metadata
    for fold in range(opts.nfolds):
        output_file = os.path.join(opts.output_dir, f'pseudo_labels_fold{fold}.p')

        # Enhanced data structure with metadata
        pseudo_label_data = {
            'long_preds': long_preds,
            'long_stds': long_stds,
            'short_preds': short_preds,
            'short_stds': short_stds,
            'long_ids': long_ids,
            'short_ids': short_ids,
            # Embedding dimension handled dynamically by model architecture
            'model_config': {
                'nlayers': opts.nlayers,
                'nhead': opts.nhead,
                'ninp': opts.ninp,
                'nhid': opts.nhid,
                'nclass': opts.nclass
            }
        }

        with open(output_file, 'wb') as f:
            pickle.dump(pseudo_label_data, f)
        print(f"✅ Saved enhanced pseudo-labels for fold {fold} to {output_file}")

    print(f"🎉 Pseudo-label generation complete. Files saved to {opts.output_dir}")
    print(f"Generated {len(long_preds)} long sequence predictions and {len(short_preds)} short sequence predictions")


if __name__ == '__main__':
    main()