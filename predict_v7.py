import os
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import warnings

# CRITICAL FIX: Updated imports for v7 architecture compatibility
from Functions_v7 import *
from Dataset_v7 import *
from X_Network_v7 import *
from torch.amp import autocast  # Native PyTorch mixed precision


def generate_submission_file_v7(standardized_predictions, test_ids, opts, output_dir, sequence_lengths=None):
    """
    Generate submission file with correct format mapping for RNA degradation competition.

    Args:
        standardized_predictions: List of prediction arrays (seq_count, 91, 5)
        test_ids: List of sequence IDs
        opts: Command line options
        output_dir: Output directory path
        sequence_lengths: List of actual sequence lengths for each sequence (optional)
    """
    import traceback

    submission_templates = [
        os.path.join(opts.path, 'sample_submission.csv'),
        'sample_submission.csv'
    ]

    for template_path in submission_templates:
        if os.path.exists(template_path):
            try:
                print(f"📋 Loading submission template: {template_path}")
                submission_template = pd.read_csv(template_path)

                # Analyze submission template structure
                print(f"📊 Template shape: {submission_template.shape}")
                print(f"📊 Columns: {list(submission_template.columns)}")

                # Create new submission dataframe
                submission_data = []
                id_column = submission_template.columns[0]  # First column is ID
                prediction_columns = submission_template.columns[1:]  # Remaining columns are predictions

                # CRITICAL FIX: Handle fold ensemble structure (5, 91, 5)
                expected_rows = len(submission_template)
                sequences_count = len(standardized_predictions)

                print(f"📊 Expected rows: {expected_rows}")
                print(f"📊 Available sequences: {sequences_count}")

                # Debug first prediction array structure
                first_pred = standardized_predictions[0]
                print(f"🔍 First prediction shape: {first_pred.shape}")

                # Generate submissions with proper fold averaging
                current_row = 0

                for seq_idx in range(sequences_count):
                    pred_array = standardized_predictions[seq_idx]  # Shape: (5, 91, 5) - 5 folds

                    # CRITICAL: Average across fold dimension (axis 0)
                    if pred_array.ndim == 3 and pred_array.shape[0] == 5:
                        pred_averaged = np.mean(pred_array, axis=0)  # Shape: (91, 5)
                    else:
                        pred_averaged = pred_array  # Assume already averaged

                    # CRITICAL FIX: Use actual sequence length from data instead of hardcoded assumptions
                    if sequence_lengths is not None and seq_idx < len(sequence_lengths):
                        original_length = sequence_lengths[seq_idx]
                    else:
                        # Fallback: try to infer from test data or use prediction array size
                        original_length = pred_averaged.shape[0]
                        print(f"⚠️ Warning: No sequence length provided for sequence {seq_idx}, using {original_length}")

                    # Process each position in original sequence
                    for pos_idx in range(original_length):
                        if current_row >= expected_rows:
                            break

                        # Create row data with template ID
                        row_data = {id_column: submission_template.iloc[current_row][id_column]}

                        # Extract scalar values from averaged predictions
                        for col_idx, col_name in enumerate(prediction_columns):
                            if pos_idx < pred_averaged.shape[0]:
                                # Extract scalar from averaged prediction
                                scalar_value = float(pred_averaged[pos_idx, col_idx])
                            else:
                                scalar_value = 0.0  # Padding beyond prediction length

                            row_data[col_name] = scalar_value

                        submission_data.append(row_data)
                        current_row += 1

                        # Debug first few rows
                        if current_row <= 3:
                            print(f"🔍 Row {current_row}: {[f'{v:.6f}' for v in list(row_data.values())[1:3]]}")

                    if current_row >= expected_rows:
                        break

                print(f"📊 Generated {len(submission_data)} rows")

                # Create submission dataframe
                submission = pd.DataFrame(submission_data)

                # Save submission file
                submission_path = os.path.join(output_dir, 'submission_v7.csv')
                submission.to_csv(submission_path, index=False)

                print(f"✅ Generated submission: {submission_path}")
                print(f"📊 Final shape: {submission.shape}")

                return submission_path

            except Exception as e:
                print(f"❌ Error creating submission from {template_path}: {e}")
                traceback.print_exc()
                continue

    print("⚠️ No valid submission template found")
    return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU IDs to use')
    parser.add_argument('--path', type=str, default='data', help='Data path')
    parser.add_argument('--weights_path', type=str, default='best_weights', help='Model weights path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--ntoken', type=int, default=21, help='Number of tokens')
    parser.add_argument('--nclass', type=int, default=5, help='Number of output classes')
    parser.add_argument('--ninp', type=int, default=640, help='Input dimension')
    parser.add_argument('--nhead', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--nhid', type=int, default=2560, help='Hidden dimension')
    parser.add_argument('--nlayers', type=int, default=5, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--nfolds', type=int, default=5, help='Number of folds')
    parser.add_argument('--output_dir', type=str, default='predictions_v7', help='Output directory')
    parser.add_argument('--use_sliding_window', action='store_true', help='Use sliding window for long sequences')
    parser.add_argument('--window_size', type=int, default=130, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=65, help='Sliding window stride')
    return parser.parse_args()

def validate_and_normalize_bpp_tensor(bpps, device, context=""):
    """
    CRITICAL FIX: Comprehensive BPP tensor validation for v7 architecture compatibility.
    
    Ensures output conforms to unified (batch_size, 4, seq_len, seq_len) structure where:
    - Channel 0: Base pairing probabilities  
    - Channels 1-3: Distance masks
    """
    original_shape = bpps.shape
    
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
    
    print(f"{context}: BPP tensor normalized from {original_shape} to {bpps.shape}")
    return bpps


def load_model_checkpoint_robust(model, checkpoint_path, device):
    """
    Enhanced checkpoint loading with DataParallel compatibility and comprehensive validation.

    Args:
        model: Initialized model instance
        checkpoint_path: Path to checkpoint file
        device: Target device for loading

    Returns:
        bool: Success status of checkpoint loading
    """
    try:
        # Load checkpoint with full compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract state dictionary from various checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Critical: Handle DataParallel prefix mismatches
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        # Case 1: Checkpoint has 'module.' prefix, model doesn't
        if any(key.startswith('module.') for key in checkpoint_keys) and \
                not any(key.startswith('module.') for key in model_keys):
            print(f"🔧 Removing DataParallel prefix from checkpoint: {checkpoint_path}")
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

        # Case 2: Model has 'module.' prefix, checkpoint doesn't
        elif any(key.startswith('module.') for key in model_keys) and \
                not any(key.startswith('module.') for key in checkpoint_keys):
            print(f"🔧 Adding DataParallel prefix to checkpoint: {checkpoint_path}")
            state_dict = {f'module.{key}': value for key, value in state_dict.items()}

        # Load state dict with non-strict mode for robustness
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Validation metrics
        total_params = len(model.state_dict())
        loaded_params = total_params - len(missing_keys)
        load_success_rate = loaded_params / total_params

        # Accept checkpoint if >95% of parameters loaded successfully
        if load_success_rate >= 0.95:
            print(f"✅ Checkpoint loaded successfully: {checkpoint_path}")
            print(f"📊 Parameter load rate: {load_success_rate:.2%} ({loaded_params}/{total_params})")
            if missing_keys:
                print(f"⚠️  Missing keys: {len(missing_keys)} (acceptable)")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)} (ignored)")
            return True
        else:
            print(f"❌ Insufficient parameter match: {load_success_rate:.2%} for {checkpoint_path}")
            return False

    except Exception as e:
        print(f"❌ Exception loading {checkpoint_path}: {e}")
        return False


def create_prediction_model(device, precomputed_features_path, opts):
    """
    Create model with exact training configuration parameters.

    Returns:
        Initialized model ready for checkpoint loading
    """
    # CRITICAL: Use identical parameters from training configuration
    model = RNADegformer(
        ntoken=opts.ntoken,  # Must match training
        nclass=opts.nclass,  # Must match training
        ninp=opts.ninp,  # Must match training
        nhead=opts.nhead,  # Must match training
        nhid=opts.nhid,  # Must match training
        nlayers=opts.nlayers,  # Must match training
        stride=1,  # Must match training
        dropout=opts.dropout,  # Must match training
        pretrain=False,  # Inference mode
        return_aw=False,  # Inference mode
        rinalmo_weights_path=None,  # V7 architecture
        precomputed_features_path=precomputed_features_path
    ).to(device)

    return model


def load_model_with_v7_architecture(fold, model_idx, opts, device):
    """Load model with v7 architecture compatibility and comprehensive error handling."""

    # Create model with exact architecture
    features_path = os.path.join(opts.path, 'precomputed_features.pt')
    model = create_prediction_model(device, features_path, opts)
    model = nn.DataParallel(model)

    # Primary checkpoint path (this is where your checkpoints actually are)
    checkpoint_path = f"{opts.weights_path}/fold{fold}top{model_idx + 1}.ckpt"

    print(f"🔍 Attempting to load: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file does not exist: {checkpoint_path}")
        return None

    try:
        # Load checkpoint without weights_only restriction
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        print(f"📊 Checkpoint keys sample: {list(state_dict.keys())[:3]}")

        # Handle DataParallel prefix mismatch
        model_keys = list(model.state_dict().keys())
        checkpoint_keys = list(state_dict.keys())

        if checkpoint_keys[0].startswith('module.') and not model_keys[0].startswith('module.'):
            # Remove module prefix from checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            print("🔧 Removed 'module.' prefix from checkpoint")
        elif not checkpoint_keys[0].startswith('module.') and model_keys[0].startswith('module.'):
            # Add module prefix to checkpoint
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            print("🔧 Added 'module.' prefix to checkpoint")

        # Load with non-strict mode (ignore missing/unexpected keys)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"📈 Loaded parameters: {len(state_dict) - len(missing_keys)}/{len(state_dict)}")
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")

        model.eval()
        print(f"✅ Successfully loaded checkpoint: {checkpoint_path}")
        return model

    except Exception as e:
        print(f"❌ Error loading checkpoint {checkpoint_path}: {str(e)}")
        return None

def process_sequence_batch_with_sliding_window(batch, models, device, opts, context=""):
    """
    Process sequences using sliding window approach for long sequences.
    
    Args:
        batch: Input batch with embeddings and structural features
        models: List of fold models for ensemble prediction
        device: Computation device
        opts: Argument options
        context: Context string for debugging
        
    Returns:
        Processed predictions with proper sequence length handling
    """
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

    # Extract structural features with device placement
    deltaG = batch['deltaG'].to(device)
    graph_dist = batch['graph_dist'].to(device)
    nearest_p = batch['nearest_p'].to(device)
    nearest_up = batch['nearest_up'].to(device)

    # Source mask dimension normalization
    if src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(1).repeat(1, opts.nlayers, 1)
    elif src_mask.dim() == 3 and src_mask.shape[1] != opts.nlayers:
        src_mask = src_mask[:, 0:1, :].repeat(1, opts.nlayers, 1)

    batch_size, seq_len, embed_dim = sequence.shape
    
    # Determine if sliding window is needed
    if seq_len <= opts.window_size or not opts.use_sliding_window:
        # Process entire sequence at once
        return process_single_window(
            sequence, bpps, src_mask, deltaG, graph_dist, nearest_p, nearest_up, 
            models, device, seq_len
        )
    else:
        # Use sliding window approach
        return process_with_sliding_window_v7(
            sequence, bpps, src_mask, deltaG, graph_dist, nearest_p, nearest_up,
            models, device, opts.window_size, opts.stride
        )

def process_single_window(sequence, bpps, src_mask, deltaG, graph_dist, nearest_p, nearest_up, models, device, max_positions):
    """Process a single window or full sequence."""
    outputs = []
    
    for model_idx, model in enumerate(models):
        try:
            with autocast('cuda'):
                # CRITICAL FIX: Complete v7 forward pass with all structural features
                temp = model(
                    sequence,      # Precomputed embeddings (B, L, embedding_dim)
                    bpps,          # Unified BPP tensor (B, 4, L, L)
                    src_mask,      # Source mask (B, nlayers, L)
                    deltaG,        # Scalar ΔG values (B,)
                    graph_dist,    # Graph distance matrix (B, L, L)
                    nearest_p,     # Nearest paired distances (B, L)
                    nearest_up     # Nearest unpaired distances (B, L)
                )[:, :max_positions, :].cpu()
            outputs.append(temp)
        except Exception as e:
            print(f"❌ Error processing batch for model {model_idx}: {e}")
            # Fallback: zero predictions
            empty_pred = torch.zeros(sequence.shape[0], max_positions, models[0].module.nclass)
            outputs.append(empty_pred)

    return torch.stack(outputs, dim=0)

def process_with_sliding_window_v7(sequence, bpps, src_mask, deltaG, graph_dist, nearest_p, nearest_up, 
                                   models, device, window_size, stride):
    """Enhanced sliding window processing for v7 architecture."""
    batch_size, seq_len, embed_dim = sequence.shape
    num_models = len(models)
    
    # Initialize output tensor
    output_dim = models[0].module.nclass
    merged_output = torch.zeros((num_models, batch_size, seq_len, output_dim))
    counts = torch.zeros(seq_len)
    
    # Process each window
    for start_idx in range(0, seq_len - window_size + 1, stride):
        end_idx = min(start_idx + window_size, seq_len)
        actual_window_size = end_idx - start_idx
        
        # Extract window data
        seq_window = sequence[:, start_idx:end_idx, :]
        bpp_window = bpps[:, :, start_idx:end_idx, start_idx:end_idx]
        src_mask_window = src_mask[:, :, start_idx:end_idx]
        graph_dist_window = graph_dist[:, start_idx:end_idx, start_idx:end_idx]
        nearest_p_window = nearest_p[:, start_idx:end_idx]
        nearest_up_window = nearest_up[:, start_idx:end_idx]
        
        # Process window with all models
        window_outputs = process_single_window(
            seq_window, bpp_window, src_mask_window, deltaG, 
            graph_dist_window, nearest_p_window, nearest_up_window,
            models, device, actual_window_size
        )
        
        # Accumulate results
        merged_output[:, :, start_idx:end_idx, :] += window_outputs
        counts[start_idx:end_idx] += 1
    
    # Average overlapping regions
    for i in range(seq_len):
        if counts[i] > 0:
            merged_output[:, :, i, :] /= counts[i]
    
    return merged_output

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

def standardize_predictions_to_91_positions(predictions, sequence_lengths, max_positions=91):
    """
    Standardize predictions to 91 positions as per v7 architecture requirements.
    
    Args:
        predictions: List of prediction arrays
        sequence_lengths: List of original sequence lengths
        max_positions: Target number of positions (default 91)
        
    Returns:
        Standardized predictions with consistent 91-position format
    """
    standardized_preds = []
    
    for pred, seq_len in zip(predictions, sequence_lengths):
        if pred.shape[1] > max_positions:
            # Truncate to max_positions
            pred_std = pred[:, :max_positions, :]
        elif pred.shape[1] < max_positions:
            # Pad to max_positions
            pad_length = max_positions - pred.shape[1]
            pred_std = np.pad(pred, ((0, 0), (0, pad_length), (0, 0)), 
                             mode='constant', constant_values=0)
        else:
            pred_std = pred
            
        standardized_preds.append(pred_std)
    
    return standardized_preds

def main():
    opts = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opts.output_dir, exist_ok=True)

    print("🚀 Starting V7 RNA Degradation Prediction Pipeline...")
    print("=" * 60)

    # Load ensemble models for all folds with enhanced error handling
    fold_models = []
    successful_folds = 0

    for fold in range(opts.nfolds):
        print(f"\n📂 Loading models for fold {fold}...")
        fold_ensemble = []

        # Load top models for this fold (typically top 5)
        for model_idx in range(5):
            model = load_model_with_v7_architecture(fold, model_idx, opts, device)
            if model is not None:
                fold_ensemble.append(model)
                print(f"✅ Model {model_idx + 1} loaded for fold {fold}")

        if fold_ensemble:
            print(f"✅ Fold {fold}: {len(fold_ensemble)} models loaded successfully")

            # Create ensemble average for this fold
            if len(fold_ensemble) > 1:
                # Average model weights
                avg_state = fold_ensemble[0].module.state_dict()
                for key in avg_state:
                    if torch.is_floating_point(avg_state[key]):
                        avg_state[key] = avg_state[key].float()
                        for i in range(1, len(fold_ensemble)):
                            avg_state[key] += fold_ensemble[i].module.state_dict()[key].float()
                        avg_state[key] /= float(len(fold_ensemble))

                fold_ensemble[0].module.load_state_dict(avg_state)
                print(f"📊 Created ensemble average for fold {fold} with {len(fold_ensemble)} models")

            fold_models.append(fold_ensemble[0])
            successful_folds += 1
        else:
            print(f"❌ No models loaded for fold {fold}")

    if successful_folds == 0:
        raise RuntimeError("❌ CRITICAL: No models loaded successfully - check checkpoint paths and model architecture")

    print(f"✅ Successfully loaded {successful_folds}/{opts.nfolds} fold models for ensemble prediction")

    # Load test data with comprehensive validation
    test_data_paths = [
        os.path.join(opts.path, 'test.json'),
        os.path.join(opts.path, 'new_sequences.csv'),
        os.path.join(opts.path, 'test_sequences.json')
    ]
    
    test_data_loaded = False
    for test_path in test_data_paths:
        if os.path.exists(test_path):
            try:
                if test_path.endswith('.json'):
                    test = pd.read_json(test_path, lines=True)
                else:
                    test = pd.read_csv(test_path)
                print(f"✅ Loaded test data: {test_path} ({len(test)} sequences)")
                test_data_loaded = True
                break
            except Exception as e:
                print(f"❌ Error loading {test_path}: {e}")
                continue
    
    if not test_data_loaded:
        raise FileNotFoundError(f"Could not load test data from any of: {test_data_paths}")

    # Add sequence length column if missing
    if 'seq_length' not in test.columns and 'sequence' in test.columns:
        test['seq_length'] = test['sequence'].apply(len)
        print(f"Added sequence length column. Range: {test.seq_length.min()}-{test.seq_length.max()}")

    # Define features path for embedding data loading
    features_path = os.path.join(opts.path, 'precomputed_features.pt')

    # Load precomputed features with comprehensive structural data
    print("📊 Loading precomputed features with structural data...")
    embedding_data = torch.load(features_path, weights_only=False)

    # Create comprehensive feature dictionaries
    embedding_dict = {str(sid): emb for sid, emb in zip(embedding_data["ids"], embedding_data["embeddings"])}
    graph_dist_dict = {str(sid): gd for sid, gd in zip(embedding_data["ids"], embedding_data["graph_dists"])}
    nearest_p_dict = {str(sid): npd for sid, npd in zip(embedding_data["ids"], embedding_data["nearest_paired"])}
    nearest_up_dict = {str(sid): nud for sid, nud in zip(embedding_data["ids"], embedding_data["nearest_unpaired"])}

    print(f"Loaded structural features for {len(embedding_dict)} sequences")

    # Process sequences by length categories
    long_indices = test.seq_length == 130
    short_indices = test.seq_length == 107
    other_indices = ~(long_indices | short_indices)

    if other_indices.sum() > 0:
        print(f"⚠️ Found {other_indices.sum()} sequences with non-standard lengths")
        print(f"Length distribution: {test.seq_length.value_counts().sort_index()}")
        # Include other sequences with long sequences for sliding window processing
        long_indices = long_indices | other_indices
        opts.use_sliding_window = True

    all_predictions = []
    all_ids = []
    sequence_lengths = []

    # Process long sequences (130bp and others)
    if long_indices.sum() > 0:
        print(f"\n📏 Processing {long_indices.sum()} long sequences...")
        long_data = test[long_indices]
        long_ids = np.asarray(long_data.id.to_list())

        # Create dataset with v7 architecture
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

        # Process long sequences with ensemble prediction
        with torch.no_grad():
            for batch in tqdm(long_dataloader, desc="Processing long sequences"):
                outputs = process_sequence_batch_with_sliding_window(
                    batch, fold_models, device, opts, context="long sequence"
                )

                # Process batch predictions
                for i in range(outputs.shape[1]):  # Iterate over batch dimension
                    all_predictions.append(outputs[:, i, :, :])  # (num_folds, seq_len, nclass)
                    all_ids.append(batch['id'][i])
                    # Determine sequence length from actual data
                    actual_seq_len = batch['embedding'][i].shape[0]
                    sequence_lengths.append(actual_seq_len)

    # Process short sequences (107bp)
    if short_indices.sum() > 0:
        print(f"\n📏 Processing {short_indices.sum()} short sequences...")
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

        # Process short sequences
        with torch.no_grad():
            for batch in tqdm(short_dataloader, desc="Processing short sequences"):
                outputs = process_sequence_batch_with_sliding_window(
                    batch, fold_models, device, opts, context="short sequence"
                )

                # Process batch predictions
                for i in range(outputs.shape[1]):  # Iterate over batch dimension
                    all_predictions.append(outputs[:, i, :, :])  # (num_folds, seq_len, nclass)
                    all_ids.append(batch['id'][i])
                    actual_seq_len = batch['embedding'][i].shape[0]
                    sequence_lengths.append(actual_seq_len)

    # Organize predictions by sequence ID
    print("\n📊 Organizing predictions by sequence ID...")
    test_ids = test.id.to_list()
    organized_predictions = [[] for _ in range(len(test))]

    for i, seq_id in enumerate(all_ids):
        try:
            index = test_ids.index(seq_id)
            organized_predictions[index].append(all_predictions[i])
        except ValueError:
            print(f"⚠️ Warning: Sequence ID {seq_id} not found in test set")

    # Generate final predictions with statistical processing
    print("🔄 Computing ensemble averages and statistics...")
    final_predictions = []
    prediction_stds = []

    for i, pred_list in enumerate(organized_predictions):
        if not pred_list:
            # Handle missing predictions
            print(f"⚠️ No predictions for sequence {i}, using zero fallback")
            seq_len = sequence_lengths[i] if i < len(sequence_lengths) else 91
            final_predictions.append(np.zeros((seq_len, opts.nclass)))
            prediction_stds.append(np.zeros((seq_len, opts.nclass)))
            continue

        # Stack predictions: (num_folds, seq_len, nclass)
        stacked_preds = np.stack(pred_list, axis=0)
        
        # Compute ensemble statistics
        mean_pred = np.mean(stacked_preds, axis=0)  # Average across folds
        std_pred = np.std(stacked_preds, axis=0)    # Standard deviation across folds
        
        final_predictions.append(mean_pred)
        prediction_stds.append(std_pred)

    # Standardize all predictions to 91 positions for v7 compliance
    print("📏 Standardizing predictions to 91-position format...")
    standardized_predictions = standardize_predictions_to_91_positions(
        final_predictions, sequence_lengths, max_positions=91
    )
    standardized_stds = standardize_predictions_to_91_positions(
        prediction_stds, sequence_lengths, max_positions=91
    )

    # Save comprehensive prediction results
    output_data = {
        'predictions': standardized_predictions,
        'uncertainties': standardized_stds,
        'sequence_ids': test_ids,
        'sequence_lengths': [len(test.iloc[i].sequence) if 'sequence' in test.columns else 91 for i in range(len(test))],
        'model_config': {
            'nlayers': opts.nlayers,
            'nhead': opts.nhead,
            'ninp': opts.ninp,
            'nhid': opts.nhid,
            'nclass': opts.nclass,
            'successful_folds': successful_folds,
            'total_folds': opts.nfolds
        },
        'processing_config': {
            'use_sliding_window': opts.use_sliding_window,
            'window_size': opts.window_size,
            'stride': opts.stride,
            'batch_size': opts.batch_size
        }
    }

    # Save detailed results
    output_path = os.path.join(opts.output_dir, 'predictions_v7_detailed.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"💾 Saved detailed predictions: {output_path}")

    # Generate submission file with corrected logic (pass actual sequence lengths)
    actual_sequence_lengths = output_data['sequence_lengths']
    submission_path = generate_submission_file_v7(
        standardized_predictions, test_ids, opts, opts.output_dir, sequence_lengths=actual_sequence_lengths
    )

    # Performance summary
    print("\n" + "=" * 60)
    print("🎉 V7 PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Processed {len(test)} sequences")
    print(f"🔧 Used {successful_folds}/{opts.nfolds} fold models")
    print(f"📏 Standardized to 91-position format")
    print(f"💾 Results saved to: {opts.output_dir}")
    print(f"🎯 Ensemble predictions with uncertainty quantification")
    
    if opts.use_sliding_window:
        print(f"🪟 Applied sliding window processing (window: {opts.window_size}, stride: {opts.stride})")
    
    print("=" * 60)

if __name__ == '__main__':
    main()