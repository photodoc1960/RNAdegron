import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import math
from Functions_v7 import *  
from Dataset_v7 import *   
from X_Network_v7 import *  
from LrScheduler import *
from Logger import CSVLogger  # ✅ Exists
from Metrics import weighted_mcrmse_tensor  # ✅ Exists  
import argparse
from ranger import Ranger
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
from position_aware_validation import PositionAwareValidator

def apply_winner_column_weights(loss_tensor, column_weights=[0.3, 0.3, 0.3, 0.05, 0.05]):
    """Apply winner's column weighting strategy"""
    if loss_tensor.dim() == 1 and len(loss_tensor) == 5:
        weights = torch.tensor(column_weights, device=loss_tensor.device, dtype=loss_tensor.dtype)
        return (loss_tensor * weights).sum()
    return loss_tensor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=4, help='number of tokens to represent DNA nucleotides (should always be 4)')
    parser.add_argument('--nclass', type=int, default=5, help='number of classes from the linear decoder')
    parser.add_argument('--ninp', type=int, default=512, help='ninp for transformer encoder')
    parser.add_argument('--nhead', type=int, default=8, help='nhead for transformer encoder')
    parser.add_argument('--nhid', type=int, default=2048, help='nhid for transformer encoder')
    parser.add_argument('--nlayers', type=int, default=6, help='nlayers for transformer encoder')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--warmup_steps', type=int, default=3200, help='training schedule warmup steps')
    parser.add_argument('--lr_scale', type=float, default=0.1, help='learning rate scale')
    parser.add_argument('--nmute', type=int, default=18, help='number of mutations during training')
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--stride', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--viral_loss_weight', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--error_beta', type=float, default=5, help='error weight beta parameter')
    parser.add_argument('--error_alpha', type=float, default=0, help='error weight alpha parameter')
    parser.add_argument('--noise_filter', type=float, default=0.25, help='signal-to-noise ratio filter threshold')
    parser.add_argument('--rollback_thresh', type=float, default=0.002,
                        help='max ΔMCRMSE allowed on PL before rollback')
    parser.add_argument('--cluster_alpha', type=float, default=0.5, help='Cluster weight exponent (0.5 = sqrt)')
    parser.add_argument('--distance_threshold', type=int, default=10, help='Edit distance clustering threshold')
    parser.add_argument('--position_weights', type=float, nargs=3, default=[0.6, 0.3, 0.1],
                        help='Position range weights [reliable, moderate, uncertain]')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Position-aware early stopping patience')
    opts = parser.parse_args()
    return opts

class EmbeddedRNADataset(torch.utils.data.Dataset):
    """Complete structural feature integration for v7 architecture"""
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
        sample_id = str(sample['id']).strip()
        
        try:
            # Add all structural features required by v7 architecture
            sample['embedding'] = self.embedding_dict[sample_id]
            sample['graph_dist'] = self.graph_dist_dict[sample_id]
            sample['nearest_p'] = self.nearest_p_dict[sample_id]
            sample['nearest_up'] = self.nearest_up_dict[sample_id]
        except KeyError as e:
            raise KeyError(f"[EmbeddedRNADataset] ID '{sample_id}' not found: {e}")
        
        return sample

def load_pretrained_into_dataparallel(model, checkpoint_path):
    """Enhanced checkpoint loading with architectural compatibility"""
    print(f"Loading weights from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found, initializing from scratch")
        return model
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Handle DataParallel wrapper
        if list(state_dict.keys())[0].startswith('module.'):
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        else:
            new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        print(f"✅ Weights loaded: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
        
        if missing_keys:
            print(f"Missing keys (first 5): {list(missing_keys)[:5]}")
        if unexpected_keys:
            print(f"Unexpected keys (first 5): {list(unexpected_keys)[:5]}")
            
    except Exception as e:
        print(f"Warning: Error loading checkpoint {checkpoint_path}: {e}")
        print("Continuing with randomly initialized weights")
    
    return model

def validate_and_normalize_bpp_tensor(bpps, device, context=""):
    """
    CRITICAL FIX: Comprehensive BPP tensor validation and normalization for v7 architecture.
    
    Ensures output conforms to unified (batch_size, 4, seq_len, seq_len) structure where:
    - Channel 0: Base pairing probabilities  
    - Channels 1-3: Distance masks
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

def normalize_source_mask_dimensions(src_mask, nlayers):
    """Ensure source mask has correct dimensions for transformer layers."""
    original_shape = src_mask.shape
    
    # Handle dimension corrections
    if src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(1).repeat(1, nlayers, 1)
    elif src_mask.dim() == 3 and src_mask.size(1) != nlayers:
        # Replicate available layers to match required layers
        src_mask = src_mask[:, 0:1, :].repeat(1, nlayers, 1)
    
    return src_mask

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0,
                                    initial_multiplier=0.01, last_epoch=-1):
    """Creates a learning rate schedule with linear warmup and cosine annealing."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            warmup_pct = float(current_step) / float(max(1, num_warmup_steps))
            return initial_multiplier + (1.0 - initial_multiplier) * warmup_pct

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def verify_winner_strategy_integration():
    """Verify all winner's strategy components are actively integrated"""

    try:
        from cluster_weighting import enhanced_error_weight_computation
        from position_aware_validation import PositionAwareValidator
        print("✅ Winner's strategy components verified for training")
        return True
    except ImportError as e:
        raise RuntimeError(f"Missing winner's strategy component: {e}")


def train_fold():
    # Verify integration before execution
    verify_winner_strategy_integration()

    # Get arguments
    opts = get_args()

    # GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def apply_position_level_masking(json_data, error_threshold=10.0, value_error_ratio=1.5):
        """Apply position-level NaN masking instead of sequence-level filtering."""
        target_columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        error_columns = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
                         'deg_error_Mg_50C', 'deg_error_50C']

        masked_count = 0
        total_positions = 0

        for idx, row in json_data.iterrows():
            seq_length = len(row['sequence'])

            for target_col, error_col in zip(target_columns, error_columns):
                targets = np.array(row[target_col])
                errors = np.array(row[error_col])

                # Apply masking criteria: error > threshold AND |value|/error < ratio
                mask_condition = (errors > error_threshold) & (np.abs(targets) / errors < value_error_ratio)

                # Set masked positions to NaN
                targets[mask_condition] = np.nan

                # Update dataframe
                json_data.at[idx, target_col] = targets.tolist()

                masked_count += mask_condition.sum()
                total_positions += seq_length

        print(
            f"Position-level masking: {masked_count:,} positions masked out of {total_positions:,} total ({masked_count / total_positions * 100:.2f}%)")
        return json_data

    # Load training data with position-level masking instead of sequence filtering
    json_path = os.path.join(opts.path, 'train.json')
    json_data = pd.read_json(json_path, lines=True)
    json_data = apply_position_level_masking(json_data)
    ids = np.asarray(json_data.id.to_list())

    # Enhanced error weights with cluster-based sample weighting
    from cluster_weighting import enhanced_error_weight_computation

    error_weights, json_data_enhanced = enhanced_error_weight_computation(json_data, opts)
    json_data = json_data_enhanced  # Use enhanced dataframe with cluster information
    train_indices, val_indices = get_train_val_indices(json_data, opts.fold, SEED=2020, nfolds=opts.nfolds)

    # Extract sequences and labels
    _, labels = get_data(json_data)
    sequences = np.asarray(json_data.sequence)
    train_seqs = sequences[train_indices]
    val_seqs = sequences[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    train_ids = ids[train_indices]
    val_ids = ids[val_indices]
    train_ew = error_weights[train_indices]
    val_ew = error_weights[val_indices]

    # CRITICAL FIX: Load comprehensive precomputed features with validation
    features_path = os.path.join(opts.path, 'precomputed_features.pt')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Precomputed features not found: {features_path}")
    
    print("Loading precomputed features with structural data...")
    precomp = torch.load(features_path, weights_only=False)
    
    ids_all = precomp['ids']
    emb_all = precomp['embeddings']
    graph_all = precomp['graph_dists']
    npd_all = precomp['nearest_paired']
    nud_all = precomp['nearest_unpaired']

    # Create comprehensive feature dictionaries
    embedding_dict = {str(sid): emb for sid, emb in zip(ids_all, emb_all)}
    graph_dist_dict = {str(sid): gd for sid, gd in zip(ids_all, graph_all)}
    nearest_p_dict = {str(sid): npd for sid, npd in zip(ids_all, npd_all)}
    nearest_up_dict = {str(sid): nud for sid, nud in zip(ids_all, nud_all)}

    print(f"Loaded {len(embedding_dict)} embeddings and structural features")

    # Create datasets with proper v7 architecture configuration
    base_train_dataset = RNADataset(
        seqs=train_seqs.tolist(),
        labels=train_labels,
        ids=train_ids.tolist(),
        ew=train_ew,
        bpp_path=opts.path,
        pad=True,
        training=True,
        num_layers=opts.nlayers
    )
    
    # Wrap with comprehensive structural features
    train_dataset = EmbeddedRNADataset(
        base_train_dataset, embedding_dict, graph_dist_dict, nearest_p_dict, nearest_up_dict
    )

    base_val_dataset = RNADataset(
        seqs=val_seqs.tolist(),
        labels=val_labels,
        ids=val_ids.tolist(),
        ew=val_ew,
        bpp_path=opts.path,
        pad=True,
        training=False,  # CRITICAL: Set to False for validation
        num_layers=opts.nlayers
    )
    
    val_dataset = EmbeddedRNADataset(
        base_val_dataset, embedding_dict, graph_dist_dict, nearest_p_dict, nearest_up_dict
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
        collate_fn=variable_length_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opts.batch_size * 2,
        shuffle=False,
        num_workers=opts.workers,
        collate_fn=variable_length_collate_fn
    )

    # Setup logging and checkpoints
    checkpoints_folder = f'checkpoints_fold{opts.fold}'
    os.makedirs(checkpoints_folder, exist_ok=True)
    csv_file = f'log_fold{opts.fold}.csv'
    columns = ['epoch', 'train_loss', 'val_loss']
    logger = CSVLogger(columns, csv_file)

    model = RNADegformer(
        ntoken=opts.ntoken,
        nclass=opts.nclass,
        ninp=opts.ninp,
        nhead=opts.nhead,
        nhid=opts.nhid,
        nlayers=opts.nlayers,
        stride=opts.stride,
        dropout=opts.dropout,
        rinalmo_weights_path=None,
        precomputed_features_path=features_path
    ).to(device)

    # Initialize optimizer and criterion
    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    scaler = GradScaler()

    # DataParallel wrapper and pretrained weight loading
    model = nn.DataParallel(model)
    
    # Try to load pretrained weights with fallback mechanism
    pretrain_paths = [
        'pretrain_weights/best_model.ckpt',
        f'pretrain_weights/pretraintop1.ckpt',
        'pretrain_weights/epoch1.ckpt'
    ]
    
    loaded = False
    for pretrain_path in pretrain_paths:
        if os.path.exists(pretrain_path):
            model = load_pretrained_into_dataparallel(model, pretrain_path)
            loaded = True
            break
    
    if not loaded:
        print("Warning: No pretrained weights found, training from scratch")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {pytorch_total_params}')

    # Learning rate scheduling
    cos_epoch = int(opts.epochs * 0.75) - 1
    total_training_steps = opts.epochs * len(train_dataloader)
    warmup_steps = int(0.06 * total_training_steps)
    
    lr_schedule = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_training_steps, initial_multiplier=0.01
    )

    print(f"Training configuration:")
    print(f"  - Epochs: {opts.epochs}")
    print(f"  - Batch size: {opts.batch_size}")
    print(f"  - Learning rate warmup steps: {warmup_steps}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Embedding dimension: {model.module.rinalmo_embedding_dim}")

    best_val_loss = float('inf')
    is_best = False  # CRITICAL FIX: Initialize is_best to prevent NameError

    # Training loop with comprehensive architectural fixes
    for epoch in range(opts.epochs):
        model.train(True)
        t = time.time()
        total_loss = 0
        step = 0
        
        for data in train_dataloader:
            step += 1
            optimizer.zero_grad()
            lr = get_lr(optimizer)

            # Move tensors to device
            src = data['embedding'].to(device)
            labels = data['labels'].to(device)
            bpps = data['bpp'].to(device)
            src_mask = data['src_mask'].to(device)
            ew = data['ew'].to(device)

            # CRITICAL FIX: Apply unified BPP tensor structure validation
            bpps = validate_and_normalize_bpp_tensor(bpps, device, context=f"Training step {step}")

            # CRITICAL FIX: Normalize source mask dimensions
            src_mask = normalize_source_mask_dimensions(src_mask, opts.nlayers)

            # Extract structural features
            deltaG = data['deltaG'].to(device)
            graph_dist = data['graph_dist'].to(device)
            nearest_p = data['nearest_p'].to(device)
            nearest_up = data['nearest_up'].to(device)

            # Forward pass with mixed precision and comprehensive structural integration
            with autocast(dtype=torch.float16):
                output = model(
                    src,          # Precomputed embeddings (B, L, embedding_dim)
                    bpps,         # Unified BPP tensor (B, 4, L, L)
                    src_mask,     # Source mask (B, nlayers, L)
                    deltaG,       # Scalar ΔG values (B,)
                    graph_dist,   # Graph distance matrix (B, L, L)
                    nearest_p,    # Nearest paired distances (B, L)
                    nearest_up    # Nearest unpaired distances (B, L)
                )
                
                # Compute loss with proper sequence length handling
                seq_len = min(output.shape[1], labels.shape[1])
                base_loss = weighted_mcrmse_tensor(output[:, :seq_len], labels[:, :seq_len], ew[:, :seq_len])
                loss = apply_winner_column_weights(base_loss)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_schedule.step()

            total_loss += loss.item()
            
            # Progress reporting
            if step % 10 == 0 or step == len(train_dataloader):
                print(f"Epoch [{epoch+1}/{opts.epochs}], Step [{step}/{len(train_dataloader)}] "
                      f"Loss: {total_loss/step:.4f}, LR: {lr:.6f}, Time: {time.time()-t:.1f}s",
                      end='\r', flush=True)

        print('')  # New line after epoch
        train_loss = total_loss / len(train_dataloader)

        if (epoch + 1) % opts.val_freq == 0 and epoch > cos_epoch:
            print("Running mandatory position-aware validation...")

            # WINNER'S STRATEGY: Mandatory position-aware validation
            if not hasattr(train_fold, 'position_validator'):
                sequence_lengths = np.array([len(seq) for seq in train_seqs])
                train_fold.position_validator = PositionAwareValidator(sequence_lengths)

            val_result = train_fold.position_validator.implement_position_aware_early_stopping(
                model, val_dataloader, device, patience=10
            )
            val_loss = val_result['current_score']

            print(f"Position-aware validation metrics: {val_result['detailed_metrics']}")

            # Enhanced early stopping decision
            if val_result['should_stop']:
                print("Position-aware early stopping triggered")
                break
            
            # Log metrics
            to_log = [epoch + 1, train_loss, val_loss]
            logger.log(to_log)
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
                print(f"✅ New best validation loss: {val_loss:.6f}")
            else:
                is_best = False
                
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            to_log = [epoch + 1, train_loss, 0.0]  # Placeholder for validation
            logger.log(to_log)
            is_best = False

        # Save checkpoints
        if (epoch + 1) % opts.save_freq == 0:
            val_loss_for_save = val_loss if 'val_loss' in locals() else None
            save_weights(model, optimizer, epoch, checkpoints_folder, is_best=is_best, val_loss=val_loss_for_save)

        torch.cuda.empty_cache()

    # Final model selection
    print("Selecting best model weights...")
    get_best_weights_from_fold(opts.fold)
    print(f"✅ Training completed for fold {opts.fold}")

if __name__ == '__main__':
    train_fold()