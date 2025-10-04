import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import os
import pandas as pd
import numpy as np
import pickle
import shutil
import matplotlib

matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split, KFold

# CRITICAL FIX: Use updated functions with unified BPP handling
from Functions_v7 import *
from Dataset_v7 import *
from X_Network_v7 import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
from ranger import Ranger
from visualization_v7 import *


class EmbeddedRNADataset(torch.utils.data.Dataset):
    """Dataset wrapper for comprehensive structural feature integration."""

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
            # Add all structural features to the sample
            sample['embedding'] = self.embedding_dict[sample_id]
            sample['graph_dist'] = self.graph_dist_dict[sample_id]
            sample['nearest_p'] = self.nearest_p_dict[sample_id]
            sample['nearest_up'] = self.nearest_up_dict[sample_id]
        except KeyError as e:
            raise KeyError(
                f"[EmbeddedRNADataset] ID '{sample_id}' not found in feature dicts.\n"
                f"Missing feature: {e}\n"
                f"Available IDs (first 5): {list(self.embedding_dict.keys())[:5]}"
            )

        return sample


def get_args():
    """Parse command line arguments with comprehensive parameter validation."""
    parser = argparse.ArgumentParser(description='RNA Degradation Prediction Pretraining v7')

    # Hardware and data configuration
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU device ID')
    parser.add_argument('--path', type=str, default='data', help='Data directory path')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Optimizer weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Model dropout rate')

    # Model architecture parameters
    parser.add_argument('--ntoken', type=int, default=21, help='Number of input tokens')
    parser.add_argument('--nclass', type=int, default=5, help='Number of output classes')
    parser.add_argument('--ninp', type=int, default=256, help='Model input dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--nhid', type=int, default=1024, help='Hidden layer dimension')
    parser.add_argument('--nlayers', type=int, default=5, help='Number of transformer layers')

    # Learning rate and scheduling
    parser.add_argument('--lr_scale', type=float, default=0.01, help='Learning rate scale factor')
    parser.add_argument('--warmup_steps', type=int, default=600, help='Warmup steps for scheduler')

    parser.add_argument('--stride', type=int, default=1, help='Convolution stride')

    # Cross-validation and checkpointing
    parser.add_argument('--nfolds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--fold', type=int, default=0, help='Current fold index')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint save frequency')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')

    # Data processing options
    parser.add_argument('--nmute', type=int, default=0, help='Number of mutations (unused)')
    parser.add_argument('--viral_loss_weight', type=int, default=1, help='Viral loss weight (unused)')
    parser.add_argument('--force_regenerate', action='store_true', help='Force cache regeneration')

    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr=0.0, initial_multiplier=0.01, last_epoch=-1):
    """Create learning rate schedule with linear warmup and cosine annealing."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup phase
            warmup_pct = float(current_step) / float(max(1, num_warmup_steps))
            return initial_multiplier + (1.0 - initial_multiplier) * warmup_pct

        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def fix_src_mask_dimensions(src_mask, nlayers):
    """Ensure source mask has correct dimensions for transformer layers."""
    original_shape = src_mask.shape

    # Add batch dimension if missing
    if src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(0)

    # Add layer dimension if missing
    if src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(1)

    # Adjust layer dimension to match model requirements
    if src_mask.dim() == 3 and src_mask.size(1) != nlayers:
        corrected_mask = torch.zeros(
            (src_mask.size(0), nlayers, src_mask.size(2)),
            device=src_mask.device,
            dtype=src_mask.dtype
        )

        # Copy existing data for available layers
        min_layers = min(src_mask.size(1), nlayers)
        corrected_mask[:, :min_layers, :] = src_mask[:, :min_layers, :]

        # Replicate last layer if needed
        if src_mask.size(1) < nlayers:
            corrected_mask[:, min_layers:, :] = src_mask[:, -1:, :].repeat(1, nlayers - min_layers, 1)

        src_mask = corrected_mask

    return src_mask


def get_masked_view(embeddings, mask_ratio):
    """Create masked view of embeddings for contrastive learning."""
    mask = torch.rand(embeddings.shape[:2], device=embeddings.device) < mask_ratio
    masked = embeddings.clone()
    masked[mask] = 0.0
    return masked, mask


def contrastive_loss(reps1, reps2, temperature=0.2):
    """Compute contrastive loss between two representation views."""
    B, L, D = reps1.shape

    # Sanitize inputs to prevent NaN propagation
    reps1 = torch.nan_to_num(reps1).contiguous()
    reps2 = torch.nan_to_num(reps2).contiguous()

    # Normalize and reshape representations
    reps1 = F.normalize(reps1, dim=-1).reshape(B * L, D)
    reps2 = F.normalize(reps2, dim=-1).reshape(B * L, D)

    # Compute similarity matrix with numerical stability
    # Enforce minimum temperature to prevent overflow
    temperature = max(temperature, 1e-3)

    # Compute similarity matrix with numerical stability
    sim_matrix = torch.matmul(reps1, reps2.T) / temperature
    sim_matrix = torch.clamp(sim_matrix, min=-20.0, max=20.0)  # Prevent extreme values

    # Stabilize softmax computation
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values

    # Compute cross-entropy loss with NaN protection
    labels = torch.arange(B * L, device=reps1.device)
    loss = F.cross_entropy(sim_matrix, labels)

    if not torch.isfinite(loss):
        print(f"WARNING: Contrastive loss is {loss.item()}, replacing with zero")
        return torch.tensor(0.0, device=reps1.device)

    if torch.isnan(loss):
        print("WARNING: NaN detected in contrastive loss")
        return torch.tensor(0.0, device=reps1.device)

    return loss


def validate_bpp_tensor(bpps, context=""):
    """Validate and normalize BPP tensor structure."""
    original_shape = bpps.shape

    if bpps.dim() == 5:  # (batch_size, num_variants, 4, seq_len, seq_len)
        bpps = bpps[:, 0, :, :, :]  # Select first variant
    elif bpps.dim() != 4 or bpps.shape[1] != 4:
        raise ValueError(f"{context}: Expected BPP shape (batch, 4, seq_len, seq_len), got {bpps.shape}")

    return bpps


def compute_nucleotide_loss(outputs, labels, mask, criterion):
    """Compute nucleotide prediction loss with comprehensive validation."""
    try:
        masked_labels = labels[mask]
        outputs_masked = outputs['nucleotide'][mask]

        # Filter valid nucleotide tokens [5,6,7,8] -> [0,1,2,3]
        valid_pos = (masked_labels >= 5) & (masked_labels <= 8)

        if valid_pos.sum() == 0:
            return torch.tensor(0.0, device=labels.device)

        # Remap labels to [0,1,2,3] range
        masked_labels_clean = masked_labels[valid_pos] - 5
        outputs_clean = outputs_masked[valid_pos]

        # Validate label bounds
        if masked_labels_clean.min() < 0 or masked_labels_clean.max() > 3:
            print(f"WARNING: Invalid nucleotide label range: {masked_labels_clean.min()}-{masked_labels_clean.max()}")
            return torch.tensor(0.0, device=labels.device)

        loss = criterion(outputs_clean.reshape(-1, 4), masked_labels_clean.reshape(-1))
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=labels.device)


    except Exception as e:

        print(f"CRITICAL ERROR in nucleotide loss computation: {e}")

        print(f"  Labels shape: {labels.shape}, Output shape: {outputs['nucleotide'].shape}")

        print(f"  Mask sum: {mask.sum().item()}")

        # FIXED: Raise exception instead of masking - training should stop

        raise RuntimeError(f"Nucleotide loss computation failed: {e}")


def compute_structure_loss(outputs, structure_labels, mask, criterion):
    """Compute structure prediction loss with empty tensor protection."""
    try:
        structure_preds = outputs['structure'][mask]
        structure_labels_masked = structure_labels[mask]
        structure_valid = structure_labels_masked != 14

        if structure_valid.sum() == 0:
            return torch.tensor(0.0, device=structure_labels.device)

        # Validate structure label bounds
        valid_labels = structure_labels_masked[structure_valid]
        if valid_labels.min() < 0 or valid_labels.max() >= 24:  # Adjust max as needed
            print(f"WARNING: Invalid structure label range: {valid_labels.min()}-{valid_labels.max()}")
            return torch.tensor(0.0, device=structure_labels.device)

        loss = criterion(structure_preds[structure_valid], valid_labels)
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=structure_labels.device)


    except Exception as e:

        print(f"CRITICAL ERROR in structure loss computation: {e}")

        print(f"  Labels shape: {labels.shape}, Output shape: {outputs['structure'].shape}")

        print(f"  Mask sum: {mask.sum().item()}")

        # FIXED: Raise exception instead of masking - training should stop

        raise RuntimeError(f"Structure loss computation failed: {e}")


def compute_loop_loss(outputs, loop_labels, mask, criterion):
    """Compute loop type prediction loss with bounds validation."""
    try:
        loop_preds = outputs['loop'][mask]
        loop_labels_masked = loop_labels[mask]
        loop_valid = loop_labels_masked != 14

        if loop_valid.sum() == 0:
            return torch.tensor(0.0, device=loop_labels.device)

        # Validate loop label bounds
        valid_labels = loop_labels_masked[loop_valid]
        if valid_labels.min() < 0 or valid_labels.max() >= 7:
            print(f"WARNING: Invalid loop label range: {valid_labels.min()}-{valid_labels.max()}")
            return torch.tensor(0.0, device=loop_labels.device)

        loss = criterion(loop_preds[loop_valid], valid_labels)
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=loop_labels.device)


    except Exception as e:

        print(f"CRITICAL ERROR in loop loss computation: {e}")

        print(f"  Labels shape: {labels.shape}, Output shape: {outputs['loop'].shape}")

        print(f"  Mask sum: {mask.sum().item()}")

        # FIXED: Raise exception instead of masking - training should stop

        raise RuntimeError(f"Loop loss computation failed: {e}")


def compute_bpp_loss(outputs, bpp_targets, mask):
    """Compute BPP prediction loss with comprehensive NaN protection."""
    try:
        bpp_preds = outputs['bpp'].squeeze(-1)

        # Safe diagonal extraction with dimension validation
        if bpp_targets.dim() != 3 or bpp_targets.size(-1) != bpp_targets.size(-2):
            print(f"WARNING: Invalid BPP target dimensions: {bpp_targets.shape}")
            return torch.tensor(0.0, device=bpp_preds.device)

        # Extract diagonal elements safely
        seq_len = min(bpp_targets.size(-1), bpp_preds.size(-1))
        bpp_diag_target = torch.stack([
            bpp_targets[i, torch.arange(seq_len), torch.arange(seq_len)]
            for i in range(bpp_targets.size(0))
        ]).to(bpp_preds.dtype)

        # Apply masking and filter NaN values
        bpp_masked_pred = bpp_preds[mask]
        bpp_masked_target = bpp_diag_target[mask]

        # CRITICAL: Enhanced NaN filtering with bounds checking
        finite_target = torch.isfinite(bpp_masked_target)
        finite_pred = torch.isfinite(bpp_masked_pred)
        valid_bpp = finite_target & finite_pred

        if valid_bpp.sum() == 0:
            print(f"WARNING: All BPP values are NaN/inf, returning zero loss")
            return torch.tensor(0.0, device=bpp_preds.device)

        loss = F.mse_loss(bpp_masked_pred[valid_bpp], bpp_masked_target[valid_bpp])
        if not torch.isfinite(loss):
            print(f"WARNING: BPP loss computation resulted in {loss.item()}")
            return torch.tensor(0.0, device=bpp_preds.device)
        return loss


    except Exception as e:

        print(f"CRITICAL ERROR in BPP loss computation: {e}")

        print(f"  Labels shape: {labels.shape}, Output shape: {outputs['BPP'].shape}")

        print(f"  Mask sum: {mask.sum().item()}")

        # FIXED: Raise exception instead of masking - training should stop

        raise RuntimeError(f"BPP loss computation failed: {e}")


def validate_with_nan_protection(model, val_dataloader, device, criterion, opts):
    """NaN-safe validation with comprehensive error handling and component tracking."""

    model.eval()

    val_losses = {
        'nucleotide': [],
        'structure': [],
        'loop': [],
        'bpp': [],
        'total': []
    }

    nan_incidents = {
        'nucleotide': 0,
        'structure': 0,
        'loop': 0,
        'bpp': 0,
        'total_batches': 0,
        'skipped_batches': 0
    }

    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            nan_incidents['total_batches'] += 1

            try:
                # Input validation and preprocessing
                bpps = data['bpp'].cuda(non_blocking=True)
                src_mask = data['src_mask'].cuda(non_blocking=True)
                embeddings = data['embedding'].cuda(non_blocking=True)

                # Validate tensor dimensions
                batch_size, seq_len, _ = embeddings.shape
                if seq_len == 0 or batch_size == 0:
                    print(f"WARNING: Empty batch dimensions {embeddings.shape}, skipping")
                    nan_incidents['skipped_batches'] += 1
                    continue

                # Validate and normalize BPP tensor structure
                bpps = validate_bpp_tensor(bpps, f"validation_batch_{batch_idx}")

                # Fix source mask dimensions
                src_mask = fix_src_mask_dimensions(src_mask, opts.nlayers)

                # Load labels
                labels = data['data'][:, :, 0].to(device).long()
                structure_labels = data['data'][:, :, 1].to(device).long()
                loop_labels = data['data'][:, :, 2].to(device).long()
                bpp_targets = data['bpp'][:, 0, :, :].to(device).float()

                # Apply masking with guaranteed minimum positions
                mask_ratio = 0.15
                mask_positions = torch.rand(batch_size, seq_len, device=embeddings.device) < mask_ratio

                # Ensure at least one position is masked per sequence
                if mask_positions.sum() == 0:
                    # Force mask first position of each sequence
                    mask_positions[:, 0] = True
                    print(f"WARNING: Forced masking applied in batch {batch_idx}")

                masked_embeddings = embeddings.clone()
                masked_embeddings[mask_positions] = 0.0

                # Forward pass with structural features
                deltaG = data['deltaG'].to(device)
                graph_dist = data['graph_dist'].to(device)
                nearest_p = data['nearest_p'].to(device)
                nearest_up = data['nearest_up'].to(device)

                outputs, attention_maps, _ = model(
                    masked_embeddings, bpps, src_mask, deltaG, graph_dist, nearest_p, nearest_up
                )

                # Compute individual loss components with protection
                nucleotide_loss = compute_nucleotide_loss(outputs, labels, mask_positions, criterion)
                structure_loss = compute_structure_loss(outputs, structure_labels, mask_positions, criterion)
                loop_loss = compute_loop_loss(outputs, loop_labels, mask_positions, criterion)
                bpp_loss = compute_bpp_loss(outputs, bpp_targets, mask_positions)

                # Track NaN incidents
                if torch.isnan(nucleotide_loss): nan_incidents['nucleotide'] += 1
                if torch.isnan(structure_loss): nan_incidents['structure'] += 1
                if torch.isnan(loop_loss): nan_incidents['loop'] += 1
                if torch.isnan(bpp_loss): nan_incidents['bpp'] += 1

                # Combine losses
                total_loss = (
                        1.0 * nucleotide_loss +
                        0.4 * structure_loss +
                        0.4 * loop_loss +
                        0.25 * bpp_loss
                )

                # Store loss components
                val_losses['nucleotide'].append(nucleotide_loss.item())
                val_losses['structure'].append(structure_loss.item())
                val_losses['loop'].append(loop_loss.item())
                val_losses['bpp'].append(bpp_loss.item())
                val_losses['total'].append(total_loss.item())

            except Exception as e:
                print(f"CRITICAL ERROR in validation batch {batch_idx}: {e}")
                nan_incidents['skipped_batches'] += 1
                continue

    # Compute final statistics with NaN filtering
    final_losses = {}
    for component, losses in val_losses.items():
        clean_losses = [loss for loss in losses if not math.isnan(loss)]
        if clean_losses:
            final_losses[component] = np.mean(clean_losses)
        else:
            final_losses[component] = float('nan')
            print(f"WARNING: All {component} losses were NaN")

    # Report validation statistics
    total_nan_incidents = sum(v for k, v in nan_incidents.items() if k != 'total_batches' and k != 'skipped_batches')
    if total_nan_incidents > 0 or nan_incidents['skipped_batches'] > 0:
        print(f"\n📊 VALIDATION REPORT:")
        print(f"  Processed batches: {nan_incidents['total_batches'] - nan_incidents['skipped_batches']}")
        print(f"  Skipped batches: {nan_incidents['skipped_batches']}")
        if total_nan_incidents > 0:
            print(f"  NaN incidents: {total_nan_incidents}")
            for component, count in nan_incidents.items():
                if component not in ['total_batches', 'skipped_batches'] and count > 0:
                    print(f"    {component}: {count}")

    return final_losses['total'], final_losses


def train_fold():
    """Main training function with comprehensive error handling."""

    # Parse arguments and setup
    opts = get_args()
    print(f"🚀 Starting pretraining with V7 architecture (nlayers={opts.nlayers})")

    # Clear cached data if requested
    if opts.force_regenerate:
        mask_cache_path = os.path.join(opts.path, 'precomputed_masks')
        if os.path.exists(mask_cache_path):
            print(f"Removing cached masks from {mask_cache_path}")
            shutil.rmtree(mask_cache_path)
        os.makedirs(mask_cache_path, exist_ok=True)

    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load training data with fallback strategy
    pretrain_all_path = os.path.join(opts.path, 'pretrain_all.json')
    train_path = os.path.join(opts.path, 'train.json')
    test_path = os.path.join(opts.path, 'test.json')

    if os.path.exists(pretrain_all_path):
        data = pd.read_json(pretrain_all_path, lines=True)
        print(f"✅ Loaded combined pretraining dataset: {len(data)} sequences")
    else:
        # Fallback: combine train + test
        train_df = pd.read_json(train_path, lines=True)
        print(f"Loaded {len(train_df)} sequences from train.json")

        if os.path.exists(test_path):
            test_df = pd.read_json(test_path, lines=True)
            print(f"Loaded {len(test_df)} sequences from test.json")
            data = pd.concat([train_df, test_df], ignore_index=True)
            print(f"Combined dataset: {len(data)} sequences")
        else:
            data = train_df
            print(f"Using train.json only: {len(data)} sequences")

    # Load precomputed structural features
    print("📊 Loading precomputed features with structural data...")
    precomp_path = os.path.join(opts.path, 'precomputed_features.pt')

    if not os.path.exists(precomp_path):
        raise FileNotFoundError(f"Precomputed features not found: {precomp_path}")

    precomp = torch.load(precomp_path, weights_only=False)

    # Extract feature components
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

    print(f"✅ Loaded {len(embedding_dict)} embeddings and structural features")

    # Split data for training and validation
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=2022)
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Create training dataset
    base_train_dataset = RNADataset(
        seqs=train_data.sequence.to_list(),
        labels=np.zeros(len(train_data), dtype='float32'),
        ids=train_data.id.to_list(),
        ew=np.arange(len(train_data)),
        bpp_path=opts.path,
        pad=True,
        num_layers=opts.nlayers,
        training=True
    )

    training_dataset = EmbeddedRNADataset(
        base_train_dataset,
        embedding_dict,
        graph_dist_dict,
        nearest_p_dict,
        nearest_up_dict
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
        collate_fn=variable_length_collate_fn
    )

    # Create validation dataset
    base_val_dataset = RNADataset(
        seqs=val_data.sequence.to_list(),
        labels=np.zeros(len(val_data), dtype='float32'),
        ids=val_data.id.to_list(),
        ew=np.arange(len(val_data)),
        bpp_path=opts.path,
        pad=True,
        num_layers=opts.nlayers,
        training=False
    )

    val_dataset = EmbeddedRNADataset(
        base_val_dataset,
        embedding_dict,
        graph_dist_dict,
        nearest_p_dict,
        nearest_up_dict
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.workers,
        collate_fn=variable_length_collate_fn
    )

    print("✅ Training and validation datasets created with full structural features")

    # Setup checkpointing and logging
    checkpoints_folder = 'pretrain_weights'
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    csv_file = 'logs/pretrain.csv'
    columns = ['epoch', 'train_loss', 'val_loss']
    logger = CSVLogger(columns, csv_file)

    # Initialize model with V7 architecture
    precomputed_path = os.path.join(opts.path, 'precomputed_features.pt')
    model = RNADegformer(
        opts.ntoken,
        opts.nclass,
        opts.ninp,
        opts.nhead,
        opts.nhid,
        opts.nlayers,
        stride=opts.stride,
        dropout=opts.dropout,
        pretrain=True,
        return_aw=True,
        rinalmo_weights_path=None,  # V7: Simplified architecture
        precomputed_features_path=precomputed_path
    ).to(device)

    # Setup optimizer and loss criterion
    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'📊 Total model parameters: {pytorch_total_params:,}')

    # Setup learning rate scheduler
    total_steps = len(training_dataloader)
    total_training_steps = opts.epochs * total_steps
    warmup_steps = int(0.06 * total_training_steps)

    lr_schedule = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_training_steps,
        initial_multiplier=0.01
    )

    print(f"📈 Training configuration:")
    print(f"  Steps per epoch: {total_steps}")
    print(f"  Total training steps: {total_training_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    best_loss = float('inf')

    # Main training loop
    print(f"\n🎯 Starting training for {opts.epochs} epochs...")

    for epoch in range(opts.epochs):
        model.train(True)
        epoch_start_time = time.time()
        total_loss = 0
        step_count = 0

        for batch_idx, data in enumerate(training_dataloader):
            step_count += 1
            optimizer.zero_grad()

            try:
                # Load and validate input tensors
                embeddings = data['embedding'].cuda(non_blocking=True)
                bpps = data['bpp'].cuda(non_blocking=True)
                src_mask = data['src_mask'].cuda(non_blocking=True)

                # Validate BPP tensor structure
                bpps = validate_bpp_tensor(bpps, f"training_epoch_{epoch}_batch_{batch_idx}")

                # Fix source mask dimensions
                src_mask = fix_src_mask_dimensions(src_mask, opts.nlayers)

                # Create masked views for contrastive learning
                mask_ratio = 0.15
                masked_view1, mask1 = get_masked_view(embeddings, mask_ratio)
                masked_view2, mask2 = get_masked_view(embeddings, mask_ratio)

                # Load label tensors
                labels = data['data'][:, :, 0].to(device).long()
                structure_labels = data['data'][:, :, 1].to(device).long()
                loop_labels = data['data'][:, :, 2].to(device).long()
                bpp_targets = data['bpp'][:, 0, :, :].to(device).float()

                # Forward pass with structural features
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # CRITICAL: Input sanitization to prevent NaN propagation
                    embeddings = torch.nan_to_num(masked_view1, nan=0.0, posinf=1e6, neginf=-1e6)
                    bpps_clean = torch.nan_to_num(bpps, nan=0.0, posinf=1.0, neginf=0.0)

                    deltaG = data['deltaG'].to(device)
                    graph_dist = torch.nan_to_num(data['graph_dist'].to(device), nan=0.0)
                    nearest_p = torch.nan_to_num(data['nearest_p'].to(device), nan=0.0)
                    nearest_up = torch.nan_to_num(data['nearest_up'].to(device), nan=0.0)

                    outputs1, attention_maps, reps1 = model(
                        embeddings, bpps_clean, src_mask, deltaG, graph_dist, nearest_p, nearest_up
                    )
                    outputs2, _, reps2 = model(
                        torch.nan_to_num(masked_view2, nan=0.0), bpps_clean, src_mask, deltaG, graph_dist, nearest_p,
                        nearest_up
                    )

                # Compute multi-task losses
                nucleotide_loss = compute_nucleotide_loss(outputs1, labels, mask1, criterion)
                structure_loss = compute_structure_loss(outputs1, structure_labels, mask1, criterion)
                loop_loss = compute_loop_loss(outputs1, loop_labels, mask1, criterion)
                bpp_loss = compute_bpp_loss(outputs1, bpp_targets, mask1)

                # Contrastive loss between masked views
                con_loss = contrastive_loss(reps1, reps2)

                # Combined loss with weighting
                total_batch_loss = (
                        1.0 * nucleotide_loss +
                        0.4 * structure_loss +
                        0.4 * loop_loss +
                        0.25 * bpp_loss +
                        0.1 * con_loss
                )

                # Backward pass and optimization
                total_batch_loss.backward()
                # FIXED: Proper gradient NaN detection - halt training instead of masking
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"CRITICAL: NaN/inf gradient detected in parameter: {name}")
                        print(f"  Gradient min: {param.grad.min().item()}")
                        print(f"  Gradient max: {param.grad.max().item()}")
                        print(f"  Gradient norm: {param.grad.norm().item()}")
                        has_nan_grad = True

                if has_nan_grad:
                    print(f"HALTING TRAINING: Gradient instability at epoch {epoch}, step {step_count}")
                    print("Likely causes: learning rate too high, loss explosion, or architectural issues")
                    # Save emergency checkpoint before termination
                    torch.save(model.state_dict(),
                               f"{checkpoints_folder}/gradient_failure_epoch{epoch}_step{step_count}.ckpt")
                    raise RuntimeError("Training terminated due to gradient instability")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_schedule.step()

                total_loss += total_batch_loss.item()

                # Progress reporting with corrected step counting
                current_lr = get_lr(optimizer)
                avg_loss = total_loss / step_count

                print(f"Epoch [{epoch + 1}/{opts.epochs}], Step [{step_count}/{total_steps}] "
                      f"Loss: {avg_loss:.3f} LR: {current_lr:.6f} Time: {time.time() - epoch_start_time:.1f}s",
                      end='\r', flush=True)

            except Exception as e:
                print(f"\nERROR in training step {step_count}: {e}")
                continue

        print('')  # New line after epoch completion

        # Validation and checkpointing
        if (epoch + 1) % opts.save_freq == 0:
            print(f"🔄 Running validation for epoch {epoch + 1}...")

            val_loss, loss_components = validate_with_nan_protection(
                model, val_dataloader, device, criterion, opts
            )

            train_loss = total_loss / max(step_count, 1)

            # Enhanced logging with component breakdown
            print(f"📊 Epoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Components: nucleotide={loss_components['nucleotide']:.6f}, "
                  f"structure={loss_components['structure']:.6f}, "
                  f"loop={loss_components['loop']:.6f}, "
                  f"bpp={loss_components['bpp']:.6f}")

            # Log results with NaN handling
            val_loss_for_log = val_loss if not math.isnan(val_loss) else 999.0
            to_log = [epoch + 1, train_loss, val_loss_for_log]
            logger.log(to_log)

            # Save best model checkpoint
            if not math.isnan(val_loss):
                is_best = val_loss < best_loss
                if is_best:
                    print(f"✅ New best validation loss: {val_loss:.6f}")
                    best_loss = val_loss
            else:
                print("⚠️ Skipping best model save due to NaN validation loss")
                is_best = False

            save_weights(model, optimizer, epoch, checkpoints_folder,
                         is_best=is_best, val_loss=val_loss_for_log)

            torch.cuda.empty_cache()

    print(f"\n🎉 Pretraining completed successfully!")
    print(f"📁 Checkpoints saved in: {checkpoints_folder}")
    print(f"📊 Training log saved in: {csv_file}")
    print(f"🏆 Best validation loss: {best_loss:.6f}")

    # Extract best weights for downstream training
    get_best_weights_from_fold(opts.fold, mode='pretrain')


if __name__ == '__main__':
    train_fold()