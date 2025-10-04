import os
import torch
import torch.nn as nn
import time
import math
from Functions import *
from Dataset import *
from X_Network import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse
from ranger import Ranger
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # Set backend to non-interactive (no Tk)
import matplotlib.pyplot as plt
import shutil

import seaborn as sns
from visualization import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=4,
                        help='number of tokens to represent DNA nucleotides (should always be 4)')
    parser.add_argument('--nclass', type=int, default=5, help='number of classes from the linear decoder')
    parser.add_argument('--ninp', type=int, default=512, help='ninp for transformer encoder')
    parser.add_argument('--nhead', type=int, default=8, help='nhead for transformer encoder')
    parser.add_argument('--nhid', type=int, default=2048, help='nhid for transformer encoder')
    parser.add_argument('--nlayers', type=int, default=6, help='nlayers for transformer encoder')
    parser.add_argument('--save_freq', type=int, default=10, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--warmup_steps', type=int, default=3200, help='training schedule warmup steps')
    parser.add_argument('--lr_scale', type=float, default=0.1, help='learning rate scale')
    parser.add_argument('--nmute', type=int, default=18, help='number of mutations during training')
    parser.add_argument('--kmers', type=int, nargs='+', default=[2, 3, 4, 5, 6], help='k-mers to be aggregated')
    # parser.add_argument('--kmer_aggregation', type=bool, default=True, help='k-mers to be aggregated')
    parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--stride', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--viral_loss_weight', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--force_regenerate', action='store_true', help='Force regeneration of cached data')
    opts = parser.parse_args()
    return opts


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0,
                                    initial_multiplier=0.01, last_epoch=-1):
    """
    Creates a learning rate schedule with linear warmup and cosine annealing.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Start at initial_multiplier and linearly increase to 1.0
            warmup_pct = float(current_step) / float(max(1, num_warmup_steps))
            return initial_multiplier + (1.0 - initial_multiplier) * warmup_pct

        # Standard cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def fix_src_mask_dimensions(src_mask, nlayers):
    """
    Ensures src_mask has the correct dimensions for the transformer model.

    Args:
        src_mask (torch.Tensor): Source mask tensor
        nlayers (int): Number of layers in the transformer model

    Returns:
        torch.Tensor: Corrected source mask tensor
    """
    # If src_mask doesn't have batch dimension, add it
    if src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(0)

    # If src_mask doesn't have layer dimension, add it
    if src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(1)

    # If layer dimension doesn't match nlayers, fix it
    if src_mask.dim() == 3 and src_mask.size(1) != nlayers:
        corrected_mask = torch.zeros(
            (src_mask.size(0), nlayers, src_mask.size(2)),
            device=src_mask.device,
            dtype=src_mask.dtype
        )

        # Copy existing data for available layers
        min_layers = min(src_mask.size(1), nlayers)
        corrected_mask[:, :min_layers, :] = src_mask[:, :min_layers, :]

        # If src_mask has fewer layers than needed, copy the last layer
        if src_mask.size(1) < nlayers:
            corrected_mask[:, min_layers:, :] = src_mask[:, -1:, :].repeat(1, nlayers - min_layers, 1)

        src_mask = corrected_mask

    return src_mask


def train_fold():
    # Get arguments
    opts = get_args()
    print(f"Creating datasets with nlayers={opts.nlayers}")

    # Clear cached mask data if requested
    if opts.force_regenerate:
        mask_cache_path = os.path.join(opts.path, 'precomputed_masks')
        if os.path.exists(mask_cache_path):
            print(f"Removing cached masks from {mask_cache_path}")
            shutil.rmtree(mask_cache_path)
        os.makedirs(mask_cache_path, exist_ok=True)

    # GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load from pretrain_all.json if it exists (train + test + new_sequences)
    pretrain_all_path = os.path.join(opts.path, 'pretrain_all.json')
    train_path = os.path.join(opts.path, 'train.json')
    test_path = os.path.join(opts.path, 'test.json')

    if os.path.exists(pretrain_all_path):
        data = pd.read_json(pretrain_all_path, lines=True)
        print(f"✅ Loaded combined pretraining dataset: {len(data)} sequences from pretrain_all.json")
    else:
        # Fallback: combine train + test
        train_df = pd.read_json(train_path, lines=True)

        if os.path.exists(test_path):
            test_df = pd.read_json(test_path, lines=True)
            print(f"Loaded {len(test_df)} sequences from test.json")
            data = pd.concat([train_df, test_df], ignore_index=True)
            print(f"Combined dataset has {len(data)} total sequences.")
        else:
            data = train_df
            print(f"Using train.json only: {len(data)} sequences")

    # ✅ Filter sequences longer than 130 bases before any dataset use
    max_len = 130
    original_len = len(data)
    data = data[data.sequence.str.len() <= max_len].reset_index(drop=True)
    print(f"Filtered {original_len - len(data)} sequences longer than {max_len} bases.")

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=2022)

    ids = np.asarray(train_data.id.to_list())
    training_dataset = RNADataset(
        train_data.sequence.to_list(),
        np.zeros(len(train_data)),
        ids,
        np.arange(len(train_data)),
        opts.path,
        pad=True,
        k=opts.kmers[0],
        num_layers=opts.nlayers
    )
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
        collate_fn=variable_length_collate_fn
    )

    val_ids = np.asarray(val_data.id.to_list())
    val_dataset = RNADataset(
        val_data.sequence.to_list(),
        np.zeros(len(val_data)),
        val_ids,
        np.arange(len(val_data)),
        opts.path,
        pad=True,
        k=opts.kmers[0],
        num_layers=opts.nlayers
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.workers,
        collate_fn=variable_length_collate_fn
    )

    # Checkpointing
    checkpoints_folder = 'pretrain_weights'
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    csv_file = 'logs/pretrain.csv'
    columns = ['epoch', 'train_loss', 'val_loss']
    logger = CSVLogger(columns, csv_file)

    # Build model
    model = RNADegformer(
        opts.ntoken,
        opts.nclass,
        opts.ninp,
        opts.nhead,
        opts.nhid,
        opts.nlayers,
        False,
        kmers=[1],
        stride=opts.stride,
        dropout=opts.dropout,
        pretrain=True,
        return_aw=True,  # Ensure this is True to get attention weights
        rinalmo_weights_path='/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_micro_pretrained.pt'
    ).to(device)

    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {pytorch_total_params}')

    total_steps = len(training_dataloader)
    total_training_steps = opts.epochs * len(training_dataloader)
    warmup_steps = int(0.06 * total_training_steps)

    lr_schedule = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_training_steps,
        initial_multiplier=0.01
    )

    best_loss = float('inf')

    # Training loop
    for epoch in range(opts.epochs):
        model.train(True)
        t = time.time()
        total_loss = 0
        optimizer.zero_grad()
        step = 0

        for data in training_dataloader:
            step += 1

            # Get data
            embeddings = data['embedding'].cuda(non_blocking=True)
            bpps = data['bpp'].cuda(non_blocking=True)
            src_mask = data['src_mask'].cuda(non_blocking=True)

            # Print source mask shape for debugging
            # print(f"Original src_mask shape: {src_mask.shape}")

            # Fix src_mask dimensions if needed
            src_mask = fix_src_mask_dimensions(src_mask, opts.nlayers)
            # print(f"Fixed src_mask shape: {src_mask.shape}")

            # Construct synthetic MLM targets
            mask_ratio = 0.15
            seq_len = embeddings.shape[1]
            batch_size = embeddings.shape[0]
            mask_positions = torch.rand(batch_size, seq_len, device=embeddings.device) < mask_ratio

            # Get labels
            labels = data['data'][:, :, 0].to(device).long()
            structure_labels = data['data'][:, :, 1].to(device).long()
            loop_labels = data['data'][:, :, 2].to(device).long()
            bpp_targets = data['bpp'][:, 0, :, :].to(device).float()

            # Create masked embeddings
            masked_embeddings = embeddings.clone()
            masked_embeddings[mask_positions] = 0.0

            # Forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs, attention_maps = model(masked_embeddings, bpps, src_mask)

            # Process outputs for nucleotide prediction
            masked_labels = labels[mask_positions]
            outputs_masked = outputs['nucleotide'][mask_positions]

            # Keep only valid nucleotide tokens
            valid_pos = (masked_labels >= 5) & (masked_labels <= 8)
            if valid_pos.sum() == 0:
                continue

            masked_labels = masked_labels[valid_pos] - 5  # map [5,6,7,8] → [0,1,2,3]
            outputs_masked = outputs_masked[valid_pos]

            # Sanity check
            assert masked_labels.min() >= 0 and masked_labels.max() <= 3, \
                f"Invalid label values after remapping: {masked_labels.unique()}"

            # Compute losses
            # === Nucleotide MLM Loss ===
            nucleotide_loss = criterion(outputs_masked.reshape(-1, 4), masked_labels.reshape(-1))

            # === Structure MLM ===
            structure_preds = outputs['structure'][mask_positions]
            structure_labels_masked = structure_labels[mask_positions]
            structure_valid = structure_labels_masked != 14
            structure_loss = criterion(structure_preds[structure_valid], structure_labels_masked[structure_valid])

            # === Loop MLM ===
            loop_preds = outputs['loop'][mask_positions]
            loop_labels_masked = loop_labels[mask_positions]
            loop_valid = loop_labels_masked != 14
            loop_loss = criterion(loop_preds[loop_valid], loop_labels_masked[loop_valid])

            # === BPP Diagonal Regression ===
            bpp_preds = outputs['bpp'].squeeze(-1)
            bpp_diag_target = torch.stack([
                bpp_targets[i, torch.arange(bpp_targets.size(-1)), torch.arange(bpp_targets.size(-1))]
                for i in range(bpp_targets.size(0))
            ]).to(bpp_preds.dtype)

            # Handle NaNs in BPP targets
            bpp_masked_pred = bpp_preds[mask_positions]
            bpp_masked_target = bpp_diag_target[mask_positions]
            valid_bpp = ~torch.isnan(bpp_masked_target)

            if valid_bpp.sum() > 0:
                bpp_loss = F.mse_loss(bpp_masked_pred[valid_bpp], bpp_masked_target[valid_bpp])
            else:
                bpp_loss = torch.tensor(0.0, device=bpp_preds.device, dtype=bpp_preds.dtype)

            # Total loss
            loss = (
                    1.0 * nucleotide_loss +
                    0.4 * structure_loss +
                    0.4 * loop_loss +
                    0.25 * bpp_loss
            )

            # Optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()

            # Get current learning rate
            lr = get_lr(optimizer)

            total_loss += loss

            print(f"Epoch [{epoch + 1}/{opts.epochs}], Step [{step + 1}/{total_steps}] "
                  f"Loss: {total_loss / (step + 1):.3f} Lr:{lr:.6f} Time: {time.time() - t:.1f}",
                  end='\r', flush=True)

        print('')

        # Validation and checkpointing
        if (epoch + 1) % opts.save_freq == 0:
            val_loss = []
            for _ in tqdm(range(5)):
                for data in val_dataloader:
                    bpps = data['bpp'].cuda(non_blocking=True)
                    src_mask = data['src_mask'].cuda(non_blocking=True)

                    # Fix src_mask dimensions
                    src_mask = fix_src_mask_dimensions(src_mask, opts.nlayers)

                    embeddings = data['embedding'].cuda(non_blocking=True)
                    labels = data['data'][:, :, 0].to(device).long()
                    structure_labels = data['data'][:, :, 1].to(device).long()
                    loop_labels = data['data'][:, :, 2].to(device).long()
                    bpp_targets = data['bpp'][:, 0, :, :].to(device).float()

                    # Apply masking
                    mask_ratio = 0.15
                    batch_size, seq_len, _ = embeddings.shape
                    mask_positions = torch.rand(batch_size, seq_len, device=embeddings.device) < mask_ratio
                    masked_embeddings = embeddings.clone()
                    masked_embeddings[mask_positions] = 0.0

                    # Visualization settings
                    layer_idx = 0
                    head_idx = 0
                    seq_idx = 0
                    # Forward pass
                    with torch.no_grad():
                        outputs, attention_maps = model(masked_embeddings, bpps, src_mask)

                        # Plot attention maps
                        if (epoch + 1) % 10 == 0:
                            try:
                                plot_all_heads_in_layer_with_bpp(
                                    attention_maps=attention_maps,
                                    bpp_targets=bpp_targets,
                                    layer_idx=0,
                                    seq_idx=0,
                                    threshold=0.5,
                                    save_path="attn_bpp_grids"
                                )

                                if seq_idx == 0:
                                    plot_attention_with_bpp(
                                        attention_maps=attention_maps,
                                        bpp_targets=bpp_targets,
                                        layer_idx=0,
                                        head_idx=0,
                                        seq_idx=0,
                                        threshold=0.5,
                                        save_path="attn_bpp_plots"
                                    )
                            except Exception as e:
                                print(f"Error plotting attention maps: {e}")

                    # Plot attention map
                    try:
                        attn = attention_maps[layer_idx][seq_idx, head_idx].detach().cpu().numpy()
                        plt.figure(figsize=(6, 5))
                        sns.heatmap(attn, cmap='Greys')
                        plt.title(f"Attention L{layer_idx + 1} H{head_idx + 1}")
                        plt.xlabel("Key Position")
                        plt.ylabel("Query Position")
                        plt.tight_layout()
                        plt.savefig(f"attn_L{layer_idx + 1}_H{head_idx + 1}.png")
                        plt.close()
                    except Exception as e:
                        print(f"Error plotting attention heatmap: {e}")

                    # Process masked tokens
                    masked_labels = labels[mask_positions]
                    valid_pos = (masked_labels >= 5) & (masked_labels <= 8)
                    if valid_pos.sum() == 0:
                        continue

                    masked_labels = masked_labels[valid_pos] - 5
                    outputs_masked = outputs['nucleotide'][mask_positions][valid_pos]

                    # Validate labels
                    assert masked_labels.min() >= 0 and masked_labels.max() <= 3, \
                        f"Label remapping failed (VALIDATION). Got range: {masked_labels.min()} to {masked_labels.max()}"

                    # Calculate losses
                    nucleotide_loss = criterion(outputs_masked.reshape(-1, 4), masked_labels.reshape(-1))

                    structure_preds = outputs['structure'][mask_positions]
                    structure_labels_masked = structure_labels[mask_positions]
                    structure_valid = structure_labels_masked != 14

                    assert structure_labels_masked[structure_valid].min() >= 0, \
                        f"structure label < 0: {structure_labels_masked[structure_valid].min().item()}"
                    assert structure_labels_masked[structure_valid].max() < 10, \
                        f"structure label >= 10: {structure_labels_masked[structure_valid].max().item()}"

                    structure_loss = criterion(structure_preds[structure_valid],
                                               structure_labels_masked[structure_valid])

                    loop_preds = outputs['loop'][mask_positions]
                    loop_labels_masked = loop_labels[mask_positions]
                    loop_valid = loop_labels_masked != 14

                    assert loop_labels_masked[loop_valid].min() >= 0, \
                        f"loop label < 0: {loop_labels_masked[loop_valid].min().item()}"
                    assert loop_labels_masked[loop_valid].max() < 7, \
                        f"loop label >= 7: {loop_labels_masked[loop_valid].max().item()}"

                    loop_loss = criterion(loop_preds[loop_valid], loop_labels_masked[loop_valid])

                    bpp_preds = outputs['bpp'].squeeze(-1)
                    bpp_diag_target = torch.stack([
                        bpp_targets[i, torch.arange(bpp_targets.size(-1)), torch.arange(bpp_targets.size(-1))]
                        for i in range(bpp_targets.size(0))
                    ])
                    bpp_loss = F.mse_loss(bpp_preds[mask_positions], bpp_diag_target[mask_positions])

                    loss = (
                            1.0 * nucleotide_loss +
                            0.4 * structure_loss +
                            0.4 * loop_loss +
                            0.25 * bpp_loss
                    )
                    val_loss.append(loss.item())

            val_loss = np.mean(val_loss)
            train_loss = total_loss / (step + 1)
            torch.cuda.empty_cache()

            # Log results
            to_log = [epoch + 1, train_loss, val_loss]
            logger.log(to_log)

            # Save best model
            is_best = val_loss < best_loss
            if is_best:
                print(f"New best_loss found at epoch {epoch}: {val_loss}")
                best_loss = val_loss

            save_weights(model, optimizer, epoch, checkpoints_folder, is_best=is_best, val_loss=val_loss)

    get_best_weights_from_fold(opts.fold, mode='pretrain')


if __name__ == '__main__':
    train_fold()