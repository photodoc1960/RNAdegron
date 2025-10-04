import os
import torch
import torch.nn as nn
import time
from Functions_v3 import *
from Dataset import *
from Network_v3 import *
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
    opts = parser.parse_args()
    return opts

# Add this function definition to the file
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

def train_fold():
    # get arguments
    opts = get_args()

    # gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate datasets
    json_path = os.path.join(opts.path, 'train.json')
    json = pd.read_json(json_path, lines=True)
    train_ids = json.id.to_list()

    json_path = os.path.join(opts.path, 'test.json')
    test = pd.read_json(json_path, lines=True)

    # aug_test=test
    # dataloader
    # ls_indices=test.seq_length==130

    data = test  # [ls_indices]

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=2022)

    ids = np.asarray(train_data.id.to_list())
    training_dataset = RNADataset(train_data.sequence.to_list(), np.zeros(len(train_data)), ids,
                                  np.arange(len(train_data)), opts.path, pad=True, k=opts.kmers[0])
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size,
                                     shuffle=True, num_workers=opts.workers, collate_fn=variable_length_collate_fn)

    val_ids = np.asarray(val_data.id.to_list())
    val_dataset = RNADataset(val_data.sequence.to_list(), np.zeros(len(val_data)), val_ids, np.arange(len(val_data)),
                             opts.path, pad=True, k=opts.kmers[0])
    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size,
                                shuffle=False, num_workers=opts.workers, collate_fn=variable_length_collate_fn)

    # checkpointing
    checkpoints_folder = 'pretrain_weights'

    os.system('mkdir logs')

    csv_file = 'logs/pretrain.csv'.format((opts.fold))
    columns = ['epoch', 'train_loss', 'val_loss']
    logger = CSVLogger(columns, csv_file)

    # build model and logger
    model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                         opts.nlayers, False, kmers=[1], stride=opts.stride,
                         dropout=opts.dropout, pretrain=True,
                         rinalmo_weights_path='/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_micro_pretrained.pt').to(device)
    print(f"Model class: {type(model)}")

    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss()
    # Mixed precision initialization
    opt_level = 'O1'
    # model = nn.DataParallel(model) - not needed for single GPU

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))
    # Add this to restore the missing variable
    # cos_epoch = int(opts.epochs * 0.75)
    total_steps = len(training_dataloader)
    # training loop
    step_counter = 0
    total_training_steps = opts.epochs * len(training_dataloader)
    warmup_steps = int(0.06 * total_training_steps)
    lr_schedule = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_training_steps,
        initial_multiplier=0.01) # Start at 1% of base learning rate
    best_loss = float('inf')

    for epoch in range(opts.epochs):
        model.train(True)
        t = time.time()
        total_loss = 0
        optimizer.zero_grad()
        train_preds = []
        ground_truths = []
        step = 0
        for data in training_dataloader:
            # for step in range(1):
            step += 1
            # Use RiNALMo embeddings directly
            rinalmo_emb = data['embedding'].to(device)  # [B, L, 480]
            src = data['data'].to(device).long()  # [B, L, 3] tokenized sequence
            bpps = data['bpp'].to(device)
            src_mask = data['src_mask'].to(device)

            labels = src[:, :, 0]
            structure_labels = src[:, :, 1]
            loop_labels = src[:, :, 2]
            bpp_targets = data['bpp'][:, 0, :, :].to(device).float()

            # Mask or mutate the tokenized input
            if np.random.uniform() > 0.5:
                src_masked, _ = mutate_rna_input(src)
            else:
                src_masked, _ = mask_rna_input(src)

            # Forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs, _ = model(src_masked, rinalmo_emb, bpps, src_mask)

            mask_positions = (src[:, :, 0] != src_masked[:, :, 0])

            masked_labels = src[:, :, 0][mask_positions]
            outputs_masked = outputs['nucleotide'][mask_positions]

            # Keep only token IDs in [5, 6, 7, 8] → valid nucleotide tokens for MLM
            valid_pos = (masked_labels >= 5) & (masked_labels <= 8)
            if valid_pos.sum() == 0:
                continue

            masked_labels = masked_labels[valid_pos] - 5  # map [5,6,7,8] → [0,1,2,3]
            outputs_masked = outputs_masked[valid_pos]

            # Sanity check
            assert masked_labels.min() >= 0 and masked_labels.max() <= 3, \
                f"Invalid label values after remapping: {masked_labels.unique()}"

            # Compute loss
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
            bpp_preds = outputs['bpp'].squeeze(-1)  # [B, L]
            bpp_diag_target = torch.stack([
                bpp_targets[i, torch.arange(bpp_targets.size(-1)), torch.arange(bpp_targets.size(-1))]
                for i in range(bpp_targets.size(0))
            ]).to(bpp_preds.dtype)  # 🔥 ensure float16 if autocast is active

            # Remove NaNs in BPP targets before computing loss
            bpp_masked_pred = bpp_preds[mask_positions]
            bpp_masked_target = bpp_diag_target[mask_positions]

            # Filter out any NaNs (common with low-complexity sequences or padding)
            valid_bpp = ~torch.isnan(bpp_masked_target)
            if valid_bpp.sum() > 0:
                bpp_loss = F.mse_loss(bpp_masked_pred[valid_bpp], bpp_masked_target[valid_bpp])
            else:
                bpp_loss = torch.tensor(0.0, device=bpp_preds.device, dtype=bpp_preds.dtype)

            # === Total multitask loss ===
            loss = (
                    1.0 * nucleotide_loss +
                    0.4 * structure_loss +
                    0.4 * loop_loss +
                    0.25 * bpp_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_schedule.step()  # Unconditional stepping
            optimizer.zero_grad()
            # Get the current learning rate after scheduler update
            lr = get_lr(optimizer)

            total_loss += loss

            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                  .format(epoch + 1, opts.epochs, step + 1, total_steps, total_loss / (step + 1), lr, time.time() - t),
                  end='\r', flush=True)  # total_loss/(step+1)

        print('')

        if (epoch + 1) % opts.save_freq == 0:
            val_loss = []
            for _ in tqdm(range(5)):
                for data in val_dataloader:
                    bpps = data['bpp'].to(device)
                    src_mask = data['src_mask'].to(device)
                    rinalmo_emb = data['embedding'].to(device)
                    src = data['data'].to(device).long()

                    structure_labels = src[:, :, 1]
                    loop_labels = src[:, :, 2]
                    bpp_targets = data['bpp'][:, 0, :, :].to(device).float()

                    # Apply dynamic token ID masking
                    if np.random.uniform() > 0.5:
                        src_masked, _ = mutate_rna_input(src)
                    else:
                        src_masked, _ = mask_rna_input(src)

                    # Extract and plot attention map: layer 0, head 0, sequence 0
                    layer_idx = 0
                    head_idx = 0
                    seq_idx = 0
                    with torch.no_grad():
                        outputs, attention_maps = model(src_masked, rinalmo_emb, bpps, src_mask)

                        if (epoch + 1) % 10 == 0:
                            plot_all_heads_in_layer_with_bpp(
                                attention_maps=attention_maps,
                                bpp_targets=bpp_targets,
                                layer_idx=0,
                                seq_idx=0,
                                threshold=0.5,
                                save_path="attn_bpp_grids"
                            )

                        if (epoch + 1) % 10 == 0 and seq_idx == 0:
                            plot_attention_with_bpp(
                                attention_maps=attention_maps,
                                bpp_targets=bpp_targets,
                                layer_idx=0,
                                head_idx=0,
                                seq_idx=0,
                                threshold=0.5,
                                save_path="attn_bpp_plots"
                            )

                    # Correct labels based on current batch
                    labels = src[:, :, 0]

                    # Build mask_positions
                    mask_positions = (src[:, :, 0] != src_masked[:, :, 0])

                    # Masked selections
                    masked_labels = src[:, :, 0][mask_positions]
                    outputs_masked = outputs['nucleotide'][mask_positions]

                    valid_pos = (masked_labels >= 5) & (masked_labels <= 8)
                    if valid_pos.sum() == 0:
                        continue

                    masked_labels = masked_labels[valid_pos] - 5
                    outputs_masked = outputs_masked[valid_pos]

                    # Nucleotide loss
                    nucleotide_loss = criterion(outputs_masked.reshape(-1, 4), masked_labels.reshape(-1))

                    # Structure loss
                    structure_preds = outputs['structure'][mask_positions]
                    structure_labels_masked = structure_labels[mask_positions]
                    structure_valid = structure_labels_masked != 14
                    structure_loss = criterion(structure_preds[structure_valid],
                                               structure_labels_masked[structure_valid])

                    # Loop loss
                    loop_preds = outputs['loop'][mask_positions]
                    loop_labels_masked = loop_labels[mask_positions]
                    loop_valid = loop_labels_masked != 14
                    loop_loss = criterion(loop_preds[loop_valid], loop_labels_masked[loop_valid])

                    # BPP loss
                    bpp_preds = outputs['bpp'].squeeze(-1)
                    bpp_diag_target = torch.stack([
                        bpp_targets[i, torch.arange(bpp_targets.size(-1)), torch.arange(bpp_targets.size(-1))]
                        for i in range(bpp_targets.size(0))
                    ]).to(bpp_preds.dtype)

                    bpp_masked_pred = bpp_preds[mask_positions]
                    bpp_masked_target = bpp_diag_target[mask_positions]

                    valid_bpp = ~torch.isnan(bpp_masked_target)
                    if valid_bpp.sum() > 0:
                        bpp_loss = F.mse_loss(bpp_masked_pred[valid_bpp], bpp_masked_target[valid_bpp])
                    else:
                        bpp_loss = torch.tensor(0.0, device=bpp_preds.device, dtype=bpp_preds.dtype)

                    # Final multitask loss
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
            to_log = [epoch + 1, train_loss, val_loss]
            logger.log(to_log)

            is_best = val_loss < best_loss
            if is_best:
                print(f"new best_loss found at epoch {epoch}: {val_loss}")
                best_loss = val_loss

            # Save checkpoint with best flag
            save_weights(model, optimizer, epoch, checkpoints_folder, is_best=is_best, val_loss=val_loss)

    get_best_weights_from_fold(opts.fold, mode='pretrain')


if __name__ == '__main__':
    train_fold()