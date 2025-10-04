import os
import torch
import torch.nn as nn
import time
from Functions_v5 import *
from Dataset_v5 import *
from X_Network_v5 import *
from LrScheduler import *
import Metrics
import numpy as np
from Logger import CSVLogger
import argparse
from ranger import Ranger

try:
    # from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split, KFold


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
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
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
    parser.add_argument('--error_beta', type=float, default=5, help='number of workers for dataloader')
    parser.add_argument('--error_alpha', type=float, default=0, help='number of workers for dataloader')
    parser.add_argument('--noise_filter', type=float, default=0.25, help='number of workers for dataloader')
    parser.add_argument('--weight_path', type=str, default='.', help='weight path')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--std_threshold', type=float, default=0.1,
                        help='discard pseudo labels with std greater than this')
    parser.add_argument('--std_eps', type=float, default=1e-6,
                        help='epsilon for std based weighting')
    opts = parser.parse_args()
    return opts


def train_fold():
    # get arguments
    opts = get_args()

    # gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate datasets
    json_path = os.path.join(opts.path, 'train.json')

    json = pd.read_json(json_path, lines=True)
    json = json[json.signal_to_noise > opts.noise_filter]
    ids = np.asarray(json.id.to_list())

    error_weights = get_errors(json)
    error_weights = opts.error_alpha + np.exp(-error_weights * opts.error_beta)
    train_indices, val_indices = get_train_val_indices(json, opts.fold, SEED=2020, nfolds=opts.nfolds)

    _, labels = get_data(json)
    sequences = np.asarray(json.sequence)
    train_seqs = list(sequences[train_indices])
    val_seqs = sequences[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    train_ids = ids[train_indices]
    val_ids = ids[val_indices]
    train_ew = error_weights[train_indices]
    val_ew = error_weights[val_indices]

    train_labels = np.pad(train_labels, ((0, 0), (0, 23), (0, 0)), constant_values=0)
    train_ew = np.pad(train_ew, ((0, 0), (0, 23), (0, 0)), constant_values=0)

    n_train = len(train_labels)

    test_json_path = os.path.join(opts.path, 'test.json')
    test = pd.read_json(test_json_path, lines=True)

    with open(f'../pseudo_labels/pseudo_labels_fold{opts.fold}.p', 'rb') as f:
        long_preds, long_stds, short_preds, short_stds = pickle.load(f)

    # Pseudo-labels now come pre-formatted as (N, 91, 5) from updated pseudo_predict_v5.py
    # No need for additional slicing or padding
    print(f"Loaded pseudo-labels shapes - Long: {long_preds.shape}, Short: {short_preds.shape}")

    ls_indices = test.seq_length == 130
    long_data = test[ls_indices]
    long_ids = np.asarray(long_data.id.to_list())
    long_sequences = np.asarray(long_data.sequence.to_list())

    # Compute inverse-std weights and filter out high-uncertainty positions
    # clamp standard deviations so we never invert below the threshold
    long_mask = long_stds <= opts.std_threshold
    bounded_stds = np.maximum(long_stds, opts.std_threshold)
    long_weights = long_mask * (1.0 / (bounded_stds + opts.std_eps))

    ss_indices = test.seq_length == 107
    short_data = test[ss_indices]
    short_ids = np.asarray(short_data.id)
    short_sequences = np.asarray(short_data.sequence)

    short_mask = short_stds <= opts.std_threshold
    short_mask[:, 68:] = 0  # Zero out positions beyond 68 for short sequences
    bounded_short_stds = np.maximum(short_stds, opts.std_threshold)
    short_weights = short_mask * (1.0 / (bounded_short_stds + opts.std_eps))

    # Ensure all arrays have consistent 91-position format (no additional padding needed)
    assert short_preds.shape[1] == 91, f"Expected short_preds to have 91 positions, got {short_preds.shape[1]}"
    assert long_preds.shape[1] == 91, f"Expected long_preds to have 91 positions, got {long_preds.shape[1]}"
    assert train_labels.shape[1] == 91, f"Expected train_labels to have 91 positions, got {train_labels.shape[1]}"

    # Directly concatenate without additional padding since everything is now 91 positions
    train_seqs = np.concatenate([train_seqs, short_sequences, long_sequences])
    train_labels = np.concatenate([train_labels, short_preds, long_preds], axis=0)
    train_ids = np.concatenate([train_ids, short_ids, long_ids], axis=0)

    # Concatenate weights (all should be 91 positions already)
    train_ew = np.concatenate([train_ew, short_weights, long_weights], axis=0)

    # Verify final shapes explicitly (authoritative debug statements)
    print("Final train_labels shape after concatenation:", train_labels.shape)
    print("Final train_ids shape after concatenation:", train_ids.shape)
    print("Final train_ew shape after concatenation:", train_ew.shape)

    pl_dataset = RNADataset(train_seqs[n_train:], train_labels[n_train:], train_ids[n_train:], train_ew[n_train:],
                            opts.path, pad=True, k=opts.kmers[0])
    val_dataset = RNADataset(val_seqs, val_labels, val_ids, val_ew, opts.path, training=True, k=opts.kmers[0])

    finetune_dataset = RNADataset(train_seqs[:n_train], train_labels[:n_train, :68], train_ids[:n_train],
                                  train_ew[:n_train, :68], opts.path, k=opts.kmers[0])

    pl_dataloader = DataLoader(pl_dataset, batch_size=opts.batch_size,
                               shuffle=True, num_workers=opts.workers,
                               collate_fn=variable_length_collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size * 2,
                                shuffle=False, num_workers=opts.workers,
                                collate_fn=variable_length_collate_fn)

    finetune_dataloader = DataLoader(finetune_dataset, batch_size=opts.batch_size // 2,
                                     shuffle=True, num_workers=opts.workers,
                                     collate_fn=variable_length_collate_fn)

    # checkpointing
    os.system('mkdir weights')
    checkpoints_folder = 'weights/checkpoints_fold{}_pl'.format((opts.fold))
    os.system('mkdir logs')
    csv_file = 'logs/log_pl_fold{}.csv'.format((opts.fold))
    columns = ['epoch', 'train_loss',
               'val_loss']
    logger = CSVLogger(columns, csv_file)

    # build model and logger
    rinalmo_weights_path = '/home/slater/RiNALMo/weights/rinalmo_micro_pretrained.pt'
    model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                         opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers, stride=opts.stride,
                         dropout=opts.dropout, rinalmo_weights_path=rinalmo_weights_path).to(device)

    criterion = weighted_MCRMSE

    # Mixed precision initialization
    opt_level = 'O1'
    model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))

    cos_epoch = int(opts.epochs * 0.75) - 1
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             (opts.epochs - cos_epoch) * len(finetune_dataloader))
    for epoch in range(opts.epochs):
        model.train(True)
        t = time.time()
        total_loss = 0
        optimizer.zero_grad()
        step = 0
        if epoch > cos_epoch:
            dataloader = finetune_dataloader
        else:
            dataloader = pl_dataloader

        for data in dataloader:
            step += 1
            lr = get_lr(optimizer)
            src = data['embedding'].to(device)

            labels = data['labels'].to(device)

            bpps = data['bpp'].to(device)
            if bpps.dim() == 5 and bpps.shape[1] == 1:
                bpps = bpps[:, 0, :, :, :]

            src_mask = data['src_mask'].to(device)
            if src_mask.dim() == 3 and src_mask.shape[1] == 1:
                src_mask = src_mask[:, 0, :]

            labels = labels.to(device)  # .float()
            src_mask = data['src_mask'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(src, bpps, src_mask)
            ew = data['ew'].to(device)

            loss = criterion(output[:, :labels.shape[1]], labels, ew).mean()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                  .format(epoch + 1, opts.epochs, step, len(dataloader), total_loss / step, lr, time.time() - t),
                  end='\r', flush=True)  # total_loss/(step+1)
            # break
            if epoch > cos_epoch:
                lr_schedule.step()

        print('')
        train_loss = total_loss / (step + 1)

        torch.cuda.empty_cache()
        val_loss = -1  # validation skipped this epoch
        if (epoch + 1) % opts.val_freq == 0 and epoch > cos_epoch:
            val_loss = validate(model, device, val_dataloader, batch_size=opts.batch_size)

        to_log = [epoch + 1, train_loss, val_loss]
        logger.log(to_log)

        if (epoch + 1) % opts.save_freq == 0:
            save_weights(model, optimizer, epoch, checkpoints_folder)


if __name__ == '__main__':
    train_fold()