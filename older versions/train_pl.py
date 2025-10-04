import os
import torch
import torch.nn as nn
import pickle
import time
import pandas as pd
import numpy as np
from Functions import *
from Dataset import *
from X_Network import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse
from ranger import Ranger
from tqdm import tqdm
# token2int definition (matches Dataset.py and Functions.py)
token2int = {x:i for i, x in enumerate('ACGU().BEHIMSX')}

try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split, KFold

def precompute_bpps(sequences, seq_ids, bpps_save_dir):
    os.makedirs(bpps_save_dir, exist_ok=True)
    for seq, seq_id in tqdm(zip(sequences, seq_ids), total=len(sequences), desc="Precomputing BPPS matrices"):
        bpps_path = os.path.join(bpps_save_dir, f'{seq_id}.npy')
        if not os.path.exists(bpps_path):
            bpps_matrix = generate_bpp_matrix(seq)
            np.save(bpps_path, bpps_matrix)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=21, help='number of tokens (must match previous scripts)')
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
    parser.add_argument('--kmers', type=int, nargs='+', default=[2,3,4,5,6], help='k-mers to be aggregated')
    #parser.add_argument('--kmer_aggregation', type=bool, default=True, help='k-mers to be aggregated')
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
    parser.add_argument('--rinalmo_weights_path', type=str,
                        default='/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_micro_pretrained.pt',
                        help='Path to pretrained RiNALMo weights')

    opts = parser.parse_args()
    return opts

class BPPSafeDatasetWithMask(torch.utils.data.Dataset):
    def __init__(self, original_dataset, num_layers):
        self.original_dataset = original_dataset
        self.num_layers = num_layers

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = self.original_dataset[idx]

        # Reshape src_mask explicitly to match the expected dimensions
        if sample['src_mask'].ndim == 2:
            sample['src_mask'] = np.repeat(sample['src_mask'], self.num_layers, axis=0)

        return sample

class RNADatasetPrecomputed(Dataset):
    def __init__(self, sequences, labels, ids, error_weights, bpps_dir, pad=True, k=5):
        self.sequences = sequences
        self.labels = labels
        self.ids = ids
        self.ew = error_weights
        self.bpps_dir = bpps_dir
        self.pad = pad
        self.k = k

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_id = self.ids[idx]
        label = self.labels[idx]
        ew = self.ew[idx]

        # Load precomputed embedding
        embedding_path = os.path.join(self.bpps_dir, '..', 'precomputed_embeddings', f"{seq_id}_emb.pt")
        embedding = torch.load(embedding_path).squeeze(0)  # shape: [L, D]

        # Load BPP matrix and set seq_len from it
        bpps_path = os.path.join(self.bpps_dir, f'{seq_id}.npy')
        bpp_matrix = np.load(bpps_path).astype(np.float32)  # shape: [L, L]
        seq_len = bpp_matrix.shape[0]  # MUST match DM and BPP

        # Truncate embedding if longer than BPP (just like train.py)
        if embedding.shape[0] > seq_len:
            embedding = embedding[:seq_len]
        elif embedding.shape[0] < seq_len:
            raise ValueError(f"[{seq_id}] embedding length {embedding.shape[0]} < bpp length {seq_len}")

        # Construct data array to match sequence length
        data_array = np.zeros((seq_len, 3), dtype=np.float32)
        for i in range(seq_len):
            token_idx = token2int.get(seq[i], token2int['X']) if i < len(seq) else token2int['X']
            data_array[i, 0] = token_idx
            data_array[i, 1] = i / seq_len
            data_array[i, 2] = 1.0

        # Generate distance mask and concatenate with BPP matrix
        dm = get_distance_mask(seq_len).astype(np.float32)  # shape: [3, L, L]
        bpps_with_dm = np.concatenate([
            bpp_matrix[np.newaxis, :, :],  # shape: [1, L, L]
            dm  # shape: [3, L, L]
        ], axis=0)  # shape: [4, L, L]

        src_mask = np.ones((1, seq_len), dtype=bool)

        return {
            'data': data_array,
            'bpp': bpps_with_dm,
            'labels': np.array(label, dtype=np.float32),
            'ew': np.array(ew, dtype=np.float32),
            'src_mask': src_mask,
            'embedding': embedding,
            'id': seq_id
        }

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

    short_preds = short_preds[:, :91]
    short_stds = short_stds[:, :91]
    long_preds = long_preds[:, :91]
    long_stds = long_stds[:, :91]
    short_stds[:, 68:] = 0

    ls_indices = test.seq_length == 130
    long_data = test[ls_indices]
    long_ids = np.asarray(long_data.id.to_list())
    long_sequences = np.asarray(long_data.sequence.to_list())

    long_stds = opts.error_alpha + np.exp(-5 * opts.error_beta * long_stds)

    ss_indices = test.seq_length == 107
    short_data = test[ss_indices]
    short_ids = np.asarray(short_data.id)
    short_sequences = np.asarray(short_data.sequence)

    short_stds = opts.error_alpha + np.exp(-5 * opts.error_beta * short_stds)

    short_preds_padded = np.pad(short_preds, ((0, 0), (0, 23), (0, 0)), constant_values=0)
    short_stds_padded = np.pad(short_stds, ((0, 0), (0, 23), (0, 0)), constant_values=0)

    train_seqs = np.concatenate([train_seqs, short_sequences, long_sequences])
    train_labels = np.concatenate([train_labels, short_preds_padded, long_preds], axis=0)
    train_ids = np.concatenate([train_ids, short_ids, long_ids], axis=0)

    short_stds_padded = np.pad(short_stds, ((0, 0), (0, train_ew.shape[1] - short_stds.shape[1]), (0, 0)), constant_values=0)

    if long_stds.shape[1] < train_ew.shape[1]:
        long_stds = np.pad(long_stds, ((0, 0), (0, train_ew.shape[1] - long_stds.shape[1]), (0, 0)), constant_values=0)

    train_ew = np.concatenate([train_ew, short_stds_padded, long_stds], axis=0)

    print("Final train_labels shape after concatenation:", train_labels.shape)
    print("Final train_ids shape after concatenation:", train_ids.shape)
    print("Final train_ew shape after concatenation:", train_ew.shape)

    bpps_save_dir_train = os.path.join(opts.path, 'bpps_train')
    bpps_save_dir_val = os.path.join(opts.path, 'bpps_val')
    bpps_save_dir_pl = os.path.join(opts.path, 'bpps_pl')

    precompute_bpps(train_seqs[:n_train], train_ids[:n_train], bpps_save_dir_train)
    precompute_bpps(val_seqs, val_ids, bpps_save_dir_val)
    precompute_bpps(train_seqs[n_train:], train_ids[n_train:], bpps_save_dir_pl)

    # Checkpointing explicitly moved here BEFORE dataset instantiation
    os.system('mkdir weights')
    checkpoints_folder = f'weights/checkpoints_fold{opts.fold}_pl'
    os.system('mkdir logs')
    csv_file = f'logs/log_pl_fold{opts.fold}.csv'
    columns = ['epoch', 'train_loss', 'val_loss']
    logger = CSVLogger(columns, csv_file)

    # Build model explicitly BEFORE dataset instantiation
    model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                         opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                         stride=opts.stride, dropout=opts.dropout,
                         rinalmo_weights_path=opts.rinalmo_weights_path).to(device)

    # optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    optimizer = Ranger(
        model.parameters(), lr=3e-4, weight_decay=5e-2,
        alpha=0.6, k=5, N_sma_threshhold=4, use_gc=True, gc_conv_only=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=5e-7
    )

    criterion = weighted_MCRMSE

    model = nn.DataParallel(model)
    pretrained_checkpoint = f'{opts.weight_path}/checkpoints_fold{opts.fold}/best_model.ckpt'
    if os.path.exists(pretrained_checkpoint):
        state_dict = torch.load(pretrained_checkpoint, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("module."):
                new_state_dict["module." + k] = v
            else:
                new_state_dict[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded fine-tuned model from {pretrained_checkpoint}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
    else:
        raise FileNotFoundError(f"Expected fine-tuned checkpoint not found: {pretrained_checkpoint}")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))

    # NOW create datasets explicitly after model definition
    pl_dataset = RNADatasetPrecomputed(train_seqs[n_train:], train_labels[n_train:], train_ids[n_train:],
                                       train_ew[n_train:], bpps_save_dir_pl)
    val_dataset_original = RNADatasetPrecomputed(val_seqs, val_labels, val_ids, val_ew, bpps_save_dir_val)
    val_dataset = BPPSafeDatasetWithMask(val_dataset_original, len(model.module.transformer_encoder))

    finetune_dataset = RNADatasetPrecomputed(train_seqs[:n_train], train_labels[:n_train, :68], train_ids[:n_train],
                                             train_ew[:n_train, :68], bpps_save_dir_train)

    pl_dataloader = DataLoader(pl_dataset, batch_size=opts.batch_size,
                               shuffle=True, num_workers=opts.workers,
                               collate_fn=variable_length_collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size * 2,
                                shuffle=False, num_workers=opts.workers,
                                collate_fn=variable_length_collate_fn)

    finetune_dataloader = DataLoader(finetune_dataset, batch_size=opts.batch_size // 2,
                                     shuffle=True, num_workers=opts.workers,
                                     collate_fn=variable_length_collate_fn)

    cos_epoch = int(opts.epochs * 0.75) - 1
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (opts.epochs - cos_epoch) * len(finetune_dataloader))

    for epoch in range(opts.epochs):
        model.train(True)
        t=time.time()
        total_loss=0
        optimizer.zero_grad()
        step=0
        if epoch > cos_epoch:
            dataloader=finetune_dataloader
        else:
            dataloader=pl_dataloader

        for data in dataloader:
            step += 1
            lr = get_lr(optimizer)
            src = data['data'].to(device)
            if src.dim() == 4 and src.shape[1] == 1:
                src = src[:, 0, :, :]

            labels = data['labels'].to(device)

            bpps = data['bpp'].to(device)
            # print(f"BPPS tensor stats: shape={bpps.shape}, min={bpps.min()}, max={bpps.max()}, mean={bpps.mean()}")

            # BPPS dimension fix explicitly here:
            if bpps.dim() == 3:
                bpps = bpps.unsqueeze(1).repeat(1, 4, 1, 1)
            elif bpps.dim() == 4 and bpps.shape[1] == 1:
                bpps = bpps.repeat(1, 4, 1, 1)
            elif bpps.dim() == 5 and bpps.shape[1] == 1:
                bpps = bpps[:, 0, :, :, :].repeat(1, 4, 1, 1)

            src_mask = data['src_mask'].to(device)
            # print(f"src_mask stats: shape={src_mask.shape}, dtype={src_mask.dtype}, unique={torch.unique(src_mask)}")

            src_mask = src_mask.repeat(1, len(model.module.transformer_encoder), 1)  # explicitly correct

            output = model(data['embedding'].to(device), bpps, src_mask)

            ew=data['ew'].to(device)


            loss=criterion(output[:,:labels.shape[1]],labels,ew).mean()
            # print(f"Computed loss: {loss.item()}, shape of output: {output.shape}, labels: {labels.shape}, ew: {ew.shape}")

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step(epoch + step / len(dataloader))

            optimizer.zero_grad()
            total_loss+=loss
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                           .format(epoch+1, opts.epochs, step+1, len(dataloader), total_loss/(step+1) , lr,time.time()-t),end='\r',flush=True) #total_loss/(step+1)
            #break
            if epoch > cos_epoch:
                lr_schedule.step()

        print('')
        train_loss=total_loss/(step+1)

        torch.cuda.empty_cache()
        val_loss = -1  # validation skipped this epoch
        if (epoch + 1) % opts.val_freq == 0 and epoch > cos_epoch:
            val_loss = validate(model, device, val_dataloader, batch_size=opts.batch_size)

        to_log = [epoch + 1, train_loss, val_loss]
        logger.log(to_log)

        if (epoch+1)%opts.save_freq==0:
            save_weights(model,optimizer,epoch,checkpoints_folder)


if __name__ == '__main__':
    train_fold()


