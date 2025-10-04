import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
from Functions_v5 import *
from Dataset_v5 import *
from X_Network_v5 import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse
from ranger import Ranger

from torchvision import transforms, utils
#from Mutation import *
from sklearn.model_selection import train_test_split, KFold

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
    opts = parser.parse_args()
    return opts

def load_pretrained_into_dataparallel(model, checkpoint_path):
    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    if list(state_dict.keys())[0].startswith('module.'):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
    else:
        new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    print("✅ Weights loaded successfully.")
    return model


def train_fold():
    #get arguments
    opts=get_args()

    #gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #instantiate datasets
    json_path=os.path.join(opts.path,'train.json')

    json=pd.read_json(json_path,lines=True)
    json=json[json.signal_to_noise > opts.noise_filter]
    ids=np.asarray(json.id.to_list())


    error_weights=get_errors(json)
    error_weights=opts.error_alpha+np.exp(-error_weights*opts.error_beta)
    train_indices,val_indices=get_train_val_indices(json,opts.fold,SEED=2020,nfolds=opts.nfolds)

    _,labels=get_data(json)
    sequences=np.asarray(json.sequence)
    train_seqs=sequences[train_indices]
    val_seqs=sequences[val_indices]
    train_labels=labels[train_indices]
    val_labels=labels[val_indices]
    train_ids=ids[train_indices]
    val_ids=ids[val_indices]
    train_ew=error_weights[train_indices]
    val_ew=error_weights[val_indices]

    dataset = RNADataset(train_seqs, train_labels, train_ids, train_ew, opts.path, pad=True, training=True)
    val_dataset = RNADataset(val_seqs, val_labels, val_ids, val_ew, opts.path, pad=True, training=True)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size,
                            shuffle=True, num_workers=opts.workers,
                            collate_fn=variable_length_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size * 2,
                                shuffle=False, num_workers=opts.workers,
                                collate_fn=variable_length_collate_fn)

    checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
    csv_file='log_fold{}.csv'.format((opts.fold))
    columns=['epoch','train_loss',
             'val_loss']
    logger=CSVLogger(columns,csv_file)

    #build model and logger
    rinalmo_weights_path = '/home/slater/RiNALMo/weights/rinalmo_micro_pretrained.pt'

    model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                        opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers, stride=opts.stride,
                        dropout=opts.dropout, rinalmo_weights_path=rinalmo_weights_path).to(device)

    optimizer=Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion=weighted_MCRMSE
    scaler = GradScaler()

    # Mixed precision initialization
    opt_level = 'O1'
    model = nn.DataParallel(model)
    best_model_path = 'pretrain_weights/best_model.ckpt'
    if os.path.exists(best_model_path):
        model = load_pretrained_into_dataparallel(model, best_model_path)
    else:
        model = load_pretrained_into_dataparallel(model, 'pretrain_weights/epoch0.ckpt')


    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))

    #training loop
    cos_epoch=int(opts.epochs*0.75)-1
    total_training_steps = opts.epochs * len(dataloader)
    warmup_steps = int(0.06 * total_training_steps)
    lr_schedule = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_training_steps,
        initial_multiplier=0.01
    )

    for epoch in range(opts.epochs):
        model.train(True)
        t=time.time()
        total_loss=0
        optimizer.zero_grad()
        step=0
        for data in dataloader:
            step+=1
            lr=get_lr(optimizer)

            src = data['embedding'].to(device)
            if src.dim() == 4 and src.shape[1] == 1:
                src = src[:, 0, :, :]

            labels = data['labels']

            bpps = data['bpp'].to(device)
            if bpps.dim() == 5 and bpps.shape[1] == 1:
                bpps = bpps[:, 0, :, :, :]

            src_mask = data['src_mask'].to(device)
            if src_mask.dim() == 3 and src_mask.shape[1] == 1:
                src_mask = src_mask[:, 0, :]

            labels=labels.to(device)#.float()
            if src_mask.dim() == 2:
                src_mask = src_mask.unsqueeze(1).repeat(1, opts.nlayers, 1)
            elif src_mask.dim() == 3 and src_mask.size(1) != opts.nlayers:
                src_mask = src_mask[:, 0:1, :].repeat(1, opts.nlayers, 1)

            with autocast(dtype=torch.float16):
                output = model(src, bpps, src_mask)
                ew = data['ew'].to(device)
                loss = criterion(output[:, :68], labels, ew).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            lr_schedule.step()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.empty_cache()
            total_loss+=loss
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                           .format(epoch+1, opts.epochs, step+1, len(dataloader), total_loss/(step+1) , lr,time.time()-t),end='\r',flush=True) #total_loss/(step+1)

            if epoch > cos_epoch:
                lr_schedule.step()
        print('')
        train_loss=total_loss/(step+1)
        torch.cuda.empty_cache()
        if (epoch+1)%opts.val_freq==0 and epoch > cos_epoch:
            val_loss=validate(model,device,val_dataloader,batch_size=opts.batch_size)
            to_log=[epoch+1,train_loss,val_loss,]
            logger.log(to_log)

        if (epoch+1)%opts.save_freq==0:
            save_weights(model,optimizer,epoch,checkpoints_folder)

    get_best_weights_from_fold(opts.fold)

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

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == '__main__':
    train_fold()
