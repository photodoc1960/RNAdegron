import os
import torch
import torch.nn as nn
import time
from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse
from ranger import Ranger
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm


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
    # Explicitly load precomputed embeddings
    precomputed_embeddings_path = 'precomputed_embeddings_train.pt'
    training_dataset = RNADataset(
    seqs=train_data.sequence.to_list(),  # explicitly required by your definition
    labels=np.zeros(len(train_data)),
    precomputed_embeddings_path=precomputed_embeddings_path,
    ids=train_data.id.to_list(),
    ew=torch.load('data/ew.pt'),  # explicitly load or provide correct ew tensor
    bpp_path='data',              # explicitly ensure correct path
    training=True
    )

    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size,
                                     shuffle=True, num_workers=opts.workers)  # Explicitly remove complex collate_fn

    val_ids = np.asarray(val_data.id.to_list())
    # Explicitly load precomputed embeddings for validation
    precomputed_embeddings_val_path = 'precomputed_embeddings_val.pt'
    val_dataset = RNADataset(
    seqs=val_data.sequence.to_list(),
    labels=np.zeros(len(val_data)),
    precomputed_embeddings_path=precomputed_embeddings_val_path,
    ids=val_data.id.to_list(),
    ew=torch.load('data/ew_val.pt'),  # explicitly load correct ew tensor for validation
    bpp_path='data',
    training=False
    )

    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size,
                                shuffle=False, num_workers=opts.workers)

    # checkpointing
    checkpoints_folder = 'pretrain_weights'

    os.system('mkdir logs')

    csv_file = 'logs/pretrain.csv'.format((opts.fold))
    columns = ['epoch', 'train_loss', 'val_loss']
    logger = CSVLogger(columns, csv_file)

    # build model and logger
    model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                         opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers, stride=opts.stride,
                         dropout=opts.dropout, pretrain=True,
                         rinalmo_weights_path='/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_micro_pretrained.pt').to(device)
    model = model.to(device)

    optimizer = Ranger(model.parameters(), weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss()
    # Mixed precision initialization
    opt_level = 'O1'
    model = nn.DataParallel(model)

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
            # Explicitly load precomputed embeddings
            embeddings = data['embedding'].to(device)  # already float tensor (RiNALMo embeddings)
            labels = data['labels'].to(device)
            bpps = data['bpp'].to(device)
            src_mask = data['src_mask'].to(device)

            # Masking/mutation no longer needed explicitly (embeddings precomputed)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs, original_tokens, mask_positions = model(embeddings, bpps, src_mask)
                # Explicit authoritative verification assertions (adjusted for precomputed embeddings)
                assert original_tokens.dtype == torch.long, "original_tokens must explicitly be integers."
                assert mask_positions.dtype == torch.bool, "mask_positions must explicitly be boolean."
                assert outputs[0].dtype == torch.float, "Model outputs must explicitly be floats."
                assert original_tokens.shape == mask_positions.shape, "original_tokens and mask_positions must explicitly match in shape."

            # Explicit authoritative semantics for tensors involved:
            # outputs[0]: Continuous float logits from the transformer decoder predicting nucleotides
            # masked_nucleotide_labels: Discrete integer ground-truth nucleotide labels at explicitly masked positions
            # mask_positions: Boolean mask explicitly identifying positions of masked tokens

            # Corrected authoritative handling of empty masked labels
            masked_nucleotide_labels = original_tokens[mask_positions]

            # Explicitly authoritative filtering: retain only valid nucleotide labels [5-8]
            valid_label_positions = masked_nucleotide_labels != 4

            # Check explicitly if tensor is empty after filtering
            if valid_label_positions.sum() == 0:
                #print(
                #   "Explicit warning: No valid nucleotide labels [5-8] found at masked positions. Explicitly skipping backward pass.")
                pass
                continue  # Explicitly skip the rest of the loop for this iteration only

            masked_nucleotide_labels = masked_nucleotide_labels[valid_label_positions]

            # Explicit authoritative label remapping from [5,6,7,8] to [0,1,2,3]
            masked_nucleotide_labels -= 5

            outputs_masked = outputs[0][mask_positions][valid_label_positions]

            # Explicitly verified tensor integrity after remapping
            assert masked_nucleotide_labels.min() >= 0 and masked_nucleotide_labels.max() <= 3, \
                f"Labels explicitly remapped incorrectly, got range [{masked_nucleotide_labels.min()}, {masked_nucleotide_labels.max()}]"

            # Explicit authoritative remapping
            label_remap = {5: 0, 6: 1, 7: 2, 8: 3}
            for orig_val, mapped_val in label_remap.items():
                masked_nucleotide_labels[masked_nucleotide_labels == orig_val] = mapped_val

            # Explicit verification
            assert masked_nucleotide_labels.min() >= 0 and masked_nucleotide_labels.max() <= 3, \
                f"Labels explicitly remapped incorrectly, got range [{masked_nucleotide_labels.min()}, {masked_nucleotide_labels.max()}]"

            loss = criterion(
                outputs_masked.reshape(-1, 4),
                masked_nucleotide_labels.reshape(-1)
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
                    # for step in range(1):
                    src = data['data']
                    labels = data['labels']
                    bpps = data['bpp'].to(device)
                    src_mask = data['src_mask'].to(device)

                    masked, mask_positions = mask_rna_input(src)

                    src = src.to(device).long()
                    masked = masked.to(device).long()
                    # Verify src labels before calculating loss

                    with torch.no_grad():
                        outputs, original_tokens, mask_positions = model(masked, src, bpps, src_mask)
                        # Explicit authoritative verification assertions
                        assert original_tokens.dtype == torch.long, "original_tokens must explicitly be integers."
                        assert mask_positions.dtype == torch.bool, "mask_positions must explicitly be boolean."
                        assert outputs[0].dtype == torch.float, "Model outputs must explicitly be floats."
                        assert original_tokens.shape == mask_positions.shape, "original_tokens and mask_positions must explicitly match in shape."

                    # Explicit authoritative semantics for tensors involved:
                    # outputs[0]: Continuous float logits from the transformer decoder predicting nucleotides
                    # masked_nucleotide_labels: Discrete integer ground-truth nucleotide labels at explicitly masked positions
                    # mask_positions: Boolean mask explicitly identifying positions of masked tokens

                    masked_nucleotide_labels = original_tokens[mask_positions]

                    # Explicitly filter out mask tokens (value 4) before remapping
                    valid_label_positions = masked_nucleotide_labels != 4

                    # Explicit authoritative check for empty tensor after filtering (VALIDATION)
                    if valid_label_positions.sum() == 0:
                        continue  # explicitly skip iteration if no valid positions remain

                    masked_nucleotide_labels = masked_nucleotide_labels[valid_label_positions]
                    outputs_masked = outputs[0][mask_positions][valid_label_positions]

                    # Explicitly authoritative label remapping
                    label_remap = {5: 0, 6: 1, 7: 2, 8: 3}
                    for orig_val, mapped_val in label_remap.items():
                        masked_nucleotide_labels[masked_nucleotide_labels == orig_val] = mapped_val

                    # Explicit verification of tensor integrity (VALIDATION)
                    assert masked_nucleotide_labels.min() >= 0 and masked_nucleotide_labels.max() <= 3, \
                        f"Labels explicitly remapped incorrectly (VALIDATION), got range [{masked_nucleotide_labels.min()}, {masked_nucleotide_labels.max()}]"

                    loss = criterion(
                        outputs_masked.reshape(-1, 4),
                        masked_nucleotide_labels.reshape(-1)
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
