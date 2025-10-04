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
try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from tqdm import tqdm
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--weights_path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=21, help='number of tokens to represent RNA tokens (must match training)')
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
    parser.add_argument('--output_dir', type=str, default='../pseudo_labels', help='directory to save pseudo-labels')
    parser.add_argument('--rinalmo_weights_path', type=str,
                        default='/home/slater/PycharmProjects/RiNALMo/RiNALMo/weights/rinalmo_micro_pretrained.pt',
                        help='Path to pretrained RiNALMo weights')
    opts = parser.parse_args()
    return opts


opts=get_args()
#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directory for bpps_short if it doesn't exist'
bpps_save_dir = './data/bpps_short/'
os.makedirs(bpps_save_dir, exist_ok=True)

# Create output directory if it doesn't exist
os.makedirs(opts.output_dir, exist_ok=True)

#build model and logger
fold_models=[]
folds=np.arange(opts.nfolds)
for fold in folds:
    MODELS=[]
    for i in range(5):
        model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                             opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                             dropout=opts.dropout,
                             rinalmo_weights_path=opts.rinalmo_weights_path
                             ).to(device)
        optimizer=torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
        criterion=nn.CrossEntropyLoss(reduction='none')
        lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)
        # Initialization
        opt_level = 'O1'
        model = nn.DataParallel(model)

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {pytorch_total_params}')

        # Replace the model loading line with:
        checkpoint_path = f"{opts.weights_path}/fold{fold}top{i + 1}.ckpt"
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            print(f"Explicitly loaded checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Explicitly required checkpoint not found: {checkpoint_path}")

        model.eval()
        MODELS.append(model)

    dict=MODELS[0].module.state_dict()
    for key in dict:
        for i in range(1,len(MODELS)):
            dict[key]=dict[key]+MODELS[i].module.state_dict()[key]

        dict[key]=dict[key]/float(len(MODELS))

    MODELS[0].module.load_state_dict(dict)
    avg_model=MODELS[0]
    fold_models.append(avg_model)

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    token2int = {x:i for i, x in enumerate('ACGU().BEHIMSX')}
    return np.transpose(
        np.array(
            df[cols]
            .map(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )

json_path=os.path.join(opts.path,'test.json')

# Add these diagnostic lines to pseudo_predict.py before line 109
print(f"Checking for file at: {json_path}")
print(f"File exists: {os.path.exists(json_path)}")
if os.path.exists(json_path):
    print(f"File size: {os.path.getsize(json_path)} bytes")
    with open(json_path, 'r') as f:
        first_line = f.readline().strip()
        print(f"First line preview: {first_line[:100]}")

test = pd.read_json(json_path, lines=True)

# Process long sequences (130bp)
ls_indices=test.seq_length==130
long_data=test[ls_indices]
data=preprocess_inputs(test[ls_indices])
data=data.reshape(1,*data.shape)

ids=np.asarray(long_data.id.to_list())
long_dataset = RNADataset(long_data.sequence.to_list(), np.zeros(len(ls_indices)), ids,
                          np.arange(len(ls_indices)), opts.path, training=True, k=opts.kmers[0])

long_dataloader=DataLoader(long_dataset, batch_size=opts.batch_size, shuffle=False)

# Process short sequences (107bp)
ss_indices=test.seq_length==107 # identify the short sequences - length 107 - for special processing
short_data=test[ss_indices] # use the ss_indices boolean mask to create short_data - containing only short sequences
ids=np.asarray(short_data.id.to_list()) # extract the ids into a numpy array
data=preprocess_inputs(test[ss_indices]) # convert the raw RNA sequences into numerical values
data=data.reshape(1,*data.shape)

# Authoritative BPPS precomputation step:
bpps_save_dir = './data/bpps_short/'
os.makedirs(bpps_save_dir, exist_ok=True)

short_df = pd.read_json(opts.path + 'test.json', lines=True)
short_df = short_df[short_df['sequence'].str.len() == 107]

for idx, row in tqdm(short_df.iterrows(), total=len(short_df), desc='Precomputing short BPPS matrices'):
    seq_id = row['id']
    seq = row['sequence']
    bpps_path = os.path.join(bpps_save_dir, f'{seq_id}.npy')

    if not os.path.exists(bpps_path):
        bpps_matrix = compute_bpps_matrix(seq)  # authoritative BPPS computation
        np.save(bpps_path, bpps_matrix)  # authoritative save

# Explicit authoritative short sequence dataset:
class ShortSequenceDataset(Dataset):
    def __init__(self, dataframe, bpps_dir):
        self.dataframe = dataframe
        self.bpps_dir = bpps_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        seq_id = row['id']
        seq = row['sequence']

        data_tensor = torch.tensor([token2int[x] for x in seq], dtype=torch.long)
        bpps_path = os.path.join(self.bpps_dir, f'{seq_id}.npy')

        # Explicit authoritative load from disk:
        bpp_matrix = np.load(bpps_path)
        bpp_tensor = torch.from_numpy(bpp_matrix).float()

        src_mask = torch.ones(len(seq), dtype=torch.bool)

        return {
            'data': data_tensor,
            'bpp': bpp_tensor,
            'src_mask': src_mask
        }

# Authoritative dataset and loader replacement:
short_dataset = ShortSequenceDataset(short_df, bpps_save_dir)
short_dataloader = DataLoader(short_dataset, batch_size=opts.batch_size,
                              shuffle=False, collate_fn=variable_length_collate_fn, num_workers=4)


# Store sequence IDs and predictions
ids = []
preds = []

with torch.no_grad():
    # Process long sequences (130bp sequences, predict first 91 positions)
    for batch in tqdm(long_dataloader, desc="Processing long sequences"):
        sequence = batch['data'].to(device)
        bpps = batch['bpp'].float().to(device)
        src_mask = batch['src_mask'].to(device)

        sequence = sequence.squeeze(1)  # explicitly remove the extra dimension
        bpps = bpps.squeeze(1)  # explicitly remove the extra dimension
        src_mask = src_mask.squeeze(1)  # explicitly remove the extra dimension

        # correction of BPPS dimensions:
        if bpps.dim() == 3:
            bpps = bpps.unsqueeze(1).repeat(1, 4, 1, 1)
        elif bpps.dim() == 4 and bpps.shape[1] == 1:
            bpps = bpps.repeat(1, 4, 1, 1)
        elif bpps.dim() == 5 and bpps.shape[1] == 1:
            bpps = bpps[:, 0, :, :, :].repeat(1, 4, 1, 1)

        outputs = []
        for model_idx, model in enumerate(fold_models):
            try:
                temp = model(sequence, sequence, bpps, src_mask)[:, :91, :].cpu()
                outputs.append(temp)
            except Exception as e:
                print(f"Error processing long sequence batch for model {model_idx}: {e}")
                print(f"Sequence shape: {sequence.shape}")
                print(f"BPP shape: {bpps.shape}")
                empty_pred = torch.zeros(sequence.shape[0], 91, opts.nclass)
                outputs.append(empty_pred)

        # Correct tensor stacking: stack along new dimension (fold dimension)
        outputs = torch.stack(outputs, dim=0).numpy()

        # Append predictions and IDs
        for i in range(outputs.shape[1]):
            preds.append(outputs[:, i, :, :])
            ids.append(batch['id'][i])

    # Process short sequences (107bp sequences, predict first 68 positions)
    for batch in tqdm(short_dataloader, desc="Processing short sequences"):
        sequence = batch['data'].to(device)
        bpps = batch['bpp'].float().to(device)
        src_mask = batch['src_mask'].to(device)

        # Explicit authoritative correction of BPPS dimensions:
        if bpps.dim() == 3:
            bpps = bpps.unsqueeze(1).repeat(1, 4, 1, 1)
        elif bpps.dim() == 4 and bpps.shape[1] == 1:
            bpps = bpps.repeat(1, 4, 1, 1)
        elif bpps.dim() == 5 and bpps.shape[1] == 1:
            bpps = bpps[:, 0, :, :, :].repeat(1, 4, 1, 1)

        outputs = []
        for model_idx, model in enumerate(fold_models):
            try:
                temp = model(sequence, sequence, bpps, src_mask)[:, :68, :].cpu()
                outputs.append(temp)
            except Exception as e:
                print(f"Error processing short sequence batch for model {model_idx}: {e}")
                print(f"Sequence shape: {sequence.shape}")
                print(f"BPP shape: {bpps.shape}")
                empty_pred = torch.zeros(sequence.shape[0], 68, opts.nclass)
                outputs.append(empty_pred)

        # Correct tensor stacking: stack along new dimension (fold dimension)
        outputs = torch.stack(outputs, dim=0).numpy()

        # Append predictions and IDs
        for i in range(outputs.shape[1]):
            preds.append(outputs[:, i, :, :])
            ids.append(batch['id'][i])

# Organize predictions by sequence ID
preds_to_csv = [[] for _ in range(len(test))]
test_ids = test.id.to_list()

for i in tqdm(range(len(preds)), desc="Organizing predictions"):
    index = test_ids.index(ids[i])
    preds_to_csv[index].append(preds[i])

# Authoritative explicit initialization before separation
long_preds, long_stds, short_preds, short_stds = [], [], [], []
long_ids_set = set(long_data.id.to_list())
short_ids_set = set(short_data.id.to_list())  # Explicit definition

# Separate explicitly into long (130bp) and short (107bp) predictions
for pred_set, seq_id in zip(preds_to_csv, test_ids):
    fold_preds = np.stack(pred_set, axis=0)  # (num_folds, seq_len, nclass)
    mean_pred = fold_preds.mean(axis=0)      # Averaging predictions across folds
    std_pred = fold_preds.std(axis=0)        # Standard deviation across folds

    if seq_id in long_ids_set:
        long_preds.append(mean_pred)
        long_stds.append(std_pred)
    elif seq_id in short_data.id.to_list():
        short_preds.append(mean_pred)
        short_stds.append(std_pred)

# Explicitly convert lists to arrays and explicitly average across fold dimension
long_preds = np.array(long_preds)
long_stds = np.array(long_stds)

short_preds = np.array(short_preds)
short_stds = np.array(short_stds)

# Explicit check and dimension correction (Average over fold dimension)
if long_preds.ndim == 4:
    long_preds = long_preds.mean(axis=1)  # Correct axis to average over folds, explicitly keep batch
if long_stds.ndim == 4:
    long_stds = long_stds.mean(axis=1)

if short_preds.ndim == 4:
    short_preds = short_preds.mean(axis=1)  # Correct axis explicitly specified
if short_stds.ndim == 4:
    short_stds = short_stds.mean(axis=1)

# Now explicitly limit to first 91 positions
long_preds = long_preds[:, :91, :]
long_stds = long_stds[:, :91, :]

short_preds = short_preds[:, :91, :]
short_stds = short_stds[:, :91, :]

# Set authoritative zeros explicitly for stds beyond position 68 in short sequences
short_stds[:, 68:, :] = 0


print(f"Long sequences predictions shape: {long_preds.shape}")
print(f"Short sequences predictions shape: {short_preds.shape}")

# Explicitly save corrected pseudo-labels for each fold
for fold in range(opts.nfolds):
    output_file = os.path.join(opts.output_dir, f'pseudo_labels_fold{fold}.p')
    with open(output_file, 'wb') as f:
        pickle.dump((long_preds, long_stds, short_preds, short_stds), f)
    print(f"Saved pseudo-labels for fold {fold} to {output_file}")

print(f"Pseudo-label generation complete. Files saved to {opts.output_dir}")
