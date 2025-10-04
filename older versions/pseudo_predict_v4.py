import os
import torch
import torch.nn as nn
import time
from Functions_v4 import *
from Dataset_v4 import *
from X_Network_v4 import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse
from tqdm import tqdm
import pickle

# Removed apex imports and dependencies

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1')
    parser.add_argument('--path', type=str, default='../')
    parser.add_argument('--weights_path', type=str, default='../')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ntoken', type=int, default=4)
    parser.add_argument('--nclass', type=int, default=5)
    parser.add_argument('--ninp', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nhid', type=int, default=2048)
    parser.add_argument('--nlayers', type=int, default=6)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--warmup_steps', type=int, default=3200)
    parser.add_argument('--lr_scale', type=float, default=0.1)
    parser.add_argument('--nmute', type=int, default=18)
    parser.add_argument('--kmers', type=int, nargs='+', default=[2,3,4,5,6])
    parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='../pseudo_labels')
    return parser.parse_args()

opts = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(opts.output_dir, exist_ok=True)

fold_models = []
folds = np.arange(opts.nfolds)
rinalmo_weights_path = '/home/slater/RiNALMo/weights/rinalmo_micro_pretrained.pt'
for fold in folds:
    MODELS = []
    for i in range(5):
        model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                              opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                              dropout=opts.dropout,
                              rinalmo_weights_path=rinalmo_weights_path).to(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
        criterion = nn.CrossEntropyLoss(reduction='none')
        lr_schedule = lr_AIAYN(optimizer, opts.ninp, opts.warmup_steps, opts.lr_scale)
        model = nn.DataParallel(model)

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {pytorch_total_params}')

        checkpoint_path = f"{opts.weights_path}/fold{fold}top{i + 1}.ckpt"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        model.eval()
        MODELS.append(model)

    avg_state = MODELS[0].module.state_dict()
    for key in avg_state:
        if torch.is_floating_point(avg_state[key]):
            avg_state[key] = avg_state[key].float()
            for i in range(1, len(MODELS)):
                avg_state[key] += MODELS[i].module.state_dict()[key].float()
            avg_state[key] /= float(len(MODELS))
        else:
            avg_state[key] = avg_state[key].clone()

    MODELS[0].module.load_state_dict(avg_state)

    fold_models.append(MODELS[0])

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    token2int = {x:i for i, x in enumerate('ACGU().BEHIMSX')}
    return np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
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

# Load precomputed RiNALMo embeddings
embedding_data = torch.load("data/precomputed_embeddings.pt")
embedding_dict = dict(zip(embedding_data["ids"], embedding_data["embeddings"]))

# Dataset wrapper to add embeddings
class EmbeddedRNADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, embedding_dict):
        self.dataset = dataset
        self.embedding_dict = embedding_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['embedding'] = self.embedding_dict[str(sample['id'])]
        return sample

long_dataset = RNADataset(
    long_data.sequence.to_list(),
    np.zeros(len(ls_indices)),
    ids,
    np.arange(len(ls_indices)),
    opts.path,
    training=False,
    k=opts.kmers[0]
)
long_dataset = EmbeddedRNADataset(long_dataset, embedding_dict)

long_dataloader=DataLoader(long_dataset, batch_size=opts.batch_size, shuffle=False)

# Process short sequences (107bp)
ss_indices=test.seq_length==107 # identify the short sequences - length 107 - for special processing
short_data=test[ss_indices] # use the ss_indices boolean mask to create short_data - containing only short sequences
ids=np.asarray(short_data.id.to_list()) # extract the ids into a numpy array
data=preprocess_inputs(test[ss_indices]) # convert the raw RNA sequences into numerical values
data=data.reshape(1,*data.shape)

class BPPSafeDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        if sample['bpp'].ndim == 2:
            sample['bpp'] = np.expand_dims(sample['bpp'], axis=0)
        sample['id'] = self.original_dataset[idx]['id']  # ✅ re-insert missing key
        return sample

original_short_dataset = RNADataset(short_data.sequence.to_list(), np.zeros(len(short_data)), ids,
                        np.arange(len(short_data)), opts.path, training=False, k=opts.kmers[0])
short_dataset = BPPSafeDataset(original_short_dataset)
short_dataset = EmbeddedRNADataset(short_dataset, embedding_dict)

short_dataloader = DataLoader(short_dataset, batch_size=opts.batch_size, shuffle=False, collate_fn=variable_length_collate_fn)

# Store sequence IDs and predictions
ids = []
preds = []

with torch.no_grad():
    # Process long sequences (130bp sequences, predict first 91 positions)
    for batch in tqdm(long_dataloader, desc="Processing long sequences"):
        sequence = batch['embedding'].to(device)
        bpps = batch['bpp'].float().to(device)
        src_mask = batch['src_mask'].to(device)

        sequence = sequence.squeeze(1)  # explicitly remove the extra dimension
        bpps = bpps.squeeze(1)  # explicitly remove the extra dimension
        src_mask = src_mask.squeeze(1)  # explicitly remove the extra dimension

        assert sequence.shape[-1] == 480, f"Expected RiNALMo embeddings with 480 dims, got: {sequence.shape}"

        outputs = []
        for model_idx, model in enumerate(fold_models):
            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    temp = model(sequence, bpps, src_mask)[:, :91, :].cpu()
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
        sequence = batch['embedding'].to(device)
        bpps = batch['bpp'].float().to(device)
        src_mask = batch['src_mask'].to(device)

        assert sequence.shape[-1] == 480, f"Expected RiNALMo embeddings with 480 dims, got: {sequence.shape}"

        outputs = []
        for model_idx, model in enumerate(fold_models):
            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    temp = model(sequence, bpps, src_mask)[:, :68, :].cpu()
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