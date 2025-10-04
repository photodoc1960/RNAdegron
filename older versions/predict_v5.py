import os
import torch
import torch.nn as nn
import time
from Functions_v5 import *
from Dataset_v5 import Dataset, variable_length_collate_fn
from torch.utils.data import DataLoader
from X_Network_v5 import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
import argparse
import glob

tokens = ['<cls>', '<pad>', '<eos>', '<unk>', '<mask>', 'A', 'C', 'G', 'T', 'I', 'R', 'Y', 'K', 'M', 'S', 'W', 'B', 'D', 'H', 'V', 'N', '-', 'E', 'X']

class RNADataset(Dataset):
    def __init__(self, seqs, labels, ids, ew, bpp_path, transform=None, training=True, pad=False, k=5, num_layers=6):
        self.transform = transform
        self.seqs = seqs
        self.data = []
        self.labels = labels.astype('float32')
        self.bpp_path = bpp_path
        self.ids = ids
        self.training = training
        self.bpps = []
        self.num_layers = num_layers
        self.lengths = []

        max_seq_len = max(len(seq) for seq in seqs)
        dm = get_distance_mask(max_seq_len)
        self.dms = np.asarray([dm for i in range(12)])

        for i, id in enumerate(self.ids):
            seq = self.seqs[i]
            seq_len = len(seq)
            self.lengths.append(seq_len)

            bpp_file = os.path.join(self.bpp_path, f"{id}_bpp.npy")
            struc_file = os.path.join(self.bpp_path, f"{id}_struc.p")
            loop_file = os.path.join(self.bpp_path, f"{id}_loop.p")

            try:
                bpps = np.load(bpp_file, allow_pickle=True)
                with open(struc_file, 'rb') as f:
                    structures = pickle.load(f)
                with open(loop_file, 'rb') as f:
                    loops = pickle.load(f)

                if len(structures[0]) != len(seq):
                    structures = [structures[0][:len(seq)]]
                if len(loops[0]) != len(seq):
                    loops = [loops[0][:len(seq)]]

            except Exception:
                bpp_matrix = generate_bpp_matrix(seq)
                structure, loop_annotation = generate_structure_and_loop(seq)
                structures = [structure]
                loops = [loop_annotation]
                bpps = np.expand_dims(bpp_matrix, axis=0)
                np.save(bpp_file, bpps)
                with open(struc_file, 'wb') as f:
                    pickle.dump(structures, f)
                with open(loop_file, 'wb') as f:
                    pickle.dump(loops, f)

            input_data = []
            for j in range(min(bpps.shape[0], len(structures))):
                input_seq = np.asarray([tokens.index(s) if s in tokens else tokens.index('<unk>') for s in seq.replace('U', 'T')])
                input_structure = np.asarray([tokens.index(s) if s in tokens else tokens.index('<unk>') for s in structures[j]])
                loop_token_map = {'B': 0, 'E': 1, 'H': 2, 'I': 3, 'M': 4, 'S': 5, 'X': 6}
                input_loop = np.asarray([loop_token_map.get(s, 6) for s in loops[j]])
                input_data.append(np.stack([input_seq, input_structure, input_loop], -1))

            input_data = np.asarray(input_data)
            self.data.append(input_data)
            self.bpps.append(np.clip(bpps, 0, 1).astype('float32'))

        self.lengths = np.asarray(self.lengths)
        self.ew = ew
        self.k = k
        self.src_masks = [np.ones((self.num_layers, l), dtype='int8') for l in self.lengths]
        self.pad = pad

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx].squeeze(),
            'bpp': self.bpps[idx],
            'src_mask': self.src_masks[idx],
            'labels': self.labels[idx].squeeze(),
            'id': self.ids[idx],
            'ew': self.ew[idx].squeeze()
        }
        return sample

class BPPSafeDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        if sample['bpp'].ndim == 2:
            sample['bpp'] = np.expand_dims(sample['bpp'], axis=0)
        return sample


# Dataset wrapper to add embeddings
class EmbeddedRNADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, embedding_dict):
        self.dataset = dataset
        self.embedding_dict = embedding_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample_id = str(sample['id']).strip()
        try:
            sample['embedding'] = self.embedding_dict[sample_id]
        except KeyError:
            raise KeyError(
                f"[EmbeddedRNADataset] ID '{sample_id}' not found in embedding_dict.\n"
                f"Sample keys (first 5): {list(self.embedding_dict.keys())[:5]}\n"
                f"Type: {type(sample_id)}, Raw: {repr(sample_id)}"
            )
        return sample


def process_with_sliding_window(fold_models, sequence, bpps, window_size=130, stride=65):
    """Process long RNA sequences using sliding window approach."""
    device = sequence.device
    seq_len = sequence.shape[1] if sequence.ndim == 3 else sequence.shape[0]

    # Ensure bpps has 4 dimensions [batch, channels, seq_len, seq_len]
    if bpps.ndim == 3:  # (channels, seq_len, seq_len)
        bpps = bpps.unsqueeze(0)  # becomes (1, channels, seq_len, seq_len)
    elif bpps.ndim != 4:
        raise ValueError(f"Unexpected BPP shape: {bpps.shape}")

    # Ensure we have exactly 4 channels for the model's mask_dense layer
    if bpps.shape[1] != 4:
        raise ValueError(f"Expected BPP tensor to have exactly 4 channels, but got {bpps.shape[1]} channels")

    # For sequences within window size, process directly
    if seq_len <= window_size:
        window_preds = []
        for model in fold_models:
            with torch.no_grad():
                pred = model(sequence, bpps)
            window_preds.append(pred)

        return [(0, seq_len, torch.stack(window_preds))]

    # Initialize storage for results
    all_preds = []

    # Process each window
    for start_idx in range(0, seq_len - window_size + 1, stride):
        end_idx = start_idx + window_size

        # Extract window from sequence
        if sequence.ndim == 3:
            seq_window = sequence[:, start_idx:end_idx, :]
        else:
            seq_window = sequence.unsqueeze(0)[:, start_idx:end_idx, :]

        # Correct slicing for BPP tensor
        bpp_window = bpps[:, :, start_idx:end_idx, start_idx:end_idx]

        # Verify that bpp_window has shape [batch, 4, window_size, window_size]
        if bpp_window.shape[1] != 4:
            raise ValueError(f"Incorrect bpp_window shape: {bpp_window.shape}")

        # Process with all models
        window_preds = []
        for model in fold_models:
            with torch.no_grad():
                pred = model(seq_window, bpp_window)
            window_preds.append(pred)

        # Store window predictions as tuple: (start, end, predictions)
        all_preds.append((start_idx, end_idx, torch.stack(window_preds)))

    return all_preds

def merge_window_predictions(window_outputs, sequence_length):
    """Merge predictions from multiple windows covering a sequence."""
    # Determine output dimensions from the first window
    _, _, first_output = window_outputs[0]
    output_dim = first_output.shape[-1]

    # Initialize merged output tensor
    merged = torch.zeros((first_output.shape[0], first_output.shape[1],
                          first_output.shape[2], sequence_length, output_dim))
    counts = torch.zeros((sequence_length,))

    # Merge all windows
    for start_idx, end_idx, output in window_outputs:
        merged[:, :, :, start_idx:end_idx, :] += output
        counts[start_idx:end_idx] += 1

    # Average the overlapping regions
    for i in range(sequence_length):
        if counts[i] > 0:
            merged[:, :, :, i, :] /= counts[i]

    return merged

try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--weights_path', type=str, default='../', help='path of csv file with DNA sequences and labels')
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
    parser.add_argument('--step_size', type=int, default=65, help='Step size for sliding window')

    opts = parser.parse_args()
    return opts


opts=get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#lr=0

sub_folder=f"{opts.weights_path}_subs"
os.system(f'mkdir {sub_folder}')


#checkpointing
checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
csv_file='log_fold{}.csv'.format((opts.fold))
columns=['epoch','train_loss','train_acc','recon_acc',
         'val_loss','val_auc','val_acc','val_sens','val_spec']

rinalmo_weights_path = '/home/slater/RiNALMo/weights/rinalmo_micro_pretrained.pt'
fold_models=[]
folds=np.arange(opts.nfolds)
for fold in folds:
    model = RNADegformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                     opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                     dropout=opts.dropout,
                     rinalmo_weights_path=rinalmo_weights_path).to(device)
    model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {pytorch_total_params}')

    # Load the single best checkpoint for each fold
    checkpoint_path = f"{opts.weights_path}/fold{fold}top1.ckpt"
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint: {checkpoint_path}")
        model.eval()
        fold_models.append(model)
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")


# Modify the preprocess_inputs function:
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    token2int = {x: i for i, x in enumerate('ACGU().BEHIMSX')}

    # Check if the expected columns exist
    available_cols = [col for col in cols if col in df.columns]

    # If 'predicted_loop_type' is missing but 'bpRNA_string' is available, use it
    if 'predicted_loop_type' not in available_cols and 'bpRNA_string' in df.columns:
        available_cols.append('bpRNA_string')
        print(f"Using 'bpRNA_string' instead of 'predicted_loop_type'")

    # If we don't have enough columns, exit with error
    if len(available_cols) < 3:
        missing = set(cols) - set(available_cols)
        raise ValueError(f"Missing required columns: {missing}. Available: {df.columns}")

    # Only use the first 3 columns to maintain the expected shape
    use_cols = available_cols[:3]
    print(f"Preprocessing using columns: {use_cols}")

    # Check for uniform sequence lengths
    seq_lens = df[use_cols[0]].apply(len)
    if not seq_lens.nunique() == 1:
        print(f"Warning: Non-uniform sequence lengths detected ({seq_lens.nunique()} different lengths)")
        print(f"Sequence length statistics: min={seq_lens.min()}, max={seq_lens.max()}, mean={seq_lens.mean():.1f}")

        # Pad all sequences to the maximum length
        max_len = seq_lens.max()

        # Process each column and sequence individually to handle variable lengths
        processed_data = []
        for _, row in df.iterrows():
            seq_data = []
            for col in use_cols:
                # Convert sequence to token indices
                seq = [token2int.get(x, token2int['X']) for x in row[col]]
                seq_data.append(seq)
            processed_data.append(np.array(seq_data).T)  # Transpose each sequence individually

        # Return the processed data without further transposition
        return np.array(processed_data, dtype=object)
    else:
        # Standard processing for uniform length sequences
        return np.transpose(
            np.array(
                df[use_cols]
                .applymap(lambda seq: [token2int.get(x, token2int['X']) for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

# Try to find the appropriate data file
if os.path.exists(os.path.join(opts.path, 'new_sequences.csv')):
    # Post-deadline data format
    test = pd.read_csv(os.path.join(opts.path, 'new_sequences.csv'))
    submission = pd.read_csv(os.path.join(opts.path, 'new_sequences_submission.csv'))
    print(f"Loaded post-deadline evaluation data: {len(test)} sequences")
elif os.path.exists(os.path.join(opts.path, 'test.json')):
    # Original competition format
    test = pd.read_json(os.path.join(opts.path, 'test.json'), lines=True)
    submission = pd.read_csv(os.path.join(opts.path, 'sample_submission.csv'))
    print(f"Loaded original test data: {len(test)} sequences")
else:
    raise FileNotFoundError(f"Could not find test data in {opts.path}")

print(f"Data columns: {test.columns}")

# Load precomputed RiNALMo embeddings from shared .pt file (used in pseudo_predict_v4.py)
embedding_data = torch.load('data/precomputed_embeddings.pt')
embedding_dict = dict(zip(embedding_data["ids"], embedding_data["embeddings"]))

# Enforce 2D shape explicitly (authoritative match with pseudo_predict)
for key in embedding_dict:
    embedding = embedding_dict[key]
    if embedding.ndim == 3:
        embedding_dict[key] = embedding.squeeze(0)
    elif embedding.ndim != 2:
        raise RuntimeError(f"Invalid embedding shape for ID {key}: {embedding.shape}")

print(f"Loaded {len(embedding_dict)} precomputed embeddings.")

# Determine sequence length based on actual sequence data
if 'sequence' in test.columns:
    # Add sequence length column if it doesn't exist
    if 'seq_length' not in test.columns:
        test['seq_length'] = test['sequence'].apply(len)
        print(f"Added sequence length column. Length range: {test.seq_length.min()}-{test.seq_length.max()}")

    # Check for variable-length sequences
    max_seq_len = test.seq_length.max()
    use_sliding_window = max_seq_len > 130

    if use_sliding_window:
        print(f"Detected variable-length sequences (max: {max_seq_len}). Will use sliding window.")
        ls_indices = test.seq_length > 0  # Treat all sequences as 'long' when using sliding window
        ss_indices = np.zeros(len(test), dtype=bool)  # Explicitly no short sequences
    else:
        # Current logic to be replaced:
        # ls_indices = test.seq_length == 130
        # ss_indices = test.seq_length == 107

        # Corrected logic (EXPLICIT authoritative fix):
        ls_indices = test['seq_length'] == 130
        ss_indices = test['seq_length'] == 107

        print(f"Long sequences detected: {ls_indices.sum()} (130bp)")
        print(f"Short sequences detected: {ss_indices.sum()} (107bp)")

        # Double-check explicitly:
        if ls_indices.sum() + ss_indices.sum() != len(test):
            raise ValueError(
                "Short and long indices do not cover all sequences in test data. Check the sequence lengths explicitly.")

        print(f"Found {sum(ls_indices)} long sequences (130bp)")
        print(f"Found {sum(ss_indices)} short sequences (107bp)")

        if sum(ls_indices) == 0 and sum(ss_indices) == 0:
            print("No sequences match standard lengths. Processing all as single group.")
            ls_indices = np.ones(len(test), dtype=bool)
            ss_indices = np.zeros(len(test), dtype=bool)
else:
    # Fall back to processing all sequences as one type
    print("Warning: No sequence column found in data - using all sequences")
    ls_indices = np.ones(len(test), dtype=bool)
    ss_indices = np.zeros(len(test), dtype=bool)
    use_sliding_window = False

# Handle datasets safely:
if ls_indices.sum() > 0:
    long_data = test[ls_indices]
    ids = np.asarray(long_data.id.to_list())

    # Explicit preprocessing of inputs outside RNADataset
    data = preprocess_inputs(long_data)
    if isinstance(data, np.ndarray) and data.dtype == object:
        print("Using object-type array format for variable-length sequences")
    else:
        data = data.reshape(1, *data.shape)

    if ls_indices.sum() > 0:
        long_data = test[ls_indices]
        ids = np.asarray(long_data.id.to_list())

        os.makedirs(os.path.join(opts.path, "new_sequences_bpps"), exist_ok=True)

        original_long_dataset = RNADataset(
            seqs=long_data.sequence.to_list(),
            labels=np.zeros(len(long_data)),
            ids=ids,
            ew=np.arange(len(long_data)),
            bpp_path=opts.path,
            training=False,
            k=opts.kmers[0],
        )
        long_dataset = BPPSafeDataset(EmbeddedRNADataset(original_long_dataset, embedding_dict))

        long_dataloader = DataLoader(
            long_dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            collate_fn=variable_length_collate_fn
        )
    else:
        print("No long sequences found.")
        long_dataloader = []

else:
    print("No long sequences found.")
    long_dataloader = []

# Corrected authoritative handling of short sequences (length=107)
if ss_indices.sum() > 0:
    short_data = test[ss_indices]
    ids_short = np.asarray(short_data.id.to_list())

    original_short_dataset = RNADataset(
        seqs=short_data.sequence.to_list(),
        labels=np.zeros(len(short_data)),
        ids=ids_short,
        ew=np.arange(len(short_data)),
        bpp_path=opts.path,
        training=False,
        k=opts.kmers[0]
    )
    short_dataset = BPPSafeDataset(EmbeddedRNADataset(original_short_dataset, embedding_dict))

    short_dataloader = DataLoader(
        short_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        collate_fn=variable_length_collate_fn
    )
else:
    print("No short sequences of length 107 detected. Skipping short sequence processing explicitly.")


nts='ACGU().BEHIMSX'
preds = []
ids = []

with torch.no_grad():
    for batch in tqdm(long_dataloader, desc="Processing long sequences"):
        sequences = batch['embedding'].to(device)
        sample_id = batch['id']
        bpp_single = batch['bpp']
        if isinstance(bpp_single, np.ndarray):
            bpp_single = torch.tensor(bpp_single).float()
        bpp_single = bpp_single.to(device).float()
        batch_size, _, seq_len, _ = bpp_single.shape

        if seq_len > 130:
            windows = process_with_sliding_window(
                fold_models, sequences, bpp_single, window_size=130, stride=65
            )

            merged = torch.zeros(
                (len(fold_models), batch_size, seq_len, fold_models[0].module.decoder.out_features),
                device=sequences.device
            )
            counts = torch.zeros(seq_len, device=sequences.device)

            for start_idx, end_idx, window_preds in windows:
                merged[:, :, start_idx:end_idx, :] += window_preds
                counts[start_idx:end_idx] += 1

            for pos in range(seq_len):
                if counts[pos] > 0:
                    merged[:, :, pos, :] /= counts[pos]

            preds.append(merged.cpu().numpy())
        else:
            outputs = []
            for model in fold_models:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    src_mask = batch['src_mask'].to(device)
                    pred = model(sequences, bpp_single, src_mask)
                outputs.append(pred.cpu())

            preds.append(torch.stack(outputs, 0).numpy())

        ids.extend(sample_id)

# Authoritative handling of short sequences:
ids_short = []
preds_short = []

ids_short = []
preds_short = []

with torch.no_grad():
    for batch in tqdm(short_dataloader, desc="Processing short sequences"):
        sequences = batch['embedding'].to(device)
        sample_id = batch['id']
        bpp_single = batch['bpp']
        if isinstance(bpp_single, np.ndarray):
            bpp_single = torch.tensor(bpp_single).float()
        bpp_single = bpp_single.to(device).float()
        batch_size, _, seq_len, _ = bpp_single.shape

        outputs = []
        for model in fold_models:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                src_mask = batch['src_mask'].to(device)
                pred = model(sequences, bpp_single, src_mask)
            outputs.append(pred.cpu())

        preds_short.append(torch.stack(outputs, 0).numpy())
        ids_short.extend(sample_id)

# Explicit merge of short and long predictions
ids_combined = ids + ids_short
preds_combined = []

for batch_preds in preds + preds_short:
    batch_size = batch_preds.shape[1]
    for batch_idx in range(batch_size):
        preds_combined.append(batch_preds[:, batch_idx])

preds_to_csv = [[] for _ in range(len(test))]
test_ids = test.id.to_list()

print(f"Length of test_ids: {len(test_ids)}, Length of ids_combined: {len(ids_combined)}, Length of preds_combined: {len(preds_combined)}")
assert len(ids_combined) == len(preds_combined), "Mismatch between IDs and predictions lengths!"

preds_to_csv = [[] for _ in range(len(test))]
test_ids = test.id.to_list()

for i, seq_id in enumerate(ids_combined):
    if seq_id in test_ids:
        index = test_ids.index(seq_id)
        preds_to_csv[index].append(preds_combined[i])
    else:
        raise ValueError(f"Sequence ID {seq_id} from predictions not found in test_ids.")



to_csv=[]
for i in tqdm(range(len(preds_to_csv))):
    # Replace line 511 with a more robust solution that normalizes tensor shapes

    # First, normalize the tensor dimensions before averaging:
    if preds_to_csv[i]:
        # Extract the prediction tensor
        pred_tensor = preds_to_csv[i][0]

        # Normalize based on exact tensor shape
        if pred_tensor.ndim == 4:  # (num_models, batch, seq_len, features)
            # Ensure batch dim is 1 before processing
            if pred_tensor.shape[1] == 1:
                to_write = np.mean(pred_tensor, axis=0)[0]  # Average across models, select single batch
            else:
                # Handle unexpected batch dimension
                to_write = np.mean(pred_tensor.reshape(pred_tensor.shape[0], -1, pred_tensor.shape[-1]), axis=0)
        elif pred_tensor.ndim == 3:  # (num_models, seq_len, features)
            to_write = np.mean(pred_tensor, axis=0)  # Average across models directly
        else:
            # Fallback with explicit shape info for debugging
            raise ValueError(f"Unexpected prediction shape: {pred_tensor.shape} for sequence {i}")
    else:
        raise ValueError(f"Missing predictions for sequence at index {i} with id {test_ids[i]}")

    for vector in to_write:
        to_csv.append(vector)
to_csv=np.asarray(to_csv)

# avail_packages=['vienna_2', 'nupack', 'contrafold_2', 'eternafold', 'rnastructure','rna_soft']
avail_packages=['contrafold_2', 'eternafold', 'nupack', 'rnastructure', 'vienna_2', 'rnasoft']
sample_submission_path = os.path.join(opts.path, 'sample_submission.csv')
if not os.path.exists(sample_submission_path):
    raise FileNotFoundError(f"Could not find required template file: {sample_submission_path}")

submission = pd.read_csv(sample_submission_path)
# Define the output path for the new sequences submission
new_sequences_submission_path = os.path.join(sub_folder, 'new_sequences_submission.csv')

# Create ensemble submission from all fold predictions
print("Creating main ensemble submission from all fold predictions")

# Initialize the main submission with zeros
main_submission = submission.copy()
prediction_columns = main_submission.columns[1:]  # Skip ID column
for col in prediction_columns:
    main_submission[col] = 0.0

# Collect all fold submission files that were successfully created
fold_submissions = []
for f in range(opts.nfolds):
    fold_file = f'{sub_folder}/submission_fold{f}.csv'
    if os.path.exists(fold_file):
        fold_submissions.append(pd.read_csv(fold_file))

if fold_submissions:
    # Average predictions across all folds
    for col in prediction_columns:
        main_submission[col] = sum(fs[col] for fs in fold_submissions) / len(fold_submissions)

    print(f"Created ensemble submission from {len(fold_submissions)} fold predictions")
else:
    print("Warning: No fold submissions found, creating fallback submission")

    # Fallback approach: use direct ensemble averaging
    avail_packages = ['contrafold_2', 'eternafold', 'nupack', 'rnastructure', 'vienna_2', 'rnasoft']

    # Get all test sequence IDs and create mappings to submission rows
    id_to_rows = {}
    for idx, row in submission.iterrows():
        id_seqpos = row['id_seqpos']
        seq_id, pos = id_seqpos.rsplit('_', 1)
        if seq_id not in id_to_rows:
            id_to_rows[seq_id] = []
        id_to_rows[seq_id].append((idx, int(pos)))

    # Average predictions directly from the original predictions
    for i, test_id in enumerate(test_ids):
        if i < len(preds_to_csv) and preds_to_csv[i]:
            # Average across folds
            avg_pred = np.mean([preds_to_csv[i][0][f] for f in range(min(opts.nfolds, len(preds_to_csv[i][0])))],
                               axis=0)

            # Extract predictions for each position
            if test_id in id_to_rows:
                for idx, pos in id_to_rows[test_id]:
                    if len(avg_pred.shape) == 4 and pos < avg_pred.shape[2]:  # (1, 1, seq_len, 5)
                        pred_values = avg_pred[0, 0, pos]
                    elif len(avg_pred.shape) == 3 and pos < avg_pred.shape[1]:  # (1, seq_len, 5)
                        pred_values = avg_pred[0, pos]
                    else:
                        continue

                    # Assign to all prediction columns
                    for j, col in enumerate(prediction_columns):
                        main_submission.loc[idx, col] = pred_values[j % 5]

# Concatenation is now handled conditionally in the block above

# Single package predictions using ViennaRNA exclusively
if to_csv.shape[1] != 5:
    raise ValueError(
        f"Expected exactly 5 columns for ViennaRNA predictions, but got {to_csv.shape[1]}"
    )

# Ensure we have the correct number of predictions
expected_rows = len(submission)
if len(to_csv) != expected_rows:
    print(f"Warning: Prediction count ({len(to_csv)}) doesn't match submission rows ({expected_rows})")

    if len(to_csv) > expected_rows:
        # Truncate to expected length
        to_csv = to_csv[:expected_rows]
    else:
        # Pad with zeros if needed
        padding = np.zeros((expected_rows - len(to_csv), to_csv.shape[1]))
        to_csv = np.vstack([to_csv, padding])

# Create a copy of the submission template
pkg_sub = submission.copy()

# Extract column names excluding the ID column
prediction_columns = pkg_sub.columns[1:]

# Assign the predictions explicitly to submission columns
for i, col in enumerate(prediction_columns):
    pkg_sub[col] = to_csv[:, i % to_csv.shape[1]]

# Save the single package submission explicitly as ViennaRNA only
pkg_sub.to_csv(f"{sub_folder}/viennaRNA_submission.csv", index=False)
print(f"Saved ViennaRNA-only submission to {sub_folder}/viennaRNA.csv")


# Compute mean across all package predictions for final submission
final_predictions = to_csv.mean(1)

# Ensure dimensions match submission template
if len(final_predictions) != len(submission):
    print(
        f"Warning: Final prediction count ({len(final_predictions)}) doesn't match submission rows ({len(submission)})")

    if len(final_predictions) > len(submission):
        final_predictions = final_predictions[:len(submission)]
    else:
        padding = np.zeros(len(submission) - len(final_predictions))
        final_predictions = np.concatenate([final_predictions, padding])

# Create a copy of the submission template
main_submission = submission.copy()

# Extract column names excluding the ID column
prediction_columns = main_submission.columns[1:]

# Populate all prediction columns with the same values
for col in prediction_columns:
    main_submission[col] = final_predictions

# Save the main submission file
main_submission.to_csv(f'{sub_folder}/submission.csv', index=False)
print(f"Saved main submission to {sub_folder}/submission.csv")

# Create package-specific submissions (optional)
for pkg_idx, pkg in enumerate(avail_packages):
    pkg_sub = main_submission.copy()

    # Use package's column as the prediction for all columns
    if len(prediction_columns) >= pkg_idx + 1:
        base_col = prediction_columns[pkg_idx]
        for col in prediction_columns:
            pkg_sub[col] = pkg_sub[base_col]

    pkg_sub.to_csv(f"{sub_folder}/{pkg}.csv", index=False)
    print(f"Saved {pkg} submission to {sub_folder}/{pkg}.csv")

for f in range(opts.nfolds):
    to_csv = []
    fold_preds = []
    position_mapping = {}  # Maps sequence positions to global indices
    current_position = 0

    for i in tqdm(range(len(preds_to_csv))):
        # Extract fold-specific predictions
        if len(preds_to_csv[i]) > 0:
            to_write = np.asarray(preds_to_csv[i][0][f])
            fold_preds.append(to_write)

            # Process tensors based on their actual shape
            if len(to_write.shape) == 4:  # Shape: (1, 1, seq_length, 5)
                seq_length = to_write.shape[2]
                # Reshape to (seq_length, 5) for easier processing
                reshaped = to_write.reshape(seq_length, 5)

                # Store each position prediction and record its mapping
                for pos in range(seq_length):
                    to_csv.append(reshaped[pos])
                    position_mapping[(i, pos)] = current_position + pos

                current_position += seq_length

            elif len(to_write.shape) == 3:  # Shape: (1, seq_length, 5)
                seq_length = to_write.shape[1]
                # Reshape to (seq_length, 5) for easier processing
                reshaped = to_write.reshape(seq_length, 5)

                for pos in range(seq_length):
                    to_csv.append(reshaped[pos])
                    position_mapping[(i, pos)] = current_position + pos

                current_position += seq_length

            elif len(to_write.shape) == 2:  # Already in (seq_length, 5) format
                seq_length = to_write.shape[0]

                for pos in range(seq_length):
                    to_csv.append(to_write[pos])
                    position_mapping[(i, pos)] = current_position + pos

                current_position += seq_length

            else:
                print(f"Warning: Unexpected shape {to_write.shape} for fold {f}, sequence {i}")
                # Single placeholder as fallback
                to_csv.append(np.zeros(5))
                position_mapping[(i, 0)] = current_position
                current_position += 1

    # Convert predictions to numpy array
    if to_csv:
        to_csv = np.asarray(to_csv)
        print(f"Fold {f} predictions shape: {to_csv.shape}")
    else:
        print(f"Warning: No predictions generated for fold {f}")
        continue

    # Extract ID_seqpos from submission file
    id_seqpos_list = submission['id_seqpos'].tolist()
    pred_cols = submission.columns[1:]  # Skip ID column

    # Create a mapping dictionary from id_seqpos to row index
    id_to_row = {id_seqpos: idx for idx, id_seqpos in enumerate(id_seqpos_list)}

    # Create per-position predictions dataframe
    fold_submission = submission.copy()

    # Initialize with zeros
    for col in pred_cols:
        fold_submission[col] = 0.0

    # Map sequence positions to submission rows
    for seq_idx, test_id in enumerate(test_ids):
        if seq_idx in [idx for (idx, _) in position_mapping.keys()]:
            seq_len = test.iloc[seq_idx].seq_length if 'seq_length' in test.columns else len(
                test.iloc[seq_idx].sequence)

            for pos in range(seq_len):
                id_seqpos = f"{test_id}_{pos}"
                if id_seqpos in id_to_row and (seq_idx, pos) in position_mapping:
                    pos_idx = position_mapping[(seq_idx, pos)]
                    if pos_idx < len(to_csv):
                        # Apply same prediction to all columns (can be refined if needed)
                        for i, col in enumerate(pred_cols):
                            fold_submission.loc[id_to_row[id_seqpos], col] = to_csv[pos_idx][i % 5]

    # Create submission for this fold
    fold_submission = submission.copy()
    prediction_columns = fold_submission.columns[1:]

    # Assign to all prediction columns
    for col in prediction_columns:
        fold_submission[col] = to_csv[:, i % to_csv.shape[1]]

    fold_submission.to_csv(f'{sub_folder}/submission_fold{f}.csv', index=False)

    # Save raw predictions
    with open(f'{sub_folder}/predictions_fold{f}.p', 'wb+') as file:
        pickle.dump(fold_preds, file)