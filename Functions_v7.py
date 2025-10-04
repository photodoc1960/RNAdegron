import torch
import os
from sklearn import metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import Metrics
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import KFold, StratifiedKFold


def get_distance_mask(L):

    m=np.zeros((3,L,L))

    for i in range(L):
        for j in range(L):
            for k in range(3):
                if abs(i-j)>0:
                    m[k,i,j]=1/abs(i-j)**(k+1)
    return m

def aug_data(df,aug_df):
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]
    ids= new_df.id.to_list()
    indices=[]
    for id in df.id:
        indices.append(ids.index(id))
    indices=np.asarray(indices)

    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = new_df[df.columns]
    return df, indices

def get_alt_structures(df):
    """
    columns in the order of 'sequence', 'structure', 'predicted_loop_type'
    """
    # pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    #train = pd.read_json(path, lines=True)

    folders=['nupack','rnastructure','vienna_2',
            'contrafold_2',]

    token2int = {x:i for i, x in enumerate('ACGU().BEHIMSX')}



    def preprocess_inputs(df, cols):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    inputs=[]
    for folder in folders:
        columns=['sequence',folder,folder+'_loop']
        inputs.append(preprocess_inputs(df,columns))
    inputs=np.asarray(inputs)
    #train_inputs = preprocess_inputs(train)
    #train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
    return inputs

def get_alt_structures_50C(df):
    """
    columns in the order of 'sequence', 'structure', 'predicted_loop_type'
    """
    # pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    #train = pd.read_json(path, lines=True)

    folders=['eternafold','nupack','rnastructure','vienna_2',
            'contrafold_2',]

    token2int = {x:i for i, x in enumerate('ACGU().BEHIMSX')}



    def preprocess_inputs(df, cols):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    inputs=[]
    for folder in folders:
        columns=['sequence',folder,folder+'_loop']
        inputs.append(preprocess_inputs(df,columns))
    inputs=np.asarray(inputs)
    #train_inputs = preprocess_inputs(train)
    #train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
    return inputs


def MCRMSE(y_pred,y_true):
    colwise_mse = torch.mean(torch.square(y_true - y_pred), axis=1)
    MCRMSE = torch.mean(torch.sqrt(colwise_mse), axis=1)
    return MCRMSE

def weighted_MCRMSE(y_pred,y_true,ew):
    colwise_mse = torch.mean(ew*torch.square(y_true - y_pred), axis=1)
    MCRMSE = torch.mean(torch.sqrt(colwise_mse), axis=1)
    return MCRMSE

def get_errors(df, cols=['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',
       'deg_error_Mg_50C', 'deg_error_50C']):
    return np.transpose(
        np.array(
            df[cols]
            .values
            .tolist()
        ),
        (0, 2, 1)
    )

def get_data(train):
    pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    #train = pd.read_json(path, lines=True)

    token2int = {x:i for i, x in enumerate('ACGU().BEHIMSX')}

    def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    train_inputs = preprocess_inputs(train)
    train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
    return train_inputs,train_labels

def get_test_data(path):
    pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    test = pd.read_json(path, lines=True)

    token2int = {x:i for i, x in enumerate('ACGU().BEHIMSX')}

    def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    test_inputs = preprocess_inputs(test)
    return test_inputs

def get_train_val_indices(df, fold, SEED=2020, nfolds=5, min_val_samples=10):
    """
    Get train/validation indices with guaranteed minimum validation samples.

    Args:
        df: DataFrame with sequence data
        fold: Current fold number
        SEED: Random seed for reproducibility
        nfolds: Number of folds
        min_val_samples: Minimum number of validation samples required
    """
    splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    splits = list(splits.split(df.sequence, df.SN_filter))
    train_indices = splits[fold][0]
    val_indices = splits[fold][1]

    # Apply signal-to-noise filtering
    high_snr_mask = np.asarray(df.signal_to_noise)[val_indices] > 1
    filtered_val_indices = val_indices[high_snr_mask]

    # Ensure minimum validation samples
    if len(filtered_val_indices) < min_val_samples:
        print(f"Warning: Fold {fold} has only {len(filtered_val_indices)} high-SNR validation samples.")
        print(f"Using top {min_val_samples} validation samples by signal-to-noise ratio.")

        # Get signal-to-noise ratios for all validation samples
        val_snr = np.asarray(df.signal_to_noise)[val_indices]

        # Sort by SNR and take top samples
        sorted_indices = np.argsort(val_snr)[::-1]  # Descending order
        top_indices = sorted_indices[:min(min_val_samples, len(val_indices))]
        filtered_val_indices = val_indices[top_indices]

        print(f"Selected {len(filtered_val_indices)} validation samples with SNR range: "
              f"{val_snr[top_indices].min():.3f} - {val_snr[top_indices].max():.3f}")

    return train_indices, filtered_val_indices


def get_best_weights_from_fold(fold, mode='supervised', top=5):
    if mode == 'pretrain':
        csv_file = 'logs/pretrain.csv'
        checkpoints_folder = 'pretrain_weights'
    else:
        csv_file = 'log_fold{}.csv'.format(fold)
        checkpoints_folder = 'checkpoints_fold{}'.format(fold)

    try:
        history = pd.read_csv(csv_file)
        # CRITICAL FIX: Handle sparse epoch numbering and NaN values
        valid_history = history.dropna(subset=['val_loss'])
        # Ensure val_loss column contains numeric data
        valid_history['val_loss'] = pd.to_numeric(valid_history['val_loss'], errors='coerce')
        valid_history = valid_history.dropna(subset=['val_loss'])
        # Filter out placeholder zero validation losses
        valid_history = valid_history[valid_history['val_loss'] > 0.0]

        if len(valid_history) == 0:
            print("Warning: No valid validation scores found")
            return

        # Convert to negative scores for sorting (lower loss = higher score)
        scores = -valid_history['val_loss'].values
        epochs = valid_history['epoch'].values

        # Get top performing indices
        top_count = min(top, len(scores))
        top_indices = np.argsort(scores)[-top_count:][::-1]

        selected_scores = scores[top_indices]
        selected_epochs = epochs[top_indices]

        print(f"Best scores: {selected_scores}")
        print(f"Corresponding epochs: {selected_epochs}")
        top_epochs = selected_epochs

        os.makedirs('best_weights', exist_ok=True)
        for i, epoch_num in enumerate(top_epochs):
            weights_path = f'{checkpoints_folder}/epoch{epoch_num}.ckpt'
            if os.path.exists(weights_path):
                os.system(
                    f'cp {weights_path} best_weights/{"pretrain_" if mode == "pretrain" else f"fold{fold}"}top{i + 1}.ckpt')
            else:
                print(f"Warning: {weights_path} not found")
    except FileNotFoundError:
        print(f"Warning: {csv_file} not found")
    except Exception as e:
        print(f"Error processing weights: {e}")

def mutate_dna_sequence(sequence,nmute=15):
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    mutation=torch.randint(4,size=(sequence.size(0),nmute))
    sequence[:,to_mutate]=mutation
    return sequence

def mutate_rna_input(sequence,nmute=.15):
    nmute=int(sequence.shape[1]*nmute)
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    sequence_mutation=torch.randint(4,size=(sequence.size(0),nmute))
    structure_mutation=torch.randint(4,7,size=(sequence.size(0),nmute))
    d_mutation=torch.randint(7,14,size=(sequence.size(0),nmute))
    mutated=sequence.clone()
    mutated[:,to_mutate,0]=sequence_mutation
    mutated[:,to_mutate,1]=structure_mutation
    mutated[:,to_mutate,2]=d_mutation
    return mutated

def mask_rna_input(sequence,nmute=.15):
    nmute=int(sequence.shape[1]*nmute)
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    masked=sequence.clone()
    masked[:,to_mutate,:]=14
    return masked

def get_MLM_mask(sequence,nmask=12):
    mask=np.zeros(sequence.shape,dtype='bool')
    to_mask=np.random.choice(len(sequence[0]),size=(nmask),replace=False)
    mask[:,to_mask]=True
    return mask

def get_complementary_sequence(sequence):
    complementary_sequence=sequence.copy()
    complementary_sequence[sequence==0]=1
    complementary_sequence[sequence==1]=0
    complementary_sequence[sequence==2]=3
    complementary_sequence[sequence==3]=2
    complementary_sequence=complementary_sequence[:,::-1]
    return complementary_sequence

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
    return lr


def save_weights(model, optimizer, epoch, folder, is_best=False, val_loss=None):
    """Save model checkpoint with optional best model tracking.

    Args:
        model: Model state dict to save
        optimizer: Optimizer state to save (unused in current implementation)
        epoch: Current epoch number (zero-indexed)
        folder: Directory path for checkpoint storage
        is_best: Flag indicating if current model has best validation performance
        val_loss: Validation loss value (for logging in best model info)
    """
    if os.path.isdir(folder) == False:
        os.makedirs(folder, exist_ok=True)

    # Always save the standard epoch-numbered checkpoint
    torch.save(model.state_dict(), f"{folder}/epoch{epoch + 1}.ckpt")

    # Additionally save best model checkpoint if flagged
    if is_best:
        best_path = f"{folder}/best_model.ckpt"
        os.system(f"cp {folder}/epoch{epoch + 1}.ckpt {best_path}")

        # Optionally store metadata about the best checkpoint
        if val_loss is not None:
            with open(f"{folder}/best_model_info.txt", "w") as f:
                f.write(f"Epoch: {epoch + 1}\nValidation Loss: {val_loss}")


def validate(model, device, val_dataloader, batch_size):
    """
    CRITICAL FIX: Updated validation function for unified BPP tensor structure compatibility.
    
    Handles unified (batch_size, 4, seq_len, seq_len) BPP format where:
    - Channel 0: Base pairing probabilities  
    - Channels 1-3: Distance masks
    """
    model.eval()
    val_loss = 0
    step = 0
    criterion = weighted_MCRMSE
    with torch.no_grad():
        for data in val_dataloader:
            bpps = data['bpp'].to(device)
            src = data['embedding'].to(device)
            src_mask = data['src_mask'].to(device)
            labels = data['labels'].to(device)
            ew = data['ew'].to(device)

            # CRITICAL FIX: Handle unified BPP tensor structure
            # Expected format: (batch_size, 4, seq_len, seq_len) 
            # where channels = [BPP, distance_mask_1, distance_mask_2, distance_mask_3]

            # FIXED: Handle both training and inference BPP formats consistently
            if bpps.dim() == 5:  # Inference: (batch_size, num_variants, 4, seq_len, seq_len)
                bpps = bpps[:, 0, :, :, :]  # -> (batch_size, 4, seq_len, seq_len)
            elif bpps.dim() == 4 and bpps.shape[1] == 4:  # Training: (batch_size, 4, seq_len, seq_len)
                pass  # Already correct format
            elif bpps.dim() == 3:  # Legacy (batch_size, seq_len, seq_len)
                print("Warning: Legacy BPP format detected, converting to unified structure")
                batch_size, seq_len = bpps.shape[0], bpps.shape[1]
                # Generate distance masks and create unified structure
                dm = get_distance_mask(seq_len)
                dm_tensor = torch.tensor(dm, device=device, dtype=bpps.dtype)
                dm_batch = dm_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                bpps = torch.cat([bpps.unsqueeze(1), dm_batch], dim=1)
            elif bpps.dim() == 4 and bpps.shape[1] != 4:
                raise ValueError(f"BPP tensor has unexpected channel count: {bpps.shape[1]}, expected 4")
            
            # Final validation of BPP structure
            if bpps.dim() != 4 or bpps.shape[1] != 4:
                raise ValueError(f"BPP tensor must have shape (batch_size, 4, seq_len, seq_len), got {bpps.shape}")

            # Handle source mask dimensions
            if src_mask.dim() == 3 and src_mask.shape[1] == 1:
                src_mask = src_mask[:, 0, :]

            # Extract structural features
            deltaG = data['deltaG'].to(device)
            graph_dist = data['graph_dist'].to(device)
            nearest_p = data['nearest_p'].to(device)
            nearest_up = data['nearest_up'].to(device)
            
            # Forward pass with unified architecture
            output = model(src, bpps, src_mask,
                           deltaG,
                           graph_dist,
                           nearest_p,
                           nearest_up)
            loss = criterion(output[:, :labels.shape[1]], labels, ew).mean()

            # FIXED: Proper NaN handling - don't increment step counter for invalid batches
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at validation step {step}")
                print(
                    f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, nan_count={torch.isnan(output).sum()}")
                print(
                    f"  Label stats: min={labels.min():.4f}, max={labels.max():.4f}, nan_count={torch.isnan(labels).sum()}")
                print(
                    f"  Error weight stats: min={ew.min():.4f}, max={ew.max():.4f}, nan_count={torch.isnan(ew).sum()}")
                continue  # Skip NaN batch without incrementing step

            val_loss += loss.item()
            step += 1

    val_loss /= max(step, 1)  # Prevent division by zero
    print(f"Validation completed: loss={val_loss:.6f}, processed_batches={step}")
    return val_loss

def revalidate(model, device, dataset, batch_size=64):
    batches = len(dataset)
    loss = 0
    criterion = MCRMSE
    model.eval()

    with torch.no_grad():
        for data in tqdm(dataset):
            X = data['data'].to(device).long()
            Y = data['labels'].to(device).float()
            bpps = data['bpp'].to(device).float()
            src_mask = data['src_mask'].to(device)
            if src_mask.dim() == 3 and src_mask.shape[1] == 1:
                src_mask = src_mask[:, 0, :]

            # Authoritative fix applied here:
            output = model(X, bpps, src_mask)[:, :68]

            loss += criterion(output, Y).mean()

    val_loss = (loss / batches).cpu()
    print(val_loss.item())
    return val_loss



def predict(model,device,dataset,batch_size=64):
    batches=int(len(dataset.val_indices)/batch_size)+1
    model.train(False)
    total=0
    ground_truths=dataset.labels[dataset.val_indices]
    predictions=[]
    attention_weights=[]
    loss=0
    criterion=nn.CrossEntropyLoss()
    dataset.switch_mode(training=False)
    dataset.update_batchsize(batch_size)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data=dataset[i]
            X=data['data'].to(device,).long()
            Y=data['labels'].to(device,dtype=torch.int64)
            output,_,_,aw= model(X,None)
            del X
            loss+=criterion(output,Y)
            classification_predictions = torch.argmax(output,dim=1).squeeze()
            for pred in output:
                predictions.append(pred.cpu().numpy())
            for weight in aw:
                attention_weights.append(weight.cpu().numpy())

            del output
    torch.cuda.empty_cache()
    val_loss=(loss/batches).cpu()
    predictions=np.asarray(predictions)
    attention_weights=np.asarray(attention_weights)
    binary_predictions=predictions.copy()
    binary_predictions[binary_predictions==2]=1
    binary_ground_truths=ground_truths.copy()
    binary_ground_truths[binary_ground_truths==2]=1
    return predictions,attention_weights,np.asarray(dataset.data[dataset.val_indices])
