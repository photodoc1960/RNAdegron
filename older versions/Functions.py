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

def get_train_val_indices(df,fold,SEED=2020,nfolds=5):
    splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    splits = list(splits.split(df.sequence,df.SN_filter))
    train_indices=splits[fold][0]
    val_indices=splits[fold][1]
    val_indices=val_indices[np.asarray(df.signal_to_noise)[val_indices]>1]
    return train_indices,val_indices

def get_train_val_indices_PL(data, fold, SEED=2020,nfolds=5):
    splits = KFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    splits = list(splits.split(data))
    train_indices=splits[fold][0]
    val_indices=splits[fold][1]
    #val_indices=val_indices[np.asarray(df.signal_to_noise)[val_indices]>1]
    return train_indices,val_indices


def get_best_weights_from_fold(fold, mode='supervised', top=5):
    if mode == 'pretrain':
        csv_file = 'logs/pretrain.csv'
        checkpoints_folder = 'pretrain_weights'
    else:
        csv_file = 'log_fold{}.csv'.format(fold)
        checkpoints_folder = 'checkpoints_fold{}'.format(fold)

    try:
        history = pd.read_csv(csv_file)
        scores = np.asarray(-history.val_loss)
        top_epochs = scores.argsort()[-top:][::-1]
        print(scores[top_epochs])
        os.makedirs('best_weights', exist_ok=True)

        for i in range(min(top, len(top_epochs))):
            weights_path = '{}/epoch{}.ckpt'.format(checkpoints_folder, history.epoch[top_epochs[i]])
            print(weights_path)
            if os.path.exists(weights_path):
                os.system('cp {} best_weights/{}top{}.ckpt'.format(
                    weights_path,
                    'pretrain_' if mode == 'pretrain' else f'fold{fold}',
                    i + 1))
            else:
                print(f"Warning: {weights_path} not found")
    except FileNotFoundError:
        print(f"Warning: {csv_file} not found")


def mutate_dna_sequence(sequence,nmute=15):
    perm = torch.randperm(sequence.size(1))
    to_mutate = perm[:nmute]
    mutation=torch.randint(4,size=(sequence.size(0),nmute))
    sequence[:,to_mutate]=mutation
    return sequence


def mutate_rna_input(sequence, nucleotide_tokens=[5,6,7,8], mask_count=1):
    mutated_src = sequence.clone()
    mask_positions = torch.zeros(sequence.size(0), sequence.size(1), dtype=torch.bool, device=sequence.device)

    for i in range(sequence.size(0)):
        nucleotide_positions = torch.isin(sequence[i, :, 0], torch.tensor(nucleotide_tokens, device=sequence.device))
        possible_positions = torch.where(nucleotide_positions)[0]

        if possible_positions.numel() == 0:
            continue  # explicitly handle rare cases with no nucleotides

        num_to_mutate = min(mask_count, possible_positions.numel())
        mutated_indices = possible_positions[torch.randperm(len(possible_positions))[:num_to_mutate]]

        for idx in mutated_indices:
            original_nucleotide = mutated_src[i, idx, 0].item()
            choices = [n for n in nucleotide_tokens if n != original_nucleotide]
            mutated_src[i, idx, 0] = np.random.choice(choices)
            mask_positions[i, idx] = True

    return mutated_src, mask_positions

def mask_rna_input(sequence, mask_token=4, mask_ratio=0.15, mask_count=None):
    masked_src = sequence.clone()
    mask_positions = torch.zeros(sequence.size(0), sequence.size(1), dtype=torch.bool, device=sequence.device)

    for i in range(sequence.size(0)):
        nucleotide_positions = (sequence[i, :, 0] >= 5) & (sequence[i, :, 0] <= 8)
        possible_positions = torch.where(nucleotide_positions)[0]

        if possible_positions.numel() == 0:
            continue  # Skip explicitly if no nucleotide positions exist (rare).

        # Explicitly authoritative fix: calculate number to mask explicitly
        if mask_count is None:
            num_to_mask = max(1, int(mask_ratio * possible_positions.numel()))
        else:
            num_to_mask = min(mask_count, possible_positions.numel())

        masked_indices = possible_positions[torch.randperm(len(possible_positions))[:num_to_mask]]
        masked_src[i, masked_indices, 0] = mask_token
        mask_positions[i, masked_indices] = True

    return masked_src, mask_positions

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
    model.eval()
    val_loss = 0
    step = 0
    criterion = weighted_MCRMSE
    with torch.no_grad():
        for data in val_dataloader:
            bpps = data['bpp'].to(device)
            src = data['data'].to(device)
            src_mask = data['src_mask'].to(device)
            labels = data['labels'].to(device)
            ew = data['ew'].to(device)

            # Explicitly fix BPPS dimensions
            if bpps.dim() == 3:
                bpps = bpps.unsqueeze(1).repeat(1, 4, 1, 1)
            elif bpps.dim() == 4 and bpps.shape[1] == 1:
                bpps = bpps.repeat(1, 4, 1, 1)
            elif bpps.dim() == 5 and bpps.shape[1] == 1:
                bpps = bpps[:, 0, :, :, :].repeat(1, 4, 1, 1)

            if src.dim() == 4 and src.shape[1] == 1:
                src = src[:, 0, :, :]
            if src_mask.dim() == 3 and src_mask.shape[1] == 1:
                src_mask = src_mask[:, 0, :]

            # Authoritative model call (matches Network.py definition)
            embeddings = data['embedding'].to(device)
            output = model(embeddings, bpps, src_mask)

            loss = criterion(output[:, :labels.shape[1]], labels, ew).mean()

            if torch.isnan(loss):
                print("Warning: NaN encountered in validation at step", step)
                continue

            val_loss += loss.item()
            step += 1

    val_loss /= max(step, 1)
    print(f"Validation Loss computed: {val_loss}")
    return val_loss

def revalidate(model, device, dataset, batch_size=64):
    batches = len(dataset)
    loss = 0
    criterion = MCRMSE
    model.eval()

    with torch.no_grad():
        for data in tqdm(dataset):
            embeddings = data['embedding'].to(device)
            Y = data['labels'].to(device).float()
            bpps = data['bpp'].to(device).float()
            src_mask = data['src_mask'].to(device)
            if src_mask.dim() == 3 and src_mask.shape[1] == 1:
                src_mask = src_mask[:, 0, :]

            output = model(embeddings, bpps, src_mask)[:, :68]

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
