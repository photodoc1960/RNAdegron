import numpy as np
import os
import pandas as pd



def get_best_weights_from_fold(fold, csv_file, weights_path, des, top=1):
    history = pd.read_csv(csv_file)
    valid_scores = history[history.val_loss.notna() & (history.val_loss != -1)]  # explicitly filter out NaNs and placeholder epochs
    scores = -valid_scores.val_loss.values
    top_indices = scores.argsort()[-top:][::-1]

    print(scores[top_indices])
    os.makedirs(des, exist_ok=True)

    for i, idx in enumerate(top_indices):
        epoch = int(valid_scores.iloc[idx].epoch)
        checkpoint_path = f"{weights_path}/epoch{epoch}.ckpt"
        os.system(f'cp {checkpoint_path} {des}/fold{fold}top{i+1}.ckpt')

    return scores[top_indices[0]]

scores=[]
for i in range(5):
    print(i)
    scores.append(get_best_weights_from_fold(i,csv_file=f"logs/log_pl_fold{i}.csv",weights_path=f"weights/checkpoints_fold{i}_pl",des='best_pl_weights'))

with open('cv.txt','w+') as f:
    f.write(str(-np.mean(scores)))
