import torch
from Dataset_v4 import RNADataset
from torch.utils.data import DataLoader

model.eval()  # explicitly ensure RiNALMo model is in eval mode
embeddings_dict = {}

for data in dataloader:
    sequences = data['data'].to(device).long()[:,:,0]  # RNA tokens only
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        embeddings = rinalmo(sequences)['representation']
    for idx, emb in zip(data['id'], embeddings.cpu()):
        embeddings_dict[idx] = emb.numpy()

# Explicitly save embeddings
torch.save(embeddings_dict, 'precomputed_embeddings.pt')
