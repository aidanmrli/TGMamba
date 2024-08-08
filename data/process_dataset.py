import torch
import numpy as np
from tqdm import tqdm
import h5py
import scipy.io as sio
from torch_geometric.data import Data, Dataset
from scipy.fft import fft
import os

# Load adjacency matrix
adj = sio.loadmat('/home/data_shared/adj_mat.mat')['adj_mat']

# Create edge index and edge weight from adjacency matrix
ind = (adj != 0) & (adj != 1)
edge_index = np.argwhere(ind == True).T 
edge_weight = np.zeros((1, edge_index.shape[1]))

for i, e in enumerate(edge_index.T):  
    edge_weight[0, i] = adj[e[0], e[1]]

edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_weight = torch.tensor(edge_weight, dtype=torch.float)
print(edge_index.shape, edge_weight.shape)
print("Max index in edge_index:", edge_index.max().item())
# Load electrode positions
pos = sio.loadmat('/home/data_shared/position.mat')['pos'].T

# Load EEG data and labels
with h5py.File('/home/data_shared/clip_data.h5', "r") as f:
    EEG = np.array(f[list(f.keys())[0]])

with h5py.File('/home/data_shared/label.h5', "r") as f: 
    Label = np.array(f[list(f.keys())[0]])

assert EEG.shape[0] == Label.shape[0], "Data and label size mismatch"
assert EEG.shape[1] == 10, "Number of time points mismatch"
assert EEG.shape[2] == 19, "Number of electrodes mismatch"
assert EEG.shape[3] == 200, "Number of frequency bins mismatch"
# Prepare the dataset
dataset = []
# batch = np.ones((1,19))
is_fft = True

for idx in tqdm(range(EEG.shape[0])):
    eeg_clip = EEG[idx,:,:,:]
    
    if is_fft:
        eeg_clip = np.log(np.abs(fft(eeg_clip,axis=2)[:,:,0:100]) + 1e-30)
        
    label = Label[idx]

    dataset.append(Data(
        x=torch.tensor(eeg_clip).transpose(1,0),
        y=torch.tensor(label, dtype=torch.long),
        # batch=batch,
        edge_weight=edge_weight.squeeze(0),    
        edge_index=edge_index,
        elec_pos=torch.tensor(pos)
    ))

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

save_dir = 'TGMamba/data/processed_dataset'
os.makedirs(save_dir, exist_ok=True)
torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pt'))
torch.save(val_dataset, os.path.join(save_dir, 'val_dataset.pt'))
torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))

print(f"Datasets saved in {save_dir}")