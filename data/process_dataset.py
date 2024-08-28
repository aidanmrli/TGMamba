# This script processes the EEG data and labels and creates a PyTorch Geometric dataset.
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
print(adj[0:5,0:5])
# Create edge index and edge weight from adjacency matrix
ind = (adj != 0) & (adj != 1)
edge_index = np.argwhere(ind == True).T 
edge_weight = np.zeros((1, edge_index.shape[1]))

for i, e in enumerate(edge_index.T):  
    edge_weight[0, i] = adj[e[0], e[1]]

edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

# This graph basically has 36 undirected/bidirectional edges in the graph because each undirected edge has two directed edges.
# There are 19 electrodes and 72 directional edges in the graph. 
# Every directional edge has its opposite edge in the graph with the same weight.
# The edge index is a 2x72 tensor, where each column represents an edge between two electrodes.
# The edge weight is a 1x72 tensor, where each element represents the weight of the corresponding edge.
# The adjacency matrix is a 19x19 tensor, where each element represents the weight of the edge between two electrodes.

# Load electrode positions
pos = sio.loadmat('/home/data_shared/position.mat')['pos'].T

# Load EEG data and labels
with h5py.File('/home/data_shared/clip_data.h5', "r") as f:
    EEG = np.array(f[list(f.keys())[0]])

with h5py.File('/home/data_shared/label.h5', "r") as f: 
    Label = np.array(f[list(f.keys())[0]])

# NOTE: EVAL Load EEG data and labels 
with h5py.File('/home/data_shared/clip_data_eval.h5', "r") as f:
    EEG_eval = np.array(f[list(f.keys())[0]])

with h5py.File('/home/data_shared/label_eval.h5', "r") as f: 
    Label_eval = np.array(f[list(f.keys())[0]])

assert EEG.shape[0] == Label.shape[0], "Data and label size mismatch"
assert EEG.shape[1] == 10, "Number of time points mismatch"
assert EEG.shape[2] == 19, "Number of electrodes mismatch"
assert EEG.shape[3] == 200, "Number of time bins in each time point mismatch"
# merge (10, 200) -> (2000)
# (B, L=10, V=19, D=200) -> (B, V, L=2000, D=1) 
# 1 feature per node

# Prepare the dataset
dataset = []
# batch = np.ones((1,19))
is_fft = False

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

# Split the dataset into train, validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

test_dataset = []

for idx in tqdm(range(EEG_eval.shape[0])):
    eeg_clip = EEG_eval[idx,:,:,:]
    
    if is_fft:
        eeg_clip = np.log(np.abs(fft(eeg_clip,axis=2)[:,:,0:100]) + 1e-30)
        
    label = Label_eval[idx]

    test_dataset.append(Data(
        x=torch.tensor(eeg_clip).transpose(1,0),
        y=torch.tensor(label, dtype=torch.long),
        # batch=batch,
        edge_weight=edge_weight.squeeze(0),    
        edge_index=edge_index,
        elec_pos=torch.tensor(pos)
    ))

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
home_dir = os.path.dirname(parent_dir)
save_dir = os.path.join(home_dir, 'processed_dataset')
os.makedirs(save_dir, exist_ok=True)
if is_fft:
    torch.save(train_dataset, os.path.join(save_dir, 'train_dataset_fft.pt'))
    torch.save(val_dataset, os.path.join(save_dir, 'val_dataset_fft.pt'))
    torch.save(test_dataset, os.path.join(save_dir, 'test_dataset_fft.pt'))
else:
    torch.save(train_dataset, os.path.join(save_dir, 'train_dataset_raw.pt'))
    torch.save(val_dataset, os.path.join(save_dir, 'val_dataset_raw.pt'))
    torch.save(test_dataset, os.path.join(save_dir, 'test_dataset_raw.pt'))

print(f"Datasets saved in {save_dir}")