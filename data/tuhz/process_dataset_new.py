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
# print(adj[0:5,0:5])
# Create edge index and edge weight from adjacency matrix
ind = (adj != 0) & (adj != 1)
edge_index = np.argwhere(ind == True).T 
edge_weight = np.zeros((1, edge_index.shape[1]))

for i, e in enumerate(edge_index.T):  
    edge_weight[0, i] = adj[e[0], e[1]]

edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

# Load electrode positions
pos = sio.loadmat('/home/data_shared/position.mat')['pos'].T

# Load EEG data and labels
with h5py.File('/home/data_shared/clip_data.h5', "r") as f:
    EEG = np.array(f[list(f.keys())[0]])

with h5py.File('/home/data_shared/label.h5', "r") as f: 
    Label = np.array(f[list(f.keys())[0]])

# Load EEG data and labels for evaluation
with h5py.File('/home/data_shared/clip_data_eval.h5', "r") as f:
    EEG_eval = np.array(f[list(f.keys())[0]])

with h5py.File('/home/data_shared/label_eval.h5', "r") as f: 
    Label_eval = np.array(f[list(f.keys())[0]])

assert EEG.shape[0] == Label.shape[0], "Data and label size mismatch"
assert EEG.shape[1] == 10, "Number of time points mismatch"
assert EEG.shape[2] == 19, "Number of electrodes mismatch"
assert EEG.shape[3] == 200, "Number of time bins in each time point mismatch"
# (B, L=10, V=19, D=200) -> (B, V, L=2000, D=1) 
# 1 scalar feature per node or 3 for rgb

# Prepare the dataset
dataset = []

# Calculate mean and std from training data
train_data = EEG.transpose(0, 2, 1, 3).reshape(-1, 19, 20, 100)

# print("Mean_train.shape: ", mean_train.shape)
# print("Std_train.shape: ", std_train.shape)

def normalize_data(data, mean, std):
    return (data - mean) / (std + 1e-10)

for idx in tqdm(range(EEG.shape[0])):
    eeg_clip = train_data[idx,:,:,:]   # shape (19, 20, 100)
    # (19, 20, 100)
    # print("eeg_clip.shape before fft: ", eeg_clip.shape)
    
    eeg_clip = np.log(np.abs(fft(eeg_clip,axis=2)) + 1e-30)
    # print("eeg_clip.shape after fft: ", eeg_clip.shape)
    
    # (19, 20, 100)
    
    # should be (10, 19, 100)
    # normalize over frequency axis
    mean = np.mean(eeg_clip, axis=2, keepdims=True)
    std = np.std(eeg_clip, axis=2, keepdims=True)
    eeg_clip = normalize_data(eeg_clip, mean, std)
    label = Label[idx]
    
    dataset.append(Data(
        x=torch.tensor(eeg_clip, dtype=torch.float32),
        y=torch.tensor(label, dtype=torch.float32),
        edge_weight=edge_weight.squeeze(0),    
        edge_index=edge_index,
        elec_pos=torch.tensor(pos)
    ))

# Prepare the test dataset
# Calculate mean and std from test data
# Initially shape is (B, L=10, V=19, D=200) -> (B, V, L=2000, D=1) -> (B, V, L=20, D=100)
test_data = EEG_eval.transpose(0, 2, 1, 3).reshape(-1, 19, 20, 100)
# mean_test = np.mean(test_data, axis=2, keepdims=True)
# std_test = np.std(test_data, axis=2, keepdims=True)
# print("Mean_test.shape: ", mean_test.shape)
# print("Std_test.shape: ", std_test.shape)
test_dataset = []

for idx in tqdm(range(EEG_eval.shape[0])):
    eeg_clip = test_data[idx,:,:,:]
    # print("eeg_clip.shape before fft: ", eeg_clip.shape)
    eeg_clip = np.log(np.abs(fft(eeg_clip,axis=2)) + 1e-30)
    # print("eeg_clip.shape after fft: ", eeg_clip.shape)
    
    # should be (10, 19, 100)
    # normalize over frequency axis
    mean = np.mean(eeg_clip, axis=2, keepdims=True)
    std = np.std(eeg_clip, axis=2, keepdims=True)
    eeg_clip = normalize_data(eeg_clip, mean, std)

    label = Label_eval[idx]

    test_dataset.append(Data(
        x=torch.tensor(eeg_clip, dtype=torch.float32),
        y=torch.tensor(label, dtype=torch.float32),
        edge_weight=edge_weight.squeeze(0),    
        edge_index=edge_index,
        elec_pos=torch.tensor(pos)
    ))

# Split the dataset into train and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Save the datasets
SAVE_DIR = '/home/amli/TGMamba/data/tuhz/processed_dataset/'

torch.save(train_dataset, os.path.join(SAVE_DIR, 'train_dataset_fft.pt'))
torch.save(val_dataset, os.path.join(SAVE_DIR, 'val_dataset_fft.pt'))
torch.save(test_dataset, os.path.join(SAVE_DIR, 'test_dataset_fft.pt'))

print(f"Normalized datasets saved in {SAVE_DIR}")