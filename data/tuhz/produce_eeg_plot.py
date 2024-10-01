import torch
import numpy as np
from tqdm import tqdm
import h5py
import scipy.io as sio
from torch_geometric.data import Data, Dataset
from scipy.fft import fft
import os
import matplotlib.pyplot as plt

def plot_eeg(eeg_data, save_path=None):
    """
    Plot EEG data for all channels with random colors, thicker lines, and smaller subplots.
    
    :param eeg_data: numpy array of shape (19, 2000)
    :param save_path: optional path to save the figure
    """
    num_channels, num_samples = eeg_data.shape
    time = np.arange(num_samples) / 200  # Assuming 200 Hz sampling rate

    # Reduce the height of the figure
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 15), sharex=True)
    fig.suptitle('EEG Data for All Channels', fontsize=16)

    # Generate random colors for the channels
    colors = [np.random.rand(3,) for _ in range(num_channels)]

    for i, ax in enumerate(axes):
        ax.plot(time, eeg_data[i, :], linewidth=2.5, color=colors[i])
        ax.set_ylabel(f'Ch {i+1}', rotation=0, labelpad=20, va='center')
        ax.set_ylim(eeg_data.min(), eeg_data.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

# Load EEG data and labels
with h5py.File('/home/data_shared/clip_data.h5', "r") as f:
    EEG = np.array(f[list(f.keys())[0]])

idx=0
eeg_clip = EEG[idx,:,:,:]
eeg_clip = eeg_clip.transpose(1, 0, 2).reshape(19, -1)

plot_eeg(eeg_clip, save_path=f'/home/amli/TGMamba/data/tuhz/eeg_plots_{idx}.png')