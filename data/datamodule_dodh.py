import sys
import git
import os
import lightning as L
import pickle
import numpy as np
import h5py
import pandas as pd
import torch
import torch_geometric

from collections import Counter
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data, Dataset
from typing import Optional, Callable
from tqdm import tqdm

from data.epoch_features_processing import epoch_features
from data.signal_processing import signal_processings

# from constants import DODH_CHANNELS

DODH_FILEMARKER_DIR = "/home/amli/TGMamba/data/file_markers_dodh"
# Dreem DOD-H
DODH_CHANNELS = [
    "C3_M2",
    "F3_F4",
    "F3_M2",
    "F3_O1",
    "F4_M1",
    "F4_O2",
    "FP1_F3",
    "FP1_M2",
    "FP1_O1",
    "FP2_F4",
    "FP2_M1",
    "FP2_O2",
    "ECG",
    "EMG",
    "EOG1",
    "EOG2",
]
DREEM_FREQ = 250
DREEM_LABEL_DICT = \
{
    0:"WAKE", 
    1: "N1", 
    2: "N2", 
    3: "N3", 
    4: "REM"
}

def reorder_channels(channels, dataset_name):

    if dataset_name == "dodh":
        channel_idxs = np.array([channels.index(ch) for ch in DODH_CHANNELS])
    else:
        raise NotImplementedError

    return channel_idxs

def read_dreem_data(data_dir, file_name):
    with h5py.File(os.path.join(data_dir, file_name), "r") as f:
        labels = f["hypnogram"][()]

        signals = []
        channels = []
        for key in f["signals"].keys():
            for ch in f["signals"][key].keys():
                signals.append(f["signals"][key][ch][()])
                channels.append(ch)
    signals = np.stack(signals, axis=0)

    return signals, channels, labels

def preprocess_signal(signals, signal_properties):
    """
    Preprocess the EEG signals based on the procedure in 
    "Dreem Open Datasets: Multi-Scored Sleep Datasets to 
    compare Human and Automated sleep staging" paper.
    
    Parameters:
    signals (np.array): Shape (num_channels, time_steps)
    signal_properties (dict): Contains 'fs' (sampling frequency) and other properties
    
    Returns:
    np.array: Shape (num_channels, num_frequencies, num_time_steps)
    """
    signals = signals.T
    # print("Preprocessing raw signal.\nSignal shape before preprocessing:", signals.shape)
    # 1. Band-pass filter
    filtered_signal, signal_props = signal_processings['filter'](signals, signal_properties)
    # print("Filtered signal shape:", filtered_signal.shape)
    # 2. Resample to 100 Hz 
    # this also sets signal_props['fs'] = 100
    resampled_signal, signal_props = signal_processings['resample'](filtered_signal, signal_props, target_frequency=100)
    # print("Resampled signal shape:", resampled_signal.shape)
    
    # 3. Clip and divide by 500 to remove extreme values
    clipped_signal = np.clip(resampled_signal, -500, 500) / 500
    # print("Clipped signal shape:", clipped_signal.shape)
    # 4. Zero-padding
    # padded_signal, signal_props = signal_processings['padding'](clipped_signal, signal_props, padding_duration=30)
    # print("Padded signal shape:", padded_signal.shape)
    
    # 5. Compute STFT
    freq_resolution = signal_properties['fs'] / 256
    stft_result = epoch_features['spectral_power'](clipped_signal, signal_props,
                                                   frequency_interval=[(i * freq_resolution, 
                                                                        (i + 1) * freq_resolution) 
                                                                       for i in range(129)],  # 129 bins as specified 
                                                   stft_duration=2.56, 
                                                   stft_overlap=1.56, 
                                                   epoch_duration=30
                                                   )
    # print("STFT shape:", stft_result.shape)
    # 6. Log-power and clipping
    log_power = np.log10(np.abs(stft_result)**2 + 1e-10)
    clipped_power = np.clip(log_power, -20, 20)
    
    # 7. Normalize the spectrogram to have zero mean and unit variance 
    # signal-wise independently of the timestep
    normalized_spectrogram = (clipped_power - np.mean(clipped_power, axis=2, keepdims=True)) / (np.std(clipped_power, axis=2, keepdims=True) + 1e-10)
    
    return normalized_spectrogram


class StandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, batched=True):
        """
        Masked inverse transform
        Args:
            data: data for inverse scaling
            is_tensor: whether data is a tensor
            device: device
            mask: shape (batch_size,) nodes where some signals are masked
        """
        device = data.device

        mean = self.mean.copy()
        std = self.std.copy()

        if batched:
            mean = np.expand_dims(mean, 0)
            std = np.expand_dims(std, 0)

        if torch.is_tensor(data):
            mean = torch.FloatTensor(mean).to(device).squeeze(-1)
            std = torch.FloatTensor(std).to(device).squeeze(-1)

        return data * std + mean


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Adapted from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
        replacement=True,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        self.replacement = replacement

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(
                self.weights, self.num_samples, replacement=self.replacement
            )
        )

    def __len__(self):
        return self.num_samples


class PreprocessedDODHDataset(Dataset):
    def __init__(self, file_path, transform=None, pre_transform=None):
        super().__init__(transform, pre_transform)
        self.file_path = file_path
        self.data = torch.load(file_path)
        self.num_samples = self.data['num_samples']

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data['data'][idx]

    def get_labels(self):
        return torch.FloatTensor([data.y.item() for data in self.data['data']])

class DODHDataModule(L.LightningDataModule):
    def __init__(
        self,
        preprocessed_data_dir,
        train_batch_size,
        test_batch_size,
        num_workers,
        pin_memory=True,
        balanced_sampling=True,
        use_class_weight=False,
    ):
        super().__init__()
        self.preprocessed_data_dir = preprocessed_data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.balanced_sampling = balanced_sampling
        self.use_class_weight = use_class_weight
        self.pin_memory = pin_memory
        if use_class_weight and balanced_sampling:
            raise ValueError(
                "Choose only one of use_class_weight or balanced_sampling!"
            )
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None

    def setup(self, stage=None):
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(self.preprocessed_data_dir, f'dodh_{split}.pt')
            dataset = PreprocessedDODHDataset(file_path=file_path)
            setattr(self, f"{split}_dataset", dataset)
            
        if self.use_class_weight:
            self.class_weights = torch.FloatTensor(
                [
                    np.sum(self.train_dataset.labels == c) / len(self.train_dataset)
                    for c in np.arange(5)
                ]
            )
            self.class_weights /= torch.sum(self.class_weights)
            print("Class weight:", self.class_weights)
            
    def prepare_data(self) -> None:
        self.file_markers = {}
        for split in ["train", "val", "test"]:
            print("{}_file_markers.csv".format(split))
            self.file_markers[split] = pd.read_csv(
                os.path.join(
                    DODH_FILEMARKER_DIR, "{}_file_markers.csv".format(split)
                )
            )   

    def train_dataloader(self):
        if self.balanced_sampling:
            class_counts = list(
                Counter(self.train_dataset.get_labels().cpu().numpy()).values()
            )
            min_samples = np.min(np.array(class_counts))
            sampler = ImbalancedDatasetSampler(
                dataset=self.train_dataset,
                num_samples=min_samples * 5,
                replacement=False,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            shuffle=shuffle,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return val_dataloader

    def test_dataloader(self):

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return test_dataloader
