import numpy as np
import torch
import h5py
import os
from tqdm import tqdm
import pandas as pd
from epoch_features_processing import epoch_features
from signal_processing import signal_processings
from torch_geometric.data import Data, Dataset
from torch_geometric.data.collate import collate


DODH_RAW_DATA_DIR='/home/amli/dreem-learning-open/data/h5/dodh'
DODH_FILEMARKER_DIR = "/home/amli/TGMamba/data/file_markers_dodh"
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

class PreprocessedDODHDataset(Dataset):
    def __init__(self, root, file_marker, raw_data_path, transform=None, pre_transform=None):
        self.file_marker = file_marker
        self.raw_data_path = raw_data_path
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        
        for idx in tqdm(range(len(self.file_marker))):
            h5_file_name = self.file_marker["record_id"].iloc[idx]
            y = self.file_marker["label"].iloc[idx]
            clip_idx = int(self.file_marker["clip_index"].iloc[idx])

            # Read and preprocess data
            signals, channels, _ = read_dreem_data(self.raw_data_path, h5_file_name)
            channel_idxs = reorder_channels(channels, "dodh")
            
            physical_len = int(DREEM_FREQ * 30)  # 30 seconds
            start_idx = clip_idx * physical_len
            end_idx = start_idx + physical_len

            x = signals[channel_idxs, start_idx:end_idx]
            signal_properties = {'fs': DREEM_FREQ, 'padding': 0}  
            x = preprocess_signal(x, signal_properties)
            
            x = torch.FloatTensor(x) # .unsqueeze(-1)
            y = torch.LongTensor([y])
            data = Data(x=x, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.file_marker)

    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        return data[idx]


for split in ['train', 'val', 'test']:
    file_marker = pd.read_csv(os.path.join(DODH_FILEMARKER_DIR, f"{split}_file_markers.csv"))
    dataset = PreprocessedDODHDataset(
        root=f'./data/preprocessed_dodh_{split}',
        file_marker=file_marker,
        raw_data_path=DODH_RAW_DATA_DIR
    )
    # This will trigger the processing and saving
    _ = dataset[0]