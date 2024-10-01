import os
import torch
from torch_geometric.loader import DataLoader
from lightning import LightningDataModule

class TUHZDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, dataset_has_fft: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_has_fft = dataset_has_fft

    def setup(self, stage=None):
        dataset_suffix = 'fft' if self.dataset_has_fft else 'raw_s4mer_normalized'
        self.train_dataset = torch.load(os.path.join(self.data_dir, f'train_dataset_{dataset_suffix}.pt'))
        self.val_dataset = torch.load(os.path.join(self.data_dir, f'val_dataset_{dataset_suffix}.pt')) 
        self.test_dataset = torch.load(os.path.join(self.data_dir, f'test_dataset_{dataset_suffix}.pt'))

    def prepare_data(self):
        # This method is called only on 1 GPU in distributed setting
        # Download data if needed. This method is called only from a single process
        # Nothing to do for TUHZ dataset as it's assumed to be already downloaded
        pass
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_num_vertices(self):
        return self.train_dataset[0].num_nodes
    
    def get_input_dim(self):
        return self.train_dataset[0].x.shape[-1]

    def get_num_edges(self):
        return self.train_dataset[0].edge_index.shape[1]