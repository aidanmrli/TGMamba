import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader, ImbalancedSampler
from lightning import LightningDataModule

SUBJECT_LIST = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]

class BCIchaDataset(InMemoryDataset):
    def __init__(self, data_list=None, root=None, transform=None, pre_transform=None):
        super(BCIchaDataset, self).__init__(root, transform, pre_transform)
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        pass
    
    def save(self, path):
        torch.save((self.data, self.slices), path)
        
    def load(self, path):
        self.data, self.slices = torch.load(path)
        
class BCIchaDataModule(LightningDataModule):
    def __init__(self, data_dir: str, subject: int, dataset_has_fft: bool, batch_size: int = 32, num_workers: int = 4, load_all_subjects=False):
        super().__init__()
        self.data_dir = data_dir
        self.subject = subject
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_has_fft = dataset_has_fft
        self.load_all_subjects = load_all_subjects
        if self.dataset_has_fft:
            self.data_dir = os.path.join(self.data_dir, 'fft')

    def prepare_data(self):
        # This method is called only on 1 GPU in distributed setting
        # You can download or create the dataset here if it doesn't exist
        # In this case, we assume the data has already been preprocessed
        pass

    def setup(self, stage=None):
        # Load the preprocessed datasets
        print("Loading from self.data_dir", self.data_dir)
        if self.load_all_subjects:
            self.train_dataset = torch.load(os.path.join(self.data_dir, f'train_dataset_all_subjects.pt'))
            self.val_dataset = torch.load(os.path.join(self.data_dir, f'val_dataset_all_subjects.pt'))
            self.test_dataset = torch.load(os.path.join(self.data_dir, f'test_dataset_all_subjects.pt'))
        else:
            self.train_dataset = torch.load(os.path.join(self.data_dir, f'train_dataset_S{self.subject:02d}.pt'))
            self.val_dataset = torch.load(os.path.join(self.data_dir, f'val_dataset_S{self.subject:02d}.pt'))
            self.test_dataset = torch.load(os.path.join(self.data_dir, f'test_dataset_S{self.subject:02d}.pt'))
            
        self.sampler = ImbalancedSampler(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_num_nodes(self):
        # Assuming all samples have the same number of nodes
        return self.train_dataset[0].num_nodes

    def get_num_edges(self):
        return self.train_dataset[0].edge_index.shape[1]

    def get_input_dim(self):
        return self.train_dataset[0].x.shape[-1]