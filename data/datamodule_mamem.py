# mamem_dataset.py

import os
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from lightning import LightningDataModule

class MAMEMDataset(Dataset):
    def __init__(self, root, subject, split='train', transform=None, pre_transform=None):
        self.root = root
        self.subject = subject
        self.split = split
        super().__init__(root, transform, pre_transform)
        
        # Load preprocessed data
        preprocessed_path = os.path.join(self.root, 'preprocessed', f'processed_U{self.subject:03d}.pt')
        self.data = torch.load(preprocessed_path)[self.split]

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def get_class_balance(self):
        y = torch.tensor([data.y for data in self.data])
        class_counts = torch.bincount(y)
        class_balance = class_counts.float() / len(y)
        return {i: balance.item() for i, balance in enumerate(class_balance)}

class MAMEMDataModule(LightningDataModule):
    def __init__(self, data_dir: str = '/home/amli/MAtt/data/MAMEM', subject: int = 1, batch_size: int = 64, num_workers: int = 12):
        super().__init__()
        self.data_dir = data_dir
        self.subject = subject
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = MAMEMDataset(self.data_dir, self.subject, split='train')
        self.val_dataset = MAMEMDataset(self.data_dir, self.subject, split='val')
        self.test_dataset = MAMEMDataset(self.data_dir, self.subject, split='test')

        print("Class balance in training set:")
        print(self.train_dataset.get_class_balance())
        print("Class balance in validation set:")
        print(self.val_dataset.get_class_balance())
        print("Class balance in test set:")
        print(self.test_dataset.get_class_balance())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def get_num_nodes(self):
        return self.train_dataset[0].num_nodes

    def get_num_edges(self):
        return self.train_dataset[0].num_edges

    def get_input_dim(self):
        return self.train_dataset[0].num_features

    def get_class_balance(self):
        return {
            'train': self.train_dataset.get_class_balance(),
            'val': self.val_dataset.get_class_balance(),
            'test': self.test_dataset.get_class_balance()
        }

# Usage example:
# data_module = MAMEMDataModule(data_dir='/home/amli/MAtt/data/MAMEM', subject=1, batch_size=64)
# data_module.setup()
# class_balance = data_module.get_class_balance()
# print(class_balance)