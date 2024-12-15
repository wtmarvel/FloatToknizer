import torch
from torch.utils.data import Dataset

class FloatDataset(Dataset):
    def __init__(self, training=True, **kwargs):
        super().__init__()
        self.training = training
        # Only store size, generate data on-the-fly
        self.size = 10_000_000 if training else 2048
        self.min_val = -1
        self.max_val = 1
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random float number on-the-fly using torch
        value = torch.empty(1).uniform_(self.min_val, self.max_val)
        return {
            'value': value
        }
