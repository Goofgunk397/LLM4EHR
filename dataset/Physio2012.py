import os
import json
import torch
from torch.utils.data import Dataset

class Physio2012Dataset(Dataset):
    def __init__(self, data_pth):
        super().__init__()
        self.full_set = torch.load(data_pth)

    def __len__(self):
        return len(self.full_set)
    
    def __getitem__(self, idx):
        X = self.full_set[idx]['X'].float()
        y = self.full_set[idx]['y'].long()

        return X, y
