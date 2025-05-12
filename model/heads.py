from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle
from torchvision.models import resnet


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, label_size, max_length=48, dense=False, pool='max'):
        super().__init__()
        if dense:
            self.proj = nn.Linear(hidden_size, hidden_size)
            assert pool in ['max', 'mean']
            self.pool = pool
        else:
            self.proj = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, label_size)
        self.dense=dense

    def forward(self, ts_hidden):
        if self.dense:
            if self.pool == 'max':
                ts_hidden,_ = ts_hidden.max(dim=1)
            elif self.pool == 'mean':
                ts_hidden = ts_hidden.mean(dim=1)
            ts_hidden = F.relu(self.proj(ts_hidden))
            ts_out = F.sigmoid(self.classifier(ts_hidden))
        else:
            ts_hidden = ts_hidden[:, -1, :]
            ts_hidden = F.relu(self.proj(ts_hidden))
            ts_out = F.sigmoid(self.classifier(ts_hidden))

        return ts_out
    
class TSAvgPooler(nn.Module):
    def __init__(self, in_size=768, max_steps=200, patch_size=5):
        super().__init__()
        self.split_size = max_steps//patch_size

    def forward(self, x_hidden):
        x_split = torch.split(x_hidden, self.split_size, dim=1)
        x_hidden = torch.stack(x_split, dim=1)
        x_hidden = x_hidden.mean(dim=1)
        return x_hidden

class DynamicClassificationHead(nn.Module):
    def __init__(self, hidden_size, label_size, encoder, max_length=100, task='ihm_rolling'):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, label_size)
        self.encoder = encoder
        self.pooler = TSAvgPooler(hidden_size, max_length, patch_size=5)
        self.task = task

    def forward(self, ts):
        ts_hidden = self.pooler(self.encoder.encode_ts(ts))
        ts_hidden = F.relu(self.proj(ts_hidden))
        if self.task == 'ihm_rolling':
            ts_out = F.sigmoid(self.classifier(ts_hidden))
        elif self.task == 'los_rolling':
            ts_out = self.classifier(ts_hidden)

        return ts_out
    
    

    

