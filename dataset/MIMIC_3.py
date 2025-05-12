import os
import json
import torch
from torch.utils.data import Dataset
import itertools


class MIMIC3Dataset(Dataset):
    def __init__(self, data_dir, sample_list=None, max_steps=200, patch_size=5):
        super().__init__()
        assert max_steps % patch_size == 0
        self.ts_dir = os.path.join(data_dir, 'timeseries')
        self.seq_dir = os.path.join(data_dir, 'ehr_seq')
        if sample_list is None:
            self.ts_list = os.listdir(self.ts_dir)
            self.seq_list = ['_'.join(x.split('_')[:2]+['sequence.json']) for x in self.ts_list]
        else:
            self.ts_list = [x+'_timeseries.pt' for x in sample_list]
            self.seq_list = [x+'_sequence.json' for x in sample_list]

        self.max_steps = max_steps
        self.max_patch = max_steps//patch_size

    def __len__(self):
        return len(self.ts_list)

    def __getitem__(self, idx):
        ts_idx = torch.load(os.path.join(self.ts_dir, self.ts_list[idx]))
        ts_idx = ts_idx[:self.max_steps]
        with open(os.path.join(self.seq_dir, self.seq_list[idx]), 'r') as f:
            obj_idx = json.load(f)

        seq_idx = obj_idx['input_sequence'][:self.max_patch]
        mask_idx = torch.tensor(obj_idx['patch_mask'][:self.max_patch])

        out_item = {'ts': ts_idx, 'seq': seq_idx, 'p_mask': mask_idx}
        return out_item


class MIMIC3ClassificationDataset(Dataset):
    def __init__(self, data_dir, sample_list=None, max_steps=200, patch_size=5, task='ihm_48'):
        super().__init__()
        assert max_steps % patch_size == 0
        assert task in ['ihm_48', 'los_48', 'pheno']
        self.ts_dir = os.path.join(data_dir, 'timeseries')
        self.label_dir = os.path.join(data_dir, 'labels')
        if sample_list is None:
            self.ts_list = os.listdir(self.ts_dir)
            self.label_list = ['_'.join(x.split('_')[:2] + ['labels.json']) for x in self.ts_list]
        else:
            self.ts_list = [x + '_timeseries.pt' for x in sample_list]
            self.label_list = [x + '_labels.json' for x in sample_list]

        self.max_steps = max_steps
        self.max_patch = max_steps // patch_size
        self.task = task

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        ts_idx = torch.load(os.path.join(self.ts_dir, self.ts_list[idx]))
        ts_idx = ts_idx[:self.max_steps].float()
        with open(os.path.join(self.label_dir, self.label_list[idx]),'r') as f:
            label_obj = json.load(f)

        if self.task == 'ihm_48':
            out_label = torch.tensor(label_obj[self.task]['label']).float()
            ts_idx = ts_idx[:48]
        elif self.task == 'los_48':
            out_label = torch.tensor(label_obj[self.task]['label']).float()
            ts_idx = ts_idx[:48]
        elif self.task == 'pheno':
            out_label = torch.tensor(label_obj[self.task]['label']).float()
            ts_idx = ts_idx[:self.max_steps]

        return ts_idx, out_label
    
class MIMIC3DynamicDataset(Dataset):
    def __init__(self, data_dir, sample_list=None, max_steps=200, patch_size=5, task='ihm_rolling'):
        super().__init__()
        assert max_steps%patch_size == 0
        assert task in ['ihm_rolling', 'los_rolling']
        self.ts_dir = os.path.join(data_dir, 'timeseries')
        self.label_dir = os.path.join(data_dir, 'labels')
        if sample_list is None:
            self.ts_list = os.listdir(self.ts_dir)
            self.label_list = ['_'.join(x.split('_')[:2] + ['labels.json']) for x in self.ts_list]
        else:
            self.ts_list = [x + '_timeseries.pt' for x in sample_list]
            self.label_list = [x + '_labels.json' for x in sample_list]

        self.max_steps = max_steps
        self.max_patch = max_steps // patch_size
        self.task = task

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        ts_idx = torch.load(os.path.join(self.ts_dir, self.ts_list[idx]))
        ts_idx = ts_idx[:self.max_steps].float()
        with open(os.path.join(self.label_dir, self.label_list[idx]),'r') as f:
            label_obj = json.load(f)

        if self.task == 'ihm_rolling':
            load_label = torch.tensor(label_obj[self.task]['label']).float()
            load_mask = torch.tensor(label_obj[self.task]['mask']).float()
            out_label = torch.zeros(self.max_steps)
            out_label[:self.max_steps] = load_label[:self.max_steps]
            out_mask = torch.zeros(self.max_steps)
            out_mask[:self.max_steps] = load_mask[:self.max_steps]
            out_mask = out_mask.float()
        elif self.task == 'los_rolling':
            load_label = torch.tensor(label_obj[self.task]['label']).float()
            load_label = load_label/24.
            load_mask = torch.tensor(label_obj[self.task]['mask']).float()
            out_label = torch.zeros(self.max_steps)
            out_label[:self.max_steps] = load_label[:self.max_steps]
            out_mask = torch.zeros(self.max_steps)
            out_mask[:self.max_steps] = load_mask[:self.max_steps]
            out_mask = out_mask.float()


        return ts_idx, out_mask, out_label

class MIMIC3Collate(object):
    def __init__(self, tokeniser, max_steps=200, patch_size=5):
        self.tokeniser = tokeniser
        self.p_size = patch_size
        self.max_patch = max_steps//patch_size

    def __call__(self, batch):
        batch_ts = torch.stack([x['ts'] for x in batch], dim=0).float()
        batch_p_mask = torch.stack([x['p_mask'] for x in batch], dim=0).float()
        b_seq_list = []
        b_attn_list = []
        b_seq_mask_list = []
        for p in batch:
            p_out_seq, p_out_attn, p_out_mask = self.process_single(p)
            b_seq_list.append(p_out_seq)
            b_attn_list.append(p_out_attn)
            b_seq_mask_list.append(p_out_mask)

        pad_tok = 0
        max_seq_len = max([len(x) for x in b_seq_list])
        batch_seq_id, batch_attn, batch_seq_mask = self.batch_padding(b_seq_list, b_attn_list,
                                                                      b_seq_mask_list,
                                                                      max_seq_len, pad_tok=pad_tok)

        return batch_ts, batch_seq_id, batch_attn, batch_p_mask, batch_seq_mask

    def process_single(self, single_item):
        seq_tok = self.tokeniser(single_item['seq'], padding='do_not_pad')
        cls_tok = seq_tok['input_ids'][0][0]
        sep_tok = seq_tok['input_ids'][0][-1]
        valid_patch = list(itertools.compress((x[1:-1] for x in seq_tok['input_ids']),
                                              single_item['p_mask']))
        valid_attn = list(itertools.compress((x[1:-1] for x in seq_tok['attention_mask']),
                                             single_item['p_mask']))
        valid_len = sum([len(x) for x in valid_patch])+2
        p_tok_mask = torch.zeros((sum(single_item['p_mask']), valid_len))
        start_len = 1
        for i, p in enumerate(valid_patch):
            p_tok_mask[i, start_len:start_len+len(p)] = 1
            start_len += len(p)

        p_out_mask = torch.zeros((self.max_patch, valid_len))
        p_out_mask[[bool(x) for x in single_item['p_mask']]] = p_tok_mask
        p_out_seq = [cls_tok] + list(itertools.chain(*valid_patch)) + [sep_tok]
        p_out_attn = [1] + list(itertools.chain(*valid_attn)) + [1]

        return p_out_seq, p_out_attn, p_out_mask

    def batch_padding(self, batch_tok, batch_attn, batch_p_tok_mask, max_len, pad_tok=0):
        batch_pad_tok = [x+[pad_tok for _ in range(max_len-len(x))] for x in batch_tok]
        batch_pad_attn = [x+[0 for _ in range(max_len-len(x))] for x in batch_attn]
        batch_pad_p_tok_mask = [torch.hstack([x, torch.zeros(self.max_patch,
                                                             max_len-x.shape[1])]) for x in batch_p_tok_mask]
        batch_pad_p_tok_mask = torch.stack(batch_pad_p_tok_mask, dim=0).float()
        batch_pad_tok = torch.tensor(batch_pad_tok)
        batch_pad_attn = torch.tensor(batch_pad_attn)

        return batch_pad_tok, batch_pad_attn, batch_pad_p_tok_mask


