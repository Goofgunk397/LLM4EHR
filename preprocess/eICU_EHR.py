import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import argparse
import torch
import random

parser = argparse.ArgumentParser(description='preprocess eICU')
parser.add_argument('--data_dir', type=str, default='data/eICU',
                    help='dir of raw eICU tables')

parser.add_argument('--ts_dir', type=str, default='processed data/eICU/eICU',
                    help='dir of processed eICU time series')

parser.add_argument('--save_dir', type=str, default='processed data/eICU/eICU',
                    help='save dir')

parser.add_argument('--step_size', type=float, default=60,
                    help='step size in minutes, should be fixed at 60')

parser.add_argument('--patch_size', type=float, default=5,
                    help='Duration of the event patches in hours')

parser.add_argument('--max_patch', type=int, default=40,
                    help='max number of patches per stay')

parser.add_argument('--use_native_codes', type=bool, default=True,
                    help='set True to use event codes as tokens')

parser.add_argument('--pad_token', type=str, default='[PAD]',
                    help='a special token used for padding in pre-trained LLM')

var_list = ['Invasive BP Diastolic', 'FiO2', 'GCS Total', 
            'glucose', 'Heart Rate', 'Invasive BP Systolic', 
            'MAP (mmHg)', 'Temperature (C)', 'pH', 'Respiratory Rate', 'O2 Saturation']

def process_single_ts(s_df, v_stats, step_size=1, max_steps=200):
    """
    Note that time series from the benchmark has already been resampled to 1-hr
    or 60 minutes intervals

    remember to correlate with discharge offset so no data after icu stay is included
    """
    s_df = s_df[s_df['itemoffset']>=0]
    s_dis = s_df['unitdischargeoffset'].max()//60
    s_df = s_df[s_df['itemoffset']<=s_dis]
    s_df = s_df[s_df['itemoffset']<max_steps]
    s_steps = s_df['itemoffset'].to_list()
    max_impute_steps = min(s_steps.max(), max_steps)
    step_idx = list(range(max_impute_steps))
    impute_array = np.zeros((max_impute_steps, len(var_list)))
    for i,v in enumerate(var_list):
        v_arr = np.interp(step_idx, s_steps, s_df[v].values)
        v_mean = v_stats[v]['mean']
        v_std = v_stats[v]['std']
        impute_array[:, i] = (v_arr - v_mean)/v_std

    array_out = np.zeros((max_steps, len(var_list)))
    array_out[step_idx] = impute_array

    return array_out


def process_single_stay(mdir, unit_id, stay_df, v_stats, step_size=1, max_steps=200, long_stay_duration=72):
    # Remember to change id datatype to str
    ts_df = pd.read_csv(os.path.join(mdir, f'{unit_id}/timeseries.csv'))
    stay_stats = stay_df[stay_df['patientunitstayid'] == unit_id]
    patient_outcome = 1 if stay_stats['unitdischargestatus'] == 1 else 0
    long_stay = 1 if (ts_df['unitdischargeoffset'].max() - long_stay_duration*60)>0 else 0

    out_array = process_single_ts(ts_df, v_stats, step_size, max_steps)

    discharge_offset_min = ts_df['unitdischargeoffset'].max()
    discharge_offset = discharge_offset_min//60  # discharge offset in hours
    max_offset = min(discharge_offset, max_steps)
    los_mask = np.zeros(max_steps)
    los_mask[:max_offset] = 1  # valid predictions
    los_mask[:24] = 0  # los prediction starts after 24 hrs
    los_labels = np.zeros(max_steps)  # remaining los is predicted hourly
    los_labels[:max_offset] = (discharge_offset_min - np.arange(0, max_offset)*60)/60  # a more precise conversion

    decomp_mask = np.zeros(max_steps)
    decomp_mask[:max_offset] = 1
    decomp_labels_s = np.zeros(discharge_offset)
    decomp_labels_s[-24:] = patient_outcome
    decomp_label = np.zeros(max_steps)
    decomp_label[:max_offset] = decomp_labels_s[:max_offset]

    label_obj = {'ihm_48': {'label':[patient_outcome]}, 
                 'los_48': {'label': [long_stay]}, 
                 'ihm_rolling':{'label':decomp_label, 'mask':decomp_mask}, 
                 'los_rolling':{'label':los_labels, 'mask':los_mask}}

    return out_array, label_obj

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    ts_dir = args.ts_dir
    save_dir = args.save_dir
    patch_size = args.patch_size
    max_patch = args.max_patch
    gone_native = args.use_native_codes
    step_size = args.step_size

    ts_list = next(os.walk(ts_dir))[1]  # get list of all eligible processed time series
    full_event_df = pd.read_csv(os.path.join(ts_dir, 'full_event_list.csv'))
    pheno_label_df = pd.read_csv(os.path.join(save_dir, 'pheno_label.csv'))
    full_event_df['offset'] = full_event_df['offset']//patch_size
    print(f'Begin processing patient event strings')

    event_string_df = full_event_df.groupby(['pid', 'offset'])['event_string'].apply(list)
    event_string_df = event_string_df.unstack('offset').fillna('PAD')
    event_nest = event_string_df.values.tolist()  # nested list
    new_idx = event_string_df.index.to_list()
    event_strings = dict(zip(new_idx, event_nest))

    eligible_samples = list(set(ts_list).union(set(new_idx)))
    print(f'Finished processing strings, {len(eligible_samples)} out of {len(ts_list)} samples are eligible for training')
    print('Begin process sequences to strings')
    string_objs = {}
    for k, v in event_strings.items():
        patch_mask = []
        v_strings = []
        for i in v[:max_patch]:
            if isinstance(i, list):
                patch_mask.append(1)
                v_strings.append(' '.join(i))
            elif isinstance(i, str):
                patch_mask.append(0)
                v_strings.append(args.pad_token)

        v_obj = {}
        v_obj['input_sequence'] = v_strings
        v_obj['patch_mask'] = patch_mask
        string_objs[k] = v_obj

    print('Finished processing strings')
    print('Begin processing time series')
    with open(os.path.join(data_dir, 'var_stats.json'), 'w') as f:
        v_stats = json.load(f)

    #def process_single_stay(mdir, unit_id, stay_df, v_stats, step_size=1, max_steps=200, long_stay_duration=72):
    stay_df = pd.read_csv(os.path.join(ts_dir, 'all_stays.csv'))
    stay_df['patientunitstayid'] = stay_df['patientunitstayid'].astype(str)
    for p in tqdm(eligible_samples, total=len(eligible_samples)):
        p_ts, p_labels = process_single_stay(ts_dir, p, stay_df, v_stats)
        p_labels['pheno'] = {'label':pheno_label_df.loc[p].to_list()[:-1]}
        p_ts_tensor = torch.tensor(p_ts).float()
        torch.save(os.path.join(save_dir, f'{p}/timeseries.pt'))
        with open(os.path.join(save_dir, f'{p}/ehr_seq.json'), 'w') as f:
            json.dump(string_objs[p], f)
        with open(os.path.join(save_dir, f'{p}/labels.json'), 'w') as f:
            json.dump(p_labels, f)

    train_size = int(0.7*len(eligible_samples))
    test_size = int(0.15*len(eligible_samples))
    random.shuffle(eligible_samples)

    train_samples = eligible_samples[:train_size]
    test_samples = eligible_samples[train_size:train_size+test_size]
    val_samples = eligible_samples[train_size+test_size:]
    partitions = {'train': train_samples, 'test': test_samples, 'val':val_samples}
    with open(os.path.join(save_dir, 'partition.json'),'w') as f:
        json.dump(partitions, f)