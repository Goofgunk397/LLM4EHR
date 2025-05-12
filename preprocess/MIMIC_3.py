import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
import torch
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description='preprocess mimic3')
parser.add_argument('--data_dir', type=str, default='data/MIMIC_3',
                    help='data dir')
parser.add_argument('--save_dir', type=str, default='processed_data/MIMIC_3',
                    help='save dir')

parser.add_argument('--max_steps', type=int, default=200,
                    help='max number of patches per stay')

pd.options.mode.chained_assignment = None
v_list = ['Capillary refill rate',
          'Diastolic blood pressure',
          'Fraction inspired oxygen',
          'Glascow coma scale total',
          'Glucose',
          'Heart Rate',
          'Systolic blood pressure',
          'Mean blood pressure',
          'Temperature', 'Weight', 'pH','Respiratory rate','pH','Oxygen saturation','Height']


def process_listfile(pth, long_stay_duration=72.):
    train_lf = pd.read_csv(os.path.join(pth, 'train_listfile.csv'))
    val_lf = pd.read_csv(os.path.join(pth, 'val_listfile.csv'))
    test_lf = pd.read_csv(os.path.join(pth, 'test_listfile.csv'))
    full_lf = pd.concat([train_lf, val_lf, test_lf])
    full_lf = full_lf.rename(columns={'in-hospital mortality task (pos;mask;label)': 'ihm_48',
                                      'length of stay task (masks;labels)': 'los_rolling',
                                      'length of stay': 'los_48',
                                      'phenotyping task (labels)': 'pheno',
                                      'decompensation task (masks;labels)': 'ihm_rolling'})

    full_lf['los_48'] = (full_lf['los_48'].astype(float) > long_stay_duration).astype(int)
    full_lf['ihm_48'] = full_lf['ihm_48'].str.split(';').str[-1].astype(int)
    full_lf['los_rolling'] = full_lf['los_rolling'].str.split(';')
    full_lf['los_rolling'] = full_lf['los_rolling'].apply(lambda x: (list(map(int,x[:len(x)//2])),
                                                                     list(map(float,x[len(x)//2:]))))
    full_lf['ihm_rolling'] = full_lf['ihm_rolling'].str.split(';')
    full_lf['ihm_rolling'] = full_lf['ihm_rolling'].apply(lambda x: (list(map(int,x[:len(x)//2])),
                                                                     list(map(int,x[len(x)//2:]))))
    full_lf['pheno'] = full_lf['pheno'].str.split(';')
    full_lf['pheno'] = full_lf['pheno'].apply(lambda x: list(map(int, x)))
    full_lf = full_lf.set_index('filename')

    return full_lf


def get_partition_list(pth,partition='train'):

    if partition == 'train':
        t_name = 'train/ehr_seq'
    elif partition == 'test':
        t_name = 'test/ehr_seq'
    f_list = os.listdir(os.path.join(pth, t_name))
    f_list_n = ['_'.join(x.split('_')[:2] + ['timeseries.csv']) for x in f_list]
    return f_list_n


def parse_meta_data(pth):
    with open(os.path.join(pth,'resource/variable_stats.json'), 'r') as f:
        v_stats_json = json.load(f)

    v_stats = {}
    for i in v_list:
        v_stats[i] = v_stats_json[i]

    return v_stats


def parse_time(t, step_size=60):
    """
    step_size in minutes
    """
    def get_time_bin(x, step_size):
        x_m = x*60
        return int(x_m / step_size - 1e-6)

    new_t = t.apply(lambda x: get_time_bin(x, step_size))
    return new_t


def parse_labels(s_labels, step_idx, final_step_size, max_time):
    s_ihm_48 = s_labels['ihm_48']
    s_los_48 = s_labels['los_48']

    def resample_labels(sl, name):
        sl_ml, sl_ll = sl[name]
        sl_ml, sl_ll = np.array(sl_ml), np.array(sl_ll)
        sl_mo, sl_lo = np.zeros(max_time//final_step_size), np.zeros(max_time//final_step_size)
        sl_mo[step_idx] = sl_ml[step_idx]
        sl_lo[step_idx] = sl_ll[step_idx]
        return sl_mo.tolist(), sl_lo.tolist()

    s_decomp_mask, s_decomp_label = resample_labels(s_labels, 'ihm_rolling')
    s_los_mask, s_los_label = resample_labels(s_labels, 'los_rolling')

    s_pheno = s_labels['pheno']
    label_obj = {'ihm_48': {'label': int(s_ihm_48)},
                 'los_48': {'label': int(s_los_48)},
                 'pheno': {'label': list(map(int,s_pheno))},
                 'los_rolling': {'mask': list(map(int,s_los_mask)),
                                 'label': list(map(float,s_los_label))},
                 'ihm_rolling': {'mask': list(map(int,s_decomp_mask)),
                                 'label': list(map(float,s_decomp_label))}}

    return label_obj


def parse_and_impute(s_df, s_l, v_stats, impute_step_size=1, final_step_size=60, max_time=60*192):
    s_df['Steps_impute'] = parse_time(s_df['Hours'], step_size=impute_step_size)
    s_df['Steps_hr'] = parse_time(s_df['Hours'], step_size=final_step_size)
    s_df = s_df[s_df['Steps_impute'] < max_time]
    idx_a_1 = np.arange(0, max_time)
    idx_a_2 = np.arange(0, max_time, final_step_size)
    out_labels = parse_labels(s_l, s_df['Steps_hr'].tolist(), final_step_size, max_time)

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    v_out_raw_list = []
    for v in v_stats.keys():
        v_item = v_stats[v]
        if all(s_df[v].isna()):
            s_df[v] = v_item['impute_value']

        assert not all(s_df[v].isna())
        v_df = s_df[['Steps_impute', v]].dropna(subset=v)
        v_df = v_df.drop_duplicates(subset='Steps_impute', keep='last')
        v_array = np.array(v_df[v].to_list())
        v_t = np.array(v_df['Steps_impute'].to_list())
        v_impute = np.interp(idx_a_1, v_t, v_array)
        v_avg = moving_average(v_impute, n=final_step_size)
        v_avg_re = v_avg[idx_a_2]
        v_raw = (v_avg_re-v_item['mean'])/v_item['std']
        v_out_raw_list.append(np.expand_dims(v_raw, -1))

    v_out_raw = np.hstack(v_out_raw_list)
    return v_out_raw, out_labels


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    max_steps = args.max_steps
    train_dir = os.path.join(data_dir,'train')
    test_dir = os.path.join(data_dir, 'test')
    train_save_dir = os.path.join(save_dir,'train/timeseries')
    test_save_dir = os.path.join(save_dir,'test/timeseries')
    train_label_dir = os.path.join(save_dir,'train/labels')
    test_label_dir = os.path.join(save_dir,'test/labels')

    train_flist = get_partition_list(save_dir, partition='train')
    test_flist = get_partition_list(save_dir, partition='test')
    all_labels = process_listfile(data_dir)
    v_stats = parse_meta_data(data_dir)

    print('Begin building training timeseries')
    for i in tqdm(train_flist, total=len(train_flist)):
        i_df = pd.read_csv(os.path.join(train_dir, i))
        i_labels = all_labels.loc[i]
        if not i_df.loc[i_df['Hours'] < max_steps, v_list].empty:
            i_out, i_out_labels = parse_and_impute(i_df,
                                                   i_labels,
                                                   v_stats,
                                                   impute_step_size=1,
                                                   final_step_size=60, max_time=60*max_steps)
            i_out_tensor = torch.tensor(i_out)
            i_name = '_'.join(i.split('_')[:2]+['timeseries.pt'])
            i_l_name = '_'.join(i.split('_')[:2]+['labels.json'])
            torch.save(i_out_tensor, os.path.join(train_save_dir, i_name))
            with open(os.path.join(train_label_dir, i_l_name), 'w') as f:
                json.dump(i_out_labels, f)
        else:
            print(f'{i} has no usable timeseries data, ignored')

    print('Begin building testing timeseries')
    for i in tqdm(test_flist, total=len(test_flist)):
        i_df = pd.read_csv(os.path.join(test_dir, i))
        i_labels = all_labels.loc[i]
        if not i_df.loc[i_df['Hours']<max_steps, v_list].empty:

            i_out,i_out_labels = parse_and_impute(i_df,
                                                  i_labels,
                                                  v_stats,
                                                  impute_step_size=1,
                                                  final_step_size=60, max_time=60 * max_steps)

            i_out_tensor = torch.tensor(i_out)
            i_name = '_'.join(i.split('_')[:2] + ['timeseries.pt'])
            i_l_name = '_'.join(i.split('_')[:2] + ['labels.json'])
            torch.save(i_out_tensor, os.path.join(test_save_dir, i_name))
            with open(os.path.join(test_label_dir, i_l_name), 'w') as f:
                json.dump(i_out_labels, f)
        else:
            print(f'{i} has no usable timeseries data, ignored')








