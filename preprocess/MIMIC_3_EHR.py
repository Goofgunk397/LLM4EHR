import os
import sys
import numpy as np
import pandas as pd
import argparse
import json

import setuptools.glob
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='preprocess mimic3')
parser.add_argument('--data_dir', type=str, default='data/MIMIC_3',
                    help='data dir')
parser.add_argument('--save_dir', type=str, default='processed_data/MIMIC_3',
                    help='save dir')

parser.add_argument('--patch_size', type=float, default=5,
                    help='Duration of the event patches in hours')

parser.add_argument('--max_patch', type=int, default=40,
                    help='max number of patches per stay')

parser.add_argument('--use_native_codes', type=bool, default=True,
                    help='set True to use event codes as tokens')

parser.add_argument('--pad_token', type=str, default='[PAD]',
                    help='a special token used for padding in pre-trained LLM')

pd.options.mode.chained_assignment = None
def get_ts_episodes(mdir, partition='train'):
    out_list = ['_'.join(x.split('_')[:2]) for x in os.listdir(os.path.join(mdir,partition))]
    return out_list


def process_stay_df(mdir):
    stay_df = pd.read_csv(os.path.join(mdir, 'all_stays.csv'))
    stay_df_s = stay_df[['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME']]
    stay_df_s['HADM_ID'] = stay_df_s['HADM_ID'].astype(str)
    stay_df_s['INTIME'] = pd.to_datetime(stay_df_s['INTIME'])
    stay_df_s['OUTTIME'] = pd.to_datetime(stay_df_s['OUTTIME'])
    stay_df_s = stay_df_s.sort_values(by='SUBJECT_ID').reset_index(drop=True)
    id_count = stay_df_s['SUBJECT_ID'].value_counts()
    id_unique = stay_df_s['SUBJECT_ID'].drop_duplicates(keep='first').values.tolist()
    id_count = id_count.loc[id_unique].apply(lambda x: [i+1 for i in range(x)])
    id_count = id_count.reset_index().explode('count').set_index(stay_df_s.index)
    stay_df_s['CUSTOM_ID'] = stay_df_s['SUBJECT_ID'].astype(str) + '_episode' + id_count['count'].astype(str)
    assert not any(stay_df_s['CUSTOM_ID'].isna())

    return stay_df_s


def load_token_dfs(mdir):
    lab_tokens = pd.read_csv(os.path.join(mdir, 'lab_tokens.csv'))
    lab_tokens['TOKEN_CODE'] = lab_tokens['TOKEN_CODE'].astype(str) + '_lab'
    icd_tokens = pd.read_csv(os.path.join(mdir,'ICD_diag_tokens.csv'))
    icd_tokens['TOKEN_CODE'] = icd_tokens['TOKEN_CODE'].astype(str) + '_diag'
    proc_tokens = pd.read_csv(os.path.join(mdir, 'MV_proc_tokens.csv'))
    proc_tokens['TOKEN_CODE'] = proc_tokens['TOKEN_CODE'].astype(str) + '_proc'
    cv_input_tokens = pd.read_csv(os.path.join(mdir, 'CV_input_tokens.csv'))
    cv_input_tokens['TOKEN_CODE'] = cv_input_tokens['TOKEN_CODE'].astype(str) + '_cv'
    mv_input_tokens = pd.read_csv(os.path.join(mdir, 'MV_input_tokens.csv'))
    mv_input_tokens['TOKEN_CODE'] = mv_input_tokens['TOKEN_CODE'].astype(str) + '_mv'
    proc_tokens['TOKEN_TYPE'] = 'procedure'
    lab_tokens['TOKEN_TYPE'] = 'lab'
    icd_tokens['TOKEN_TYPE'] = 'diagnoses'
    cv_input_tokens['TOKEN_TYPE'] = 'cv_input'
    mv_input_tokens['TOKEN_TYPE'] = 'mv_input'

    return pd.concat([icd_tokens[['TOKEN_CODE', 'TOKEN_STRING', 'TOKEN_TYPE']],
                      lab_tokens[['TOKEN_CODE', 'TOKEN_STRING', 'TOKEN_TYPE']],
                      proc_tokens[['TOKEN_CODE', 'TOKEN_STRING', 'TOKEN_TYPE']],
                      cv_input_tokens[['TOKEN_CODE', 'TOKEN_STRING', 'TOKEN_TYPE']],
                      mv_input_tokens[['TOKEN_CODE', 'TOKEN_STRING', 'TOKEN_TYPE']]])


def process_cv_events(mdir, code_df, stay_df, step_size=5., use_code=False):
    cv_event_df = pd.read_csv(os.path.join(mdir, 'cv_input_events.csv'))
    cv_event_df['TIMESTAMP'] = pd.to_datetime(cv_event_df['TIMESTAMP'])
    cv_event_df = cv_event_df.dropna(subset=['HADM_ID'])
    cv_event_df['HADM_ID'] = cv_event_df['HADM_ID'].astype(int).astype(str)
    cv_event_df['QUANTILE_VALUE'] = cv_event_df['QUANTILE_VALUE'].astype(int).astype(str)
    cv_event_df['QUANTILE_VALUE'] = 'Q_' + cv_event_df['QUANTILE_VALUE']
    cv_event_df = cv_event_df.merge(stay_df[['HADM_ID', 'INTIME', 'OUTTIME', 'CUSTOM_ID']],
                                      left_on='HADM_ID', right_on='HADM_ID', how='inner')
    cv_event_df = cv_event_df[cv_event_df['TIMESTAMP'] <= cv_event_df['OUTTIME']]
    cv_event_df['TOKEN_CODE'] = cv_event_df['TOKEN_CODE'].astype(str) + '_cv'
    if use_code:
        cv_event_df['OUT_STRING'] = cv_event_df['TOKEN_CODE']
        cv_event_df['OUT_STRING'] = cv_event_df['OUT_STRING'] + ' ' + cv_event_df['QUANTILE_VALUE']
    else:
        cv_event_df = cv_event_df.merge(code_df[['TOKEN_CODE', 'TOKEN_STRING']],
                                          left_on='TOKEN_CODE', right_on='TOKEN_CODE')
        cv_event_df = cv_event_df.rename(columns={'TOKEN_STRING': 'OUT_STRING'})
        cv_event_df['OUT_STRING'] = cv_event_df['OUT_STRING'] + ' ' + cv_event_df['QUANTILE_VALUE']
    time_steps = (cv_event_df['TIMESTAMP'] - cv_event_df['INTIME']).dt.total_seconds() // (step_size * 3600 + 1e-4)
    cv_event_df = cv_event_df.assign(TIMESTEP=time_steps)
    cv_event_df = cv_event_df[cv_event_df['TIMESTEP'] >= 0]
    cv_event_df = cv_event_df.drop_duplicates(subset=['CUSTOM_ID', 'TOKEN_CODE'], keep='last')
    return cv_event_df[['CUSTOM_ID', 'OUT_STRING', 'TIMESTEP', 'TIMESTAMP', 'TOKEN_CODE']]


def process_mv_events(mdir, code_df, stay_df, step_size=5., use_code=False):
    mv_event_df = pd.read_csv(os.path.join(mdir, 'mv_input_events.csv'))
    mv_event_df['TIMESTAMP'] = pd.to_datetime(mv_event_df['TIMESTAMP'])
    mv_event_df = mv_event_df.dropna(subset=['HADM_ID'])
    mv_event_df['HADM_ID'] = mv_event_df['HADM_ID'].astype(int).astype(str)
    mv_event_df['QUANTILE_VALUE'] = mv_event_df['QUANTILE_VALUE'].astype(int).astype(str)
    mv_event_df['QUANTILE_VALUE'] = 'Q_' + mv_event_df['QUANTILE_VALUE']
    mv_event_df = mv_event_df.merge(stay_df[['HADM_ID', 'INTIME', 'OUTTIME', 'CUSTOM_ID']],
                                      left_on='HADM_ID', right_on='HADM_ID', how='inner')
    mv_event_df = mv_event_df[mv_event_df['TIMESTAMP'] <= mv_event_df['OUTTIME']]
    mv_event_df['TOKEN_CODE'] = mv_event_df['TOKEN_CODE'].astype(str) + '_mv'
    if use_code:
        mv_event_df['OUT_STRING'] = mv_event_df['TOKEN_CODE']
        mv_event_df['OUT_STRING'] = mv_event_df['OUT_STRING'] + ' ' + mv_event_df['QUANTILE_VALUE']
    else:
        mv_event_df = mv_event_df.merge(code_df[['TOKEN_CODE', 'TOKEN_STRING']],
                                          left_on='TOKEN_CODE', right_on='TOKEN_CODE')
        mv_event_df = mv_event_df.rename(columns={'TOKEN_STRING': 'OUT_STRING'})
        mv_event_df['OUT_STRING'] = mv_event_df['OUT_STRING'] + ' ' + mv_event_df['QUANTILE_VALUE']
    time_steps = (mv_event_df['TIMESTAMP'] - mv_event_df['INTIME']).dt.total_seconds() // (step_size * 3600 + 1e-4)
    mv_event_df = mv_event_df.assign(TIMESTEP=time_steps)
    mv_event_df = mv_event_df[mv_event_df['TIMESTEP'] >= 0]
    mv_event_df = mv_event_df.drop_duplicates(subset=['CUSTOM_ID', 'TOKEN_CODE'], keep='last')
    return mv_event_df[['CUSTOM_ID', 'OUT_STRING', 'TIMESTEP', 'TIMESTAMP', 'TOKEN_CODE']]


def process_lab_event(mdir, code_df, stay_df, step_size=5., use_code=False):
    lab_event_df = pd.read_csv(os.path.join(mdir, 'lab_events.csv'))
    lab_event_df['TIMESTAMP'] = pd.to_datetime(lab_event_df['TIMESTAMP'])
    lab_event_df['HADM_ID'] = lab_event_df['HADM_ID'].astype(int).astype(str)
    lab_event_df['QUANTILE_VALUE'] = lab_event_df['QUANTILE_VALUE'].astype(int).astype(str)
    lab_event_df['QUANTILE_VALUE'] = 'Q_'+lab_event_df['QUANTILE_VALUE']
    lab_event_df = lab_event_df.merge(stay_df[['HADM_ID', 'INTIME', 'OUTTIME', 'CUSTOM_ID']],
                                      left_on='HADM_ID', right_on='HADM_ID', how='inner')
    lab_event_df = lab_event_df[lab_event_df['TIMESTAMP'] <= lab_event_df['OUTTIME']]
    lab_event_df['TOKEN_CODE'] = lab_event_df['TOKEN_CODE'].astype(str) + '_lab'
    if use_code:
        lab_event_df['OUT_STRING'] = lab_event_df['TOKEN_CODE']
        lab_event_df['OUT_STRING'] = lab_event_df['OUT_STRING'] + ' ' + lab_event_df['QUANTILE_VALUE']
    else:
        lab_event_df = lab_event_df.merge(code_df[['TOKEN_CODE','TOKEN_STRING']],
                                          left_on='TOKEN_CODE', right_on='TOKEN_CODE')
        lab_event_df = lab_event_df.rename(columns={'TOKEN_STRING':'OUT_STRING'})
        lab_event_df['OUT_STRING'] = lab_event_df['OUT_STRING'] + ' ' + lab_event_df['QUANTILE_VALUE']
    time_steps = (lab_event_df['TIMESTAMP'] - lab_event_df['INTIME']).dt.total_seconds() // (step_size * 3600 + 1e-4)
    lab_event_df = lab_event_df.assign(TIMESTEP=time_steps)
    lab_event_df = lab_event_df[lab_event_df['TIMESTEP'] >= 0]
    lab_event_df = lab_event_df.drop_duplicates(subset=['CUSTOM_ID', 'TOKEN_CODE'], keep='last')
    return lab_event_df[['CUSTOM_ID', 'OUT_STRING', 'TIMESTEP', 'TIMESTAMP', 'TOKEN_CODE']]


def process_proc_event(mdir, code_df, stay_df, step_size=5., use_code=False):
    proc_event_df = pd.read_csv(os.path.join(mdir, 'mv_proc_events.csv'))
    proc_event_df['TIMESTAMP'] = pd.to_datetime(proc_event_df['TIMESTAMP'])
    proc_event_df['HADM_ID'] = proc_event_df['HADM_ID'].astype(int).astype(str)
    proc_event_df = proc_event_df.merge(stay_df[['HADM_ID', 'INTIME', 'OUTTIME', 'CUSTOM_ID']],
                                        left_on='HADM_ID', right_on='HADM_ID', how='inner')
    proc_event_df = proc_event_df[proc_event_df['STATUSDESCRIPTION'] == 'FinishedRunning']
    proc_event_df['TOKEN_CODE'] = proc_event_df['TOKEN_CODE'].astype(str) + '_proc'
    if use_code:
        proc_event_df['OUT_STRING'] = proc_event_df['TOKEN_CODE']
    else:
        proc_event_df = proc_event_df.merge(code_df[['TOKEN_CODE','TOKEN_STRING']],
                                            left_on='TOKEN_CODE', right_on='TOKEN_CODE')
        proc_event_df = proc_event_df.rename(columns={'TOKEN_STRING': 'OUT_STRING'})
    time_steps = (proc_event_df['TIMESTAMP'] - proc_event_df['INTIME']).dt.total_seconds()//(step_size*3600+1e-4)
    proc_event_df = proc_event_df.assign(TIMESTEP=time_steps)
    proc_event_df = proc_event_df[proc_event_df['TIMESTEP'] >= 0]

    return proc_event_df[['CUSTOM_ID', 'OUT_STRING', 'TIMESTEP', 'TIMESTAMP', 'TOKEN_CODE']]


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    patch_size = args.patch_size
    max_patch = args.max_patch
    gone_native = args.use_native_codes

    print('Begin processing EHR events')
    stay_mapping = process_stay_df(data_dir)
    token_mapping = load_token_dfs(os.path.join(data_dir, 'events'))
    train_list = get_ts_episodes(data_dir,'train')
    test_list = get_ts_episodes(data_dir,'test')
    lab_event = process_lab_event(os.path.join(data_dir, 'events'),
                                  code_df=token_mapping, stay_df=stay_mapping,
                                  step_size=patch_size, use_code=gone_native)
    #lab_event.to_csv(os.path.join(save_dir, 'processed_lab_events.csv'))
    proc_event = process_proc_event(os.path.join(data_dir, 'events'), code_df=token_mapping,
                                    stay_df=stay_mapping, step_size=patch_size, use_code=gone_native)
    #proc_event.to_csv(os.path.join(save_dir, 'processed_mv_proc_events.csv'))
    cv_event = process_cv_events(os.path.join(data_dir, 'events'), code_df=token_mapping,
                                    stay_df=stay_mapping, step_size=patch_size, use_code=gone_native)
    #cv_event.to_csv(os.path.join(save_dir, 'processed_cv_input_events.csv'))
    mv_event = process_mv_events(os.path.join(data_dir, 'events'), code_df=token_mapping,
                                    stay_df=stay_mapping, step_size=patch_size, use_code=gone_native)
    #mv_event.to_csv(os.path.join(save_dir, 'processed_mv_input_events.csv'))
    event_df = pd.concat([lab_event, proc_event, cv_event, mv_event])
    event_df = event_df.sort_values(by=['CUSTOM_ID',
                                        'TIMESTAMP']).reset_index(drop=True).drop(columns='TIMESTAMP')

    event_df.to_csv(os.path.join(save_dir, 'processed_events.csv'))
    token_mapping = token_mapping[token_mapping['TOKEN_CODE'].isin(event_df['TOKEN_CODE'])]
    token_mapping.to_csv(os.path.join(save_dir, 'token_lists.csv'))
    event_df = event_df.drop(columns=['TOKEN_CODE'])
    print('Events processed successfully')
    print('Begin building EHR event sequences')
    event_strings = event_df[['CUSTOM_ID', 'OUT_STRING',
                              'TIMESTEP']].groupby(['CUSTOM_ID', 'TIMESTEP'])['OUT_STRING'].apply(list)

    event_strings = event_strings.unstack('TIMESTEP').fillna('PAD')

    event_nest = event_strings.values.tolist()  # nested list
    new_idx = event_strings.index.to_list()
    event_strings = dict(zip(new_idx, event_nest))

    new_train_list = list(set.intersection(set(new_idx), set(train_list)))
    new_test_list = list(set.intersection(set(new_idx), set(test_list)))
    new_train_len = len(new_train_list)
    new_test_len = len(new_test_list)

    with open(os.path.join(save_dir,'data_partition.json'),'w') as f:
        part_list = {'train': new_train_list, 'test': new_test_list}
        json.dump(part_list, f)
    print(f'Finished building event sequences for {new_train_len} training and {new_test_len} testing samples')
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
    print('Begin saving EHR sequences')

    train_save_dir = os.path.join(save_dir, 'train/ehr_seq')
    test_save_dir = os.path.join(save_dir, 'test/ehr_seq')
    for k, v in string_objs.items():
        k_fname = k+'_sequence.json'
        if k in new_train_list:
            with open(os.path.join(train_save_dir, k_fname), 'w') as f:
                json.dump(v, f)
        elif k in new_test_list:
            with open(os.path.join(test_save_dir, k_fname), 'w') as f:
                json.dump(v, f)



