import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from dataset.MIMIC_3 import MIMIC3ClassificationDataset, MIMIC3DynamicDataset
import model.clinical_roberta as c_roberta
import model.ehr_gpt2 as c_gpt
import model.clinical_longformer as c_longformer
import model.bio_clinical_bert as c_bert
from model.heads import SequenceClassificationHead, DynamicClassificationHead
from dataset.eICU import eICUClassificationDataset, eICUDynamicDataset
import argparse
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, root_mean_squared_error


parser = argparse.ArgumentParser(description='Args for finetuning')
parser.add_argument('--max_steps', type=int, default=200,
                    help='Max lengths of TS and EHR sequences in hours')
parser.add_argument('--n_vars', type=int, default=11, help='Number of TS variables')
parser.add_argument('--patch_size', type=int, default=5, help='Size of temporal patche in hours')
parser.add_argument('--LLM_name', type=str, default='gpt2', help='Name of the huggingface LLM')
parser.add_argument('--pt_path', type=str, default='test_folder/test_model', help='Path to the pretrained model')
parser.add_argument('--dataset', type=str, default='MIMIC-III', help='Name of the pretraining dataset')
parser.add_argument('--task', type=str, default='ihm_48', help='Name of the downstream task')
parser.add_argument('--n_runs', type=int, default=10, help='Number of runs')
parser.add_argument('--freeze_word_embedding', type=bool, default=True, help='Freeze LLM word embedding')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_epoch', type=int, default=10, help='Number of finetuning epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--clipping_value', type=float, default=2, help='Gradient clipping')
parser.add_argument('--save_dir', type=str, default='test_folder', help='Directory for saving results')
parser.add_argument('--save_name', type=str, default='test_model', help='Name of the output file')


def prepare_model(tok_pth, tokeniser_name, model_pth, dataset, 
                  add_pad=False, n_vars=11, max_steps=200, patch_size=5, freeze_word_ebd=True):

    token_list = pd.read_csv(tok_pth)
    quantile_tokens = [f'Q_{1}' for i in range(10)]
    if dataset == 'MIMIC-III':
        token_code_list = token_list['TOKEN_CODE'].to_list()
        token_string_list = token_list['TOKEN_STRING'].to_list()
    elif dataset == 'eICU': 
        token_code_list = token_list['event_token'].to_list()
        token_string_list = token_list['token_string'].to_list()
    
    tokeniser = AutoTokenizer.from_pretrained(model_pth)
    if add_pad:
        tokeniser.add_special_tokens({'pad_token': '[PAD]'})

    if tokeniser_name == 'gpt2':
        test_model = c_gpt.EHRGPTModel(n_vars, max_steps, patch_size)
        test_model.resize_token_embeddings(len(tokeniser))

    elif tokeniser_name == 'emilyalsentzer/Bio_ClinicalBERT':
        test_model = c_bert.BioClinModel(n_vars, max_steps, patch_size)
        test_model.resize_token_embeddings(len(tokeniser))

    elif tokeniser_name == 'yikuan8/Clinical-Longformer':
        test_model = c_longformer.EHRLongformerModel(n_vars, max_steps, patch_size)
        test_model.resize_token_embeddings(len(tokeniser))

    elif tokeniser_name == 'FacebookAI/roberta-base':
        test_model = c_roberta.EHRRobertaModel(n_vars, max_steps, patch_size)
        test_model.resize_token_embeddings(len(tokeniser))


    model_checkpoint = torch.load(model_pth+'.pt')
    test_model.load_state_dict(model_checkpoint)
    test_model = test_model.encoder.eval()  # Only encoder is used for pre-training

    # Hard freeze weights
    for param in test_model.parameters():
        param.requires_grad = False 

    
    return test_model, tokeniser

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert args.dataset in ['MIMIC-III', 'eICU']
    assert args.LLM_name in ['gpt2', 'emilyalsentzer/Bio_ClinicalBERT', 
                              'yikuan8/Clinical-Longformer', 'FacebookAI/roberta-base']
    assert args.task in ['ihm_48', 'pheno', 'ihm_rolling', 'los_rolling']
    if args.dataset == 'MIMIC-III':
        tok_pth = 'processed data/MIMIC_3/MIMIC_3/token_lists.csv'
    elif args.dataset == 'eICU':
        tok_pth = 'processed data/eICU/eICU/new_token_list.csv'

    add_pad = False

    test_model, tokeniser = prepare_model(tok_pth, args.LLM_name, args.pt_path, args.dataset, add_pad, 
                                          args.n_vars, args.max_steps, 
                                          args.patch_size, args.freeze_word_embedding)
    
    test_model.to(device)
    met_1 = []
    met_2 = []
    for _ in range(args.n_runs):

        if args.dataset == 'MIMIC-III':
            test_dir = 'processed data/MIMIC_3/MIMIC_3/test'
            test_list = os.listdir('processed data/MIMIC_3/MIMIC_3/test/timeseries')
            test_list = ['_'.join(x.split('_')[:2]) for x in test_list]
            tune_len = int(0.7*len(test_list))
            tune_list = test_list[:tune_len]
            eval_list = test_list[tune_len:]
            if args.task in ['ihm_48', 'pheno']:
                tune_dataset = MIMIC3ClassificationDataset(test_dir, tune_list, max_steps=args.max_steps, 
                                                        patch_size=args.patch_size, task=args.task)
                eval_dataset = MIMIC3ClassificationDataset(test_dir, eval_list, max_steps=args.max_steps, 
                                                        patch_size=args.patch_size, task=args.task)
                tune_loader = DataLoader(tune_dataset,  batch_size=args.batch_size, shuffle=True)
                eval_loader = DataLoader(eval_dataset,  batch_size=args.batch_size, shuffle=False)
                if args.task == 'ihm_48':
                    classifier = SequenceClassificationHead(hidden_size=test_model.ebd_size, 
                                                            label_size=1, max_length=48, dense=True, pool='max') 
                elif args.task == 'pheno':
                    classifier = SequenceClassificationHead(hidden_size=test_model.ebd_size, 
                                                            label_size=25, max_length=args.max_steps, 
                                                            dense=False) 
                loss = torch.nn.BCELoss()

            elif args.task in ['ihm_rolling', 'los_rolling']:
                tune_dataset = MIMIC3DynamicDataset(test_dir, tune_list, max_steps=args.max_steps, patch_size=args.patch_size, 
                                                    task=args.task)
                eval_dataset = MIMIC3DynamicDataset(test_dir, eval_list, max_steps=args.max_steps, 
                                                    patch_size=args.patch_size, task=args.task)
                tune_loader = DataLoader(tune_dataset,  batch_size=args.batch_size, shuffle=True)
                eval_loader = DataLoader(eval_dataset,  batch_size=args.batch_size, shuffle=False)
                classifier = DynamicClassificationHead(hidden_size=test_model.ebd_size, 
                                                    label_size=1, encoder=test_model, 
                                                    max_length=args.max_steps, task=args.task)
                if args.task == 'ihm_rolling':
                    loss = torch.nn.BCELoss()
                elif args.task == 'los_rolling':
                    loss = torch.nn.MSELoss()

        elif args.dataset == 'eICU':
            data_dir = 'processed data/eICU/eICU'
            with open(os.path.join(data_dir, 'revised_partition.json'), 'r') as f:
                partition = json.load(f)
            tune_list = partition['val']
            eval_list = partition['test']
            if args.task in ['ihm_48', 'pheno']:
                tune_dataset = eICUClassificationDataset(data_dir, tune_list, max_steps=args.max_steps, 
                                                        patch_size=args.patch_size, task=args.task)
                eval_dataset = eICUClassificationDataset(data_dir, eval_list, max_steps=args.max_steps, 
                                                        patch_size=args.patch_size, task=args.task)
                tune_loader = DataLoader(tune_dataset,  batch_size=args.batch_size, shuffle=True)
                eval_loader = DataLoader(eval_dataset,  batch_size=args.batch_size, shuffle=False)
                if args.task == 'ihm_48':
                    classifier = SequenceClassificationHead(hidden_size=test_model.ebd_size, 
                                                            label_size=1, max_length=48, dense=True, pool='max') 
                elif args.task == 'pheno':
                    classifier = SequenceClassificationHead(hidden_size=test_model.ebd_size, 
                                                            label_size=25, max_length=args.max_steps, 
                                                            dense=False) 
                loss = torch.nn.BCELoss()

            if args.task in ['ihm_rolling', 'los_rolling']:
                tune_dataset = eICUDynamicDataset(data_dir, tune_list, max_steps=args.max_steps, 
                                                patch_size=args.patch_size, task=args.task)
                eval_dataset = eICUDynamicDataset(data_dir, eval_list, max_steps=args.max_steps, 
                                                patch_size=args.patch_size, task=args.task)
                tune_loader = DataLoader(tune_dataset,  batch_size=args.batch_size, shuffle=True)
                eval_loader = DataLoader(eval_dataset,  batch_size=args.batch_size, shuffle=False)
                classifier = DynamicClassificationHead(hidden_size=test_model.ebd_size, 
                                                    label_size=1, encoder=test_model, 
                                                    max_length=args.max_steps, task=args.task)
                
                if args.task == 'ihm_rolling':
                    loss = torch.nn.BCELoss()
                elif args.task == 'los_rolling':
                    loss = torch.nn.MSELoss()

        classifier.to(device)
        loss.to(device)        
        if args.task in ['ihm_48', 'pheno']:        
            optim = Adam(classifier.parameters(), lr=args.lr)
            for e in range(args.max_epoch):
                epoch_loss = 0.
                for i, (x,y) in tqdm(enumerate(tune_loader), total=len(tune_loader)):
                    x,y = x.to(device), y.to(device)
                    z = test_model.encode_ts(x)
                    y_dot = classifier(z)
                    b_loss = loss(y_dot.squeeze(), y.squeeze())
                    b_loss.backward()
                    optim.step()
                    optim.zero_grad()
                    epoch_loss += b_loss.detach().cpu().item()
                    if i%50 == 0:
                        avg_loss = epoch_loss/(i+1)
                        print(f'epoch: {e}, step: {i+1}, loss: {avg_loss}')
                epoch_avg = epoch_loss/len(tune_loader)
                print(f'epoch: {e}, loss: {epoch_avg}')
                classifier.eval()
                all_out = []
                all_true = []
                for i, (x,y) in tqdm(enumerate(eval_loader), total=len(eval_loader)):
                    x,y = x.to(device), y.to(device)
                    z = test_model.encode_ts(x)
                    y_dot = classifier(z)
                    b_loss = loss(y_dot.squeeze(), y.squeeze())
                    all_out.append(np.array(y_dot.squeeze().detach().cpu()))
                    all_true.append(np.array(y.detach().cpu()))
                classifier.train()
                all_out_array = np.concatenate(all_out)
                all_true_array = np.concatenate(all_true)
                if args.task == 'ihm_48':
                    m1 = roc_auc_score(all_true_array, all_out_array)
                    m2 = average_precision_score(all_true_array, all_out_array)
                elif args.task == 'pheno':
                    m1 = roc_auc_score(all_true_array, all_out_array, average='macro')
                    m2 = roc_auc_score(all_true_array, all_out_array, average='micro')
                print(f'epoch: {e}, metric_1: {m1}, metric_2: {m2} ')

            met_1.append(m1)
            met_2.append(m2)
            del classifier
            del loss
            del optim

        elif args.task in ['ihm_rolling', 'los_rolling']:
                optim = Adam(classifier.parameters(), lr=args.lr)
                for e in range(args.max_epoch):
                    epoch_loss = 0.
                    for i, (x,m,y) in tqdm(enumerate(tune_loader), total=len(tune_loader)):
                        x,m,y = x.to(device), m.to(device), y.to(device)
                        m = torch.stack(torch.split(m, args.patch_size, dim=1), dim=1)
                        m_flat,_ = m.max(dim=-1)
                        y = torch.stack(torch.split(y, args.patch_size, dim=1), dim=1)
                        if args.task == 'los_rolling':
                            y_flat,_ = y.min(dim=-1)
                        else:
                            y_flat,_ = y.max(dim=-1)
                        y_flat = y_flat.flatten(0,1)
                        m_flat = m_flat.flatten(0,1)
                        y_dot = classifier(x)
                        y_dot = y_dot.flatten(0,1)
                        b_loss = loss(y_dot[m_flat==1].squeeze(), y_flat[m_flat==1])
                        b_loss = b_loss.mean()
                        b_loss.backward()
                        optim.step()
                        optim.zero_grad()
                        epoch_loss += b_loss.detach().cpu().item()
                        if i%50 == 0:
                            avg_loss = epoch_loss/(i+1)
                            print(f'epoch: {e}, step: {i+1}, loss: {avg_loss}')
                    epoch_avg = epoch_loss/len(tune_loader)
                    print(f'epoch: {e}, loss: {epoch_avg}')
                    classifier.eval()
                    all_out = []
                    all_true = []
                    for i, (x,m,y) in tqdm(enumerate(eval_loader), total=len(eval_loader)):
                        x,m,y = x.to(device), m.to(device), y.to(device)
                        m = torch.stack(torch.split(m, args.patch_size, dim=1), dim=1)
                        m_flat,_ = m.max(dim=-1)
                        y = torch.stack(torch.split(y, args.patch_size, dim=1), dim=1)
                        if args.task == 'los_rolling':
                            y_flat,_ = y.min(dim=-1)
                        else:
                            y_flat,_ = y.max(dim=-1)
                        y_flat = y_flat.flatten(0,1)
                        m_flat = m_flat.flatten(0,1)
                        y_dot = classifier(x)
                        y_dot = y_dot.flatten(0,1)
                        b_loss = loss(y_dot[m_flat==1].squeeze(), y_flat[m_flat==1])
                        b_loss = b_loss.mean()
                        all_out.append(np.array(y_dot[m_flat==1].squeeze().detach().cpu()))
                        all_true.append(np.array(y_flat[m_flat==1].squeeze().detach().cpu()))
                    classifier.train()
                    all_out_array = np.concatenate(all_out)
                    all_true_array = np.concatenate(all_true)
                    if args.task == 'ihm_rolling':
                        m1 = roc_auc_score(all_true_array, all_out_array)
                        m2 = average_precision_score(all_true_array, all_out_array)
                    elif args.task == 'los_rolling':
                        m1 = root_mean_squared_error(all_true_array, all_out_array)
                        m2 = r2_score(all_true_array, all_out_array)
                    print(f'epoch: {e}, metric_1: {m1}, metric_2: {m2} ')

                met_1.append(m1)
                met_2.append(m2)
                del classifier
                del loss
                del optim

    met_1_f = [float(x) for x in met_1]
    met_2_f = [float(x) for x in met_2]

    met_1_mean = np.mean(met_1_f).item()
    met_2_mean = np.mean(met_2_f).item()

    met_1_std = np.std(met_1_f).item()
    met_2_std = np.std(met_2_f).item()

    eval_obj = {'metric_1': {'all':met_1_f, 'mean':met_1_mean, 'std':met_1_std},
                'metric_2': {'all':met_2_f, 'mean':met_2_mean, 'std':met_2_std}}
    
    with open(os.path.join(args.save_dir, args.save_name + '.json'), 'w') as f:
        json.dump(eval_obj, f)

               
