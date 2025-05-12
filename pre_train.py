import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from dataset.MIMIC_3 import MIMIC3Dataset, MIMIC3Collate
from dataset.eICU import eICUDataset, eICUCollate
import model.clinical_roberta as c_roberta
import model.ehr_gpt2 as c_gpt
import model.clinical_longformer as c_longformer
import model.bio_clinical_bert as c_bert
from model.loss import PatchContrastiveLoss, PatchReconLoss
import argparse
from tqdm import tqdm
from torch.optim import Adam

"""
Warning message regarding newly initialised weights is largely harmless,
New token embedding weights are re-computed after initialisation
"""

parser = argparse.ArgumentParser(description='physionet 2012')
parser.add_argument('--max_steps', type=int, default=200,
                    help='Max lengths of TS and EHR sequences in hours')
parser.add_argument('--n_vars', type=int, default=11, help='Number of TS variables')
parser.add_argument('--patch_size', type=int, default=5, help='Size of temporal patche in hours')
parser.add_argument('--LLM_name', type=str, default='gpt2', help='Name of the huggingface LLM')
parser.add_argument('--dataset', type=str, default='MIMIC-III', help='Name of the pretraining dataset')
parser.add_argument('--cont_temp', type=float, default=0.02, help='The value of tau')
parser.add_argument('--mask_omega', type=bool, default=True, help='Masking the upper diag of omega')
parser.add_argument('--freeze_word_embedding', type=bool, default=True, help='Freeze LLM word embedding')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_epoch', type=int, default=5, help='Number of pretraining epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--clipping_value', type=float, default=2, help='Gradient clipping')
parser.add_argument('--save_dir', type=str, default='test_folder', help='Directory for model checkpoints')
parser.add_argument('--save_name', type=str, default='test_model', help='Name for saving the pretrained model')



def prepare_model(tok_pth, tokeniser_name, add_pad=False, n_vars=11, max_steps=200, patch_size=5, 
                  freeze_word_ebd=True, dataset='MIMIC-III'):

    token_list = pd.read_csv(tok_pth)
    quantile_tokens = [f'Q_{1}' for i in range(10)]
    if dataset == 'MIMIC-III':
        token_code_list = token_list['TOKEN_CODE'].to_list()
        token_string_list = token_list['TOKEN_STRING'].to_list()
    elif dataset == 'eICU': 
        token_code_list = token_list['event_token'].to_list()
        token_string_list = token_list['token_string'].to_list()
    
    tokeniser = AutoTokenizer.from_pretrained(tokeniser_name)
    if add_pad:
        tokeniser.add_special_tokens({'pad_token': '[PAD]'})

    if tokeniser_name in ['gpt2', 'gpt2-xl']:
        test_model = c_gpt.EHRGPTModel(n_vars, max_steps, patch_size, gpt_name=tokeniser_name)
        test_model.resize_token_embeddings(len(tokeniser))
        new_tokens = token_code_list+quantile_tokens
        new_token_strings = token_string_list+quantile_tokens
        c_gpt.expand_and_init_token_embedding(new_tokens, new_token_strings, tokeniser, test_model)

    elif tokeniser_name == 'emilyalsentzer/Bio_ClinicalBERT':
        test_model = c_bert.BioClinModel(n_vars, max_steps, patch_size)
        test_model.resize_token_embeddings(len(tokeniser))
        new_tokens = token_code_list+quantile_tokens
        new_token_strings = token_string_list+quantile_tokens
        c_bert.expand_and_init_token_embedding(new_tokens, new_token_strings, tokeniser, test_model)

    elif tokeniser_name == 'yikuan8/Clinical-Longformer':
        test_model = c_longformer.EHRLongformerModel(n_vars, max_steps, patch_size)
        test_model.resize_token_embeddings(len(tokeniser))
        new_tokens = token_code_list+quantile_tokens
        new_token_strings = token_string_list+quantile_tokens
        c_longformer.expand_and_init_token_embedding(new_tokens, new_token_strings, tokeniser, test_model)

    elif tokeniser_name == 'FacebookAI/roberta-base':
        test_model = c_roberta.EHRRobertaModel(n_vars, max_steps, patch_size)
        test_model.resize_token_embeddings(len(tokeniser))
        new_tokens = token_code_list+quantile_tokens
        new_token_strings = token_string_list+quantile_tokens
        c_roberta.expand_and_init_token_embedding(new_tokens, new_token_strings, tokeniser, test_model)

    new_token_ids = tokeniser.convert_tokens_to_ids(new_tokens)
    test_model.resize_token_embeddings(len(tokeniser))
    if freeze_word_ebd:
        test_model.fix_token_embeddings(new_token_idx=new_token_ids)
    
    return test_model, tokeniser


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert args.dataset in ['MIMIC-III', 'eICU']
    assert args.LLM_name in ['gpt2', 'gpt2-xl', 'emilyalsentzer/Bio_ClinicalBERT', 
                              'yikuan8/Clinical-Longformer', 'FacebookAI/roberta-base']
    if args.dataset == 'MIMIC-III':
        tok_pth = 'processed data/MIMIC_3/MIMIC_3/token_lists.csv'
    elif args.dataset == 'eICU':
        tok_pth = 'processed data/eICU/eICU/new_token_list.csv'

    add_pad = False
    if args.LLM_name == 'gpt2':
        add_pad = True

    test_model, tokeniser = prepare_model(tok_pth, args.LLM_name, add_pad, 
                                          args.n_vars, args.max_steps, 
                                          args.patch_size, args.freeze_word_embedding, args.dataset)

    if args.dataset == 'MIMIC-III':
        train_dir = 'processed data/MIMIC_3/MIMIC_3/train'
        test_dir = 'processed data/MIMIC_3/MIMIC_3/test'

        train_dataset = MIMIC3Dataset(train_dir, max_steps=args.max_steps, patch_size=args.patch_size)
        test_dataset = MIMIC3Dataset(test_dir, max_steps=args.max_steps, patch_size=args.patch_size)

        bio_collate_fn = MIMIC3Collate(tokeniser, max_steps=args.max_steps, patch_size=args.patch_size)

        train_loader = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=True, collate_fn=bio_collate_fn)
        test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, collate_fn=bio_collate_fn)

    elif args.dataset == 'eICU':
        train_dir = 'processed data/eICU/eICU'
        test_dir = 'processed data/eICU/eICU'

        with open(os.path.join(train_dir, 'revised_partition.json'),'r') as f:
            partition = json.load(f)

        train_list = partition['train']
        test_list = partition['test']
        train_dataset = eICUDataset(train_dir, train_list, max_steps=args.max_steps, patch_size=args.patch_size)
        test_dataset = eICUDataset(test_dir, test_list, max_steps=args.max_steps, patch_size=args.patch_size)

        bio_collate_fn = eICUCollate(tokeniser, max_steps=args.max_steps, patch_size=args.patch_size)

        train_loader = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=True, collate_fn=bio_collate_fn)
        test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, collate_fn=bio_collate_fn)

    
    test_model.to(device)
    patch_cont_loss = PatchContrastiveLoss(temperature=args.cont_temp).to(device)
    patch_recon_loss = PatchReconLoss().to(device)

    optim = Adam(test_model.parameters(), lr=args.lr)

    for e in range(args.max_epoch):
        epoch_cont_loss = 0.
        epoch_recon_loss = 0.
        epoch_total_loss = 0.
        for i, (b_ts, b_id, b_attn, b_mask, b_tok_mask) in tqdm(enumerate(train_loader), total=len(train_loader)):
            b_ts, b_id, b_attn, b_mask, b_tok_mask = b_ts.to(device), b_id.to(device), b_attn.to(device), b_mask.to(device), b_tok_mask.to(device)
            if args.LLM_name == 'emilyalsentzer/Bio_ClinicalBERT':
                force_max_len = 512
            else:
                force_max_len = tokeniser.model_max_length
            if b_id.shape[-1] > force_max_len:
                b_id = b_id[:, :force_max_len]
                b_attn = b_attn[:, :force_max_len]
                b_tok_mask = b_tok_mask[:,:, :force_max_len]
            b_seq_out, b_ts_out, b_ts_pred, b_omega = test_model(b_ts, b_id, b_attn, b_tok_mask)
            b_mask = b_mask.flatten().unsqueeze(dim=-1)
            b_mask = b_mask @ b_mask.T
            b1,b2 = patch_cont_loss(b_ts_out, b_seq_out, b_omega, b_mask)
            b_cont_loss = b1+b2
            b_recon_loss = torch.clamp(patch_recon_loss(b_ts_pred, b_ts), max=10)
            b_loss = b_cont_loss+b_recon_loss
            b_loss.backward()
            torch.nn.utils.clip_grad_norm_(test_model.parameters(), args.clipping_value)
            optim.step()
            optim.zero_grad()
            b_cont_val = b_cont_loss.detach().cpu().item()
            b_recon_val = b_recon_loss.detach().cpu().item()
            b_total_val = b_loss.detach().cpu().item()
            epoch_cont_loss += b_cont_val
            epoch_recon_loss += b_recon_val
            epoch_total_loss += b2.detach().cpu().item()
            if i%50 == 0:
                tqdm.write(f'epoch: {e}, batch: {i}, total_loss: {b_total_val}, cont loss: {b_cont_val}, recon loss: {b_recon_val}')
        epoch_cont_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_total_loss /= len(train_loader)
        print(f'epoch: {e}, avg loss: {epoch_total_loss}, cont loss: {epoch_cont_loss}, recon loss: {epoch_recon_loss}')

    tokeniser.save_pretrained(os.path.join(args.save_dir, args.save_name))
    torch.save(test_model.state_dict(), os.path.join(args.save_dir, args.save_name + '.pt'))