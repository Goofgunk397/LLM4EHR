import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle
from transformers import AutoTokenizer, AutoModel, utils


def expand_and_init_token_embedding(new_token, new_token_string, tokeniser, encoder):
    new_token_tok = tokeniser(new_token_string, padding='do_not_pad')
    all_ebd = []
    for s in new_token_tok['input_ids']:
        s_embed = encoder.encoder.base_model.embeddings.word_embeddings.weight.data[s]
        s_embed = s_embed.mean(dim=0)
        all_ebd.append(s_embed)

    new_token_embed = torch.stack(all_ebd, dim=0)
    tokeniser.add_tokens(new_token)
    new_token_ids = tokeniser.convert_tokens_to_ids(new_token)
    encoder.resize_token_embeddings(len(tokeniser))
    encoder.set_token_embeddings(new_token_ids, new_token_embed)

class TSAvgPooler(nn.Module):
    def __init__(self, in_size=768, max_steps=200, patch_size=5):
        super().__init__()
        self.split_size = patch_size

    def forward(self, x_hidden):
        x_split = torch.split(x_hidden, self.split_size, dim=1)
        x_hidden = torch.stack(x_split, dim=1)
        x_hidden = F.tanh(x_hidden.mean(dim=2))
        return x_hidden


class TSEncoder(nn.Module):
    def __init__(self, n_channel, out_size=768):
        super().__init__()
        self.l1 = nn.Linear(n_channel, out_size)
        self.l2 = nn.Linear(out_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)

    def forward(self, x):
        x_out = F.relu(self.l1(x))
        x_out = F.relu(self.l2(x_out)).transpose(1,2)
        x_out = self.bn(x_out)
        return x_out.transpose(1, 2)


class TSDecoder(nn.Module):
    """
    GPT 2 ts decoding is autoregressive
    """
    def __init__(self, n_channel, in_size=768, patch_size=5):
        super().__init__()
        self.d1 = nn.Linear(in_size, n_channel)

    def forward(self, x_hidden):
        """
        input: (bsz, n_patch, hidden_size)
        about 3m params for 200 time steps with 5 hour patches
        """
        x_out = F.relu(self.d1(x_hidden))
        return x_out
    

class EHRRobertaEncoder(nn.Module):
    """
    Using clinical longformer
    """
    def __init__(self, n_channel, max_steps=200, patch_size=5):
        super().__init__()
        self.base_model = AutoModel.from_pretrained("FacebookAI/roberta-base")
        self.ebd_size = self.base_model.get_input_embeddings().weight.shape[1]
        self.seq_proj = nn.Linear(self.ebd_size, self.ebd_size)
        self.ts_encoder = TSEncoder(n_channel, self.ebd_size)
        self.ts_pooler = TSAvgPooler(self.ebd_size, max_steps, patch_size)
        self.split_size = patch_size
        self.max_steps = max_steps

        for param in self.base_model.parameters():
            param.requires_grad = False

    def token_string_embedding(self, input_ids, attention_mask):

        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)

    def resize_token_embeddings(self, *kwargs):

        return self.base_model.resize_token_embeddings(*kwargs)

    def set_token_embeddings(self, idx, new_embedding):
        self.base_model.embeddings.word_embeddings.weight.data[idx] = new_embedding

    def fix_token_embeddings(self, new_token_idx=None):
        for param in self.base_model.embeddings.word_embeddings.parameters():
            param.requires_grad = False

        if new_token_idx is not None:
            self.base_model.embeddings.word_embeddings.weight.data[new_token_idx].requires_grad = True

    def encode_ts(self, ts_in):
        ts_out = self.ts_encoder(ts_in)
        ts_hidden = self.base_model(inputs_embeds=ts_out)['last_hidden_state']

        return ts_hidden

    def forward(self, ts_in, seq_in, seq_attn_mask, seq_token_mask):
        seq_out = self.base_model(input_ids=seq_in, attention_mask=seq_attn_mask)
        seq_pool_out = torch.bmm(seq_token_mask, seq_out['last_hidden_state'])
        seq_div = seq_token_mask.sum(dim=-1).unsqueeze(dim=-1)
        seq_div += 1e-6
        seq_pool_out = seq_pool_out/seq_div
        seq_pool_out = F.tanh(self.seq_proj(seq_pool_out))
        seq_norm = F.normalize(seq_pool_out, dim=-1)
        omega = torch.bmm(seq_norm, seq_norm.transpose(1,2))
        omega = torch.block_diag(*[x.squeeze() for x in torch.split(omega, 1, dim=0)])
        ts_out = self.encode_ts(ts_in)
        ts_pool = self.ts_pooler(ts_out)
        ts_out_final = (ts_out, ts_pool)
        return seq_pool_out, ts_out_final, omega
    

class EHRRobertaModel(nn.Module):
    def __init__(self, n_channel, max_steps=200, patch_size=5):
        super().__init__()
        self.encoder = EHRRobertaEncoder(n_channel, max_steps, patch_size)
        self.ts_decoder = TSDecoder(n_channel, in_size=self.encoder.ebd_size, patch_size=5)

    def fix_token_embeddings(self, new_token_idx):
        self.encoder.fix_token_embeddings(new_token_idx)

    def set_token_embeddings(self, idx, new_embedding):
        self.encoder.set_token_embeddings(idx, new_embedding)

    def resize_token_embeddings(self, *kwargs):

        return self.encoder.resize_token_embeddings(*kwargs)

    def token_string_embedding(self, input_ids, attention_mask):
        return self.encoder.token_string_embedding(input_ids, attention_mask)

    def forward(self, ts_in, seq_in, seq_attn_mask, seq_token_mask):
        seq_out, ts_out, omega = self.encoder(ts_in, seq_in, seq_attn_mask, seq_token_mask)
        ts_decode = self.ts_decoder(ts_out[0])

        return seq_out, ts_out[1], ts_decode, omega
