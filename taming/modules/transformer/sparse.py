import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from .gpt import GPTConfig, LayerNorm, Attention
from .sandwich import SandwichBlock, GPT
from fast_transformers.attention import CausalLinearAttention
logger = logging.getLogger(__name__)

class SparseAttention(Attention):
    def __init__(self, config):
        super().__init__(config)

    def att_score(self, q , k):
        # (B, nh, T_q, hs) x (B, nh, T_k, hs) -> (B, nh, T_q, T_k)
        if self.PBrelax:
            att = torch.einsum("bnqh,bnkh->bnqk", q * (1.0 / (self.alpha*math.sqrt(k.size(-1)))), k)
            att = (att - att.abs().max().detach()) * self.alpha
        else:
            att = torch.einsum("bnqh,bnkh->bnqk", q * (1.0 / math.sqrt(k.size(-1))), k)
        return att

    def self_att_score(self, q, k):
        # (B, nh, T_q, hs) x (B, nh, T_q, hs) -> (B, nh, T_q)
        if self.PBrelax:
            att = torch.einsum("bnlh,bnlh->bnl", q * (1.0/ self.alpha / math.sqrt(k.size(-1))), k)
            att = (att - att.abs().max().detach()) * self.alpha
        else:
            att = torch.einsum("bnlh,bnlh->bnl", q, k)
        return att

    def forward(self, query, key, value, att_mask=None, layer_past=None, return_present=False):
        B, C = query.shape[0], query.shape[2]
        T_q, T_k, T_v = query.shape[1], key.shape[1], value.shape[1]
        assert T_q == T_k and T_k == T_v

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.W_q(query).view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_q, hs)
        k = self.W_k(key).view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_k, hs)
        v = self.W_v(value).view(B, T_v, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_v, hs)
        
        att_score = torch.empty(B, self.n_head, T_q, T_q, device=v.device)
        for t in range(int(math.sqrt(T_q))):
            att_score[:,:,t**2:(t+1)**2,:(t+1)**2] = self.att_score(q[:,:,t**2:(t+1)**2,:], k[:,:,:(t+1)**2,:])
        if att_mask is not None:
            att_score = att_score.masked_fill(att_mask == 0, float('-inf'))

        att_score = self.attn_drop(F.softmax(att_score, dim=-1)) # (B, nh, 2*t+1, t**2+1)
        y = att_score @ v # (B, nh, T_q, T_q) x (B, nh, T_v, hs) -> (B, nh, T_q, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        # output projection
        y = self.resid_drop(self.proj(y))

        present = None
        if return_present:
            present = torch.stack((k, v))
            if layer_past is not None:
                past_key, past_value = layer_past
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
        return y, present # TODO: check that this does not break anything

class SparseBlock(SandwichBlock):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.ln1 = LayerNorm(config.n_embd)
        self.ln1_2 = LayerNorm(config.n_embd)
        self.ln2 = LayerNorm(config.n_embd)
        self.ln2_2 = LayerNorm(config.n_embd)
        self.attn = SparseAttention(config)
        if config.add_cross:
            self.cross_attn = Attention(config)
            self.ln_cross = LayerNorm(config.n_embd)
            self.ln_cross_2 = LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

class SparseGPT(GPT):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, text_vbs, img_vbs, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., dim_cond=512, 
                 add_cross=False, full_head=False, PBrelax=False):
        nn.Module.__init__(self)
        config = GPTConfig(vocab_size=text_vbs+img_vbs, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                           add_cross=add_cross, full_head=full_head, PBrelax=PBrelax)
        # input embedding stem
        if add_cross:
            self.eh_proj = nn.Linear(dim_cond, n_embd)
        #self.ctx_proj = nn.Linear(dim_cond, n_embd)
        self.tok_emb = nn.Embedding(config.vocab_size+1, config.n_embd) # 1 for pad
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([SparseBlock(config) for _ in range(config.n_layer)])
        # decoder head
        head_size = text_vbs + img_vbs if full_head else img_vbs
        self.ln_f = LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, head_size, bias=False)   
        self.apply(self._init_weights)

        self.block_size = config.block_size
        self.add_cross = add_cross
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
