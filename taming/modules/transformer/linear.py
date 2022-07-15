import logging
from turtle import forward

import torch
import torch.nn as nn
from .gpt import GPTConfig, LayerNorm
from .sandwich import GPT
from performer_pytorch import SelfAttention, CrossAttention
logger = logging.getLogger(__name__)

class LinearBlock(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.ln1 = LayerNorm(config.n_embd)
        self.ln1_2 = LayerNorm(config.n_embd)
        self.ln2 = LayerNorm(config.n_embd)
        self.ln2_2 = LayerNorm(config.n_embd)
        self.attn = SelfAttention(config.n_embd, causal=True, heads=config.n_head)
        if config.add_cross:
            self.cross_attn = CrossAttention(config.n_embd, config.n_head)
            self.ln_cross = LayerNorm(config.n_embd)
            self.ln_cross_2 = LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, eh=None,  *args, **kargs):
        # layer past: tuple of length two with B, nh, T, hs
        x_norm =self.ln1(x)
        att = self.attn(x_norm)
        x = x + (self.ln1_2(att))

        if hasattr(self, "cross_attn") and eh is not None:
            x = x + self.ln_cross_2(self.cross_attn(self.ln_cross(x, context=eh)))

        x = x + self.ln2_2(self.mlp(self.ln2(x)))
        return x

class LinearGPT(GPT):
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
        self.blocks = nn.ModuleList([LinearBlock(config) for _ in range(config.n_layer)])
        # decoder head
        head_size = text_vbs + img_vbs if full_head else img_vbs
        self.ln_f = LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, head_size, bias=False)   
        self.apply(self._init_weights)

        self.block_size = config.block_size
        self.add_cross = add_cross
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))