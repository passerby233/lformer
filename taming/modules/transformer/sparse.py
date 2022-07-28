import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt
from deepspeed.ops.sparse_attention.sparsity_config import SparsityConfig
from deepspeed.ops.sparse_attention import SparseSelfAttention
import deepspeed

from .sandwich import LayerNorm, Attention, SandwichLayer, GPT

checkpoint =  deepspeed.checkpointing.checkpoint
import logging
logger = logging.getLogger(__name__)

class LformerSparsityConfig(SparsityConfig):
    def __init__(self, num_heads, block=16, css=32, different_layout_per_head=False):
        super().__init__(num_heads, block, different_layout_per_head)
        self.css = css

    def make_layout(self, seq_len):
        layout = self.setup_layout(seq_len)
        num_blocks = layout.shape[1]
    
        att_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for i in range(self.css):
            att_mask[i**2:(i+1)**2, :(i+1)**2] = True
        for row in range(num_blocks):
            for col in  range(num_blocks):
                layout[0, row, col] = \
                    att_mask[row*self.block:(row+1)*self.block, 
                        col*self.block:(col+1)*self.block].any()
        
        layout = self.check_and_propagate_first_head_layout(layout)
        return layout

class SparseInnerAttetion(nn.Module):
    def __init__(self, config, att_block):
        super().__init__() 
        self.att_block = att_block
        css = int(sqrt(config.block_size))
        sparsity_config = LformerSparsityConfig(num_heads=config.n_head,
                                                block=self.att_block,
                                                css=css)
        self.inner_att = SparseSelfAttention(sparsity_config, 
                                            attn_mask_mode='add', 
                                            max_seq_length=config.block_size)

    def forward(self, q, k, v, att_mask):
        T_q = q.shape[2]
        if T_q % self.att_block != 0: #pad to att_block size
            pad_len = (T_q//self.att_block + 1) * self.att_block - T_q
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            if att_mask is not None:
                att_mask = F.pad(att_mask,(0, pad_len, 0, pad_len))
        if att_mask is not None:
            att_mask = att_mask.masked_fill(att_mask == 0, float('-inf'))-1
        y = self.inner_att(q, k ,v, attn_mask=att_mask)[:, :, :T_q, :]
        return y

class LformerSparseAttention(Attention):
    def __init__(self, config):
        super().__init__(config)
        self.inner_att = SparseInnerAttetion(config, att_block=16)
        
    def forward(self, query, key, value, att_mask=None, *args, **kargs):
        B, C = query.shape[0], query.shape[2]
        T_q, T_k, T_v = query.shape[1], key.shape[1], value.shape[1]
        assert T_q == T_k and T_k == T_v # only support self attention
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.W_q(query).view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_q, hs)
        k = self.W_k(key).view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_k, hs)
        v = self.W_v(value).view(B, T_v, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_v, hs)

        y = self.inner_att(q, k ,v, att_mask=att_mask)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        y = self.resid_drop(self.proj(y))
        return y, None

class SparseLayer(SandwichLayer):
    def __init__(self, config):
        super().__init__(config)
        self.ln1 = LayerNorm(config.n_embd)
        self.ln1_2 = LayerNorm(config.n_embd)
        self.ln2 = LayerNorm(config.n_embd)
        self.ln2_2 = LayerNorm(config.n_embd)
        self.attn = LformerSparseAttention(config)
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
    def __init__(self, text_vbs, img_vbs, block_size, 
                 n_layer=12, n_head=8, n_embd=256, dim_cond=512, 
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., 
                 add_cross=False, full_head=False, PBrelax=False, checkpoint=0):
        super().__init__(text_vbs=text_vbs, img_vbs=img_vbs, block_size=block_size,
                       n_layer=n_layer, n_head=n_head, n_embd=n_embd, dim_cond=dim_cond,
                       embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                       add_cross=add_cross, full_head=full_head, PBrelax=PBrelax, checkpoint=checkpoint)

    def get_layers(self, config):
        return nn.ModuleList([SparseLayer(config) for _ in range(config.n_layer)])   

    def forward(self, input_ids, att_mask, pos_ids, embeddings=None, eh=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(input_ids) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            #embeddings = self.ctx_proj(embeddings)
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        t = token_embeddings.shape[1]

        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb(pos_ids)
        x = self.drop(token_embeddings + position_embeddings)
        if len(att_mask.shape) == 2:
            att_mask = att_mask[:, None, None, :]
        elif len(att_mask.shape) == 3:
            att_mask = att_mask[:, None, :, :] # [B, H, L, L]

        if self.add_cross and eh is not None:
            proj_eh = self.eh_proj(eh) 

        seg = self.config.checkpoint
        if self.add_cross and eh is not None:
            for _, layer in enumerate(self.layers[:seg]):
                x = checkpoint(layer, x, att_mask, proj_eh)
            for _, layer in enumerate(self.layers[seg:]):    
                x = layer(x, att_mask, proj_eh)
        else:
            x = self.layers[0](x, att_mask)
            for _, layer in enumerate(self.layers[:seg]):
                x = checkpoint(layer, x, att_mask)
            for _, layer in enumerate(self.layers[seg:]):
                x = layer(x, att_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits 

