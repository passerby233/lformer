import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class LayerNorm(nn.LayerNorm):
    # scale layernorm from cogview
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return super().forward(x / (x.abs().max().detach()/8))

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.W_q= nn.Linear(config.n_embd, config.n_embd)
        self.W_k = nn.Linear(config.n_embd, config.n_embd)
        self.W_v = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.PBrelax = config.PBrelax
        self.alpha = 32 # for PBrelax

    def forward(self, query, key, value, att_mask=None, layer_past=None, return_present=False):
        B, C = query.shape[0], query.shape[2]
        T_q, T_k, T_v = query.shape[1], key.shape[1], value.shape[1]
        assert T_k == T_v

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.W_q(query).view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_q, hs)
        k = self.W_k(key).view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_k, hs)
        v = self.W_v(value).view(B, T_v, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_v, hs)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # Self-attend: (B, nh, T_q, hs) x (B, nh, hs, T_k) -> (B, nh, T_q, T_k)
        if self.PBrelax:
            att = q * (1.0/ self.alpha / math.sqrt(k.size(-1))) @ k.transpose(-2, -1)
            att = (att - att.abs().max().detach()) * self.alpha
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if att_mask is not None and layer_past is None:
            att = att.masked_fill(att_mask[:,:,:T_q,:T_k] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T_q, T_k) x (B, nh, T_v, hs) -> (B, nh, T_q, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        # return y and cache
        present = None
        if layer_past is not None or return_present:
            present = (k[..., -T_q:,:], v[..., -T_q:,:])
        return y, present
        #return y, present   # TODO: check that this does not break anything

class SandwichLayer(nn.Module):
    """ an unassuming Transformer Layer """
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.ln1_2 = LayerNorm(config.n_embd)
        self.ln2 = LayerNorm(config.n_embd)
        self.ln2_2 = LayerNorm(config.n_embd)
        self.attn = Attention(config)
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

    def forward(self, x, att_mask=None, eh=None, eh_mask=None, 
                layer_past=None,  return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        x_norm = self.ln1(x)
        attn, present = self.attn(x_norm, x_norm, x_norm, att_mask, 
                                  layer_past, return_present)
        x = x + (self.ln1_2(attn))

        if hasattr(self, "cross_attn") and eh is not None:
            x_norm_cross = self.ln_cross(x)
            cross_attn, _ = self.cross_attn(x_norm_cross, eh, eh, eh_mask)
            x = x + self.ln_cross_2(cross_attn)

        x = x + self.ln2_2(self.mlp(self.ln2(x)))

        if return_present:
            return x, present
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, text_vbs, img_vbs, block_size, 
                 n_layer=12, n_head=8, n_embd=256, dim_cond=512, 
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., 
                 add_cross=False, full_head=False, PBrelax=False, checkpoint=0):
        super().__init__()
        config = GPTConfig(vocab_size=text_vbs+img_vbs, block_size=block_size,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           add_cross=add_cross, full_head=full_head, PBrelax=PBrelax, checkpoint=checkpoint)
        # input embedding stem
        if add_cross:
            self.eh_proj = nn.Linear(dim_cond, n_embd)
        #self.ctx_proj = nn.Linear(dim_cond, n_embd)
        self.tok_emb = nn.Embedding(config.vocab_size+1, config.n_embd) # 1 for pad
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.layers = self.get_layers(config)
        # decoder head
        head_size = text_vbs + img_vbs if full_head else img_vbs
        self.ln_f = LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, head_size, bias=False)   
        self.apply(self._init_weights)

        self.block_size = config.block_size
        self.add_cross = add_cross
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_layers(self, config):
        return nn.ModuleList([SandwichLayer(config) for _ in range(config.n_layer)])

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
            for _, layer in enumerate(self.layers[:seg]):
                x = checkpoint(layer, x, att_mask)
            for _, layer in enumerate(self.layers[seg:]):
                x = layer(x, att_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def forward_with_past(self, input_ids, pos_ids, past=None, past_length=0, embeddings=None, eh=None):
        token_embeddings = self.tok_emb(input_ids) # each index maps to a (learnable) vector
        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        position_embeddings = self.pos_emb(pos_ids)
        x = self.drop(token_embeddings + position_embeddings)

        if self.add_cross and eh is not None:
            proj_eh = self.eh_proj(eh)

        input_length = x.shape[1]
        for i, layer in enumerate(self.layers):
            output = layer(x, return_present=True,
                            eh=proj_eh if eh is not None else None,
                            layer_past=past[i, ..., :past_length, :] if past is not None else None)
            x = output[0]
            past[i, 0, ..., past_length:past_length+input_length, :] = output[1][0] 
            past[i, 1, ..., past_length:past_length+input_length, :] = output[1][1]

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, past