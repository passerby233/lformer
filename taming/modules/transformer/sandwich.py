import logging

import torch
import torch.nn as nn
from .gpt import GPTConfig, Attention, LayerNorm

logger = logging.getLogger(__name__)

class SandwichBlock(nn.Module):
    """ an unassuming Transformer block """
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

    def forward(self, x, att_mask=None, eh=None, eh_mask=None, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        x_norm = self.ln1(x)
        attn, present = self.attn(x_norm, x_norm, x_norm, att_mask, layer_past, return_present)
        x = x + (self.ln1_2(attn))

        if hasattr(self, "cross_attn") and eh is not None:
            cross_attn, cross_present = self.cross_attn(self.ln_cross(x), eh, eh, eh_mask, return_present)
            x = x + self.ln_cross_2(cross_attn)

        x = x + self.ln2_2(self.mlp(self.ln2(x)))

        if layer_past is not None or return_present:
            return x, present
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, text_vbs, img_vbs, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., dim_cond=512, 
                 add_cross=False, full_head=False, PBrelax=False):
        super().__init__()
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
        self.blocks = nn.ModuleList([SandwichBlock(config) for _ in range(config.n_layer)])
        # decoder head
        head_size = text_vbs + img_vbs if full_head else img_vbs
        self.ln_f = LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, head_size, bias=False)   
        self.apply(self._init_weights)

        self.block_size = config.block_size
        self.add_cross = add_cross
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

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
        for _, block in enumerate(self.blocks):
            if self.add_cross and eh is not None:
                x  = block(x, att_mask, proj_eh)
            else:
                x  = block(x, att_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None):
        # inference only
        assert not self.training
        token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
        if embeddings is not None:              # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head
