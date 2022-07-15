import torch
import torch.nn.functional as F

from .base import GenModel

import numpy as np
torch.set_printoptions(threshold=np.inf)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class ARModel(GenModel):
    def __init__(self,
                 css,
                 transformer_config,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 pkeep=1.0
                 ):
        super().__init__(css,
                         transformer_config,
                         first_stage_config,
                         pkeep
                         )

        self.register_buffer("pos_ids", torch.arange(self.block_size).unsqueeze(0))
        self.register_buffer("att_mask", self.get_att_mask())

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def get_att_mask(self, css=None): 
        css = self.css if css is None else css
        seq_len = css**2
        """
        img_mask = torch.tril(torch.ones((seq_len, seq_len)))
        partial_mask_zeros = torch.zeros((77, seq_len)).to(img_mask)
        partial_mask_ones = torch.ones(77+seq_len, 77).to(img_mask) 
        att_mask = torch.cat((partial_mask_zeros, img_mask), dim=0) 
        att_mask = torch.cat((partial_mask_ones, att_mask), dim=1) 
        """
        att_mask = torch.tril(torch.ones(self.block_size, self.block_size))
        return att_mask.unsqueeze(0)

    def get_input(self, batch):
        text_idx, img_idx =  batch['text_idx'], batch['img_idx']
        return text_idx, img_idx

    def shared_step(self, batch, batch_idx):
        text_idx, img_idx  = self.get_input(batch)
        logits = self(text_idx, img_idx)[:, text_idx.shape[1]-1:, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), img_idx.reshape(-1))
        return loss

    def forward(self, text_idx, img_idx):
        input_idx = torch.cat((text_idx+self.img_vbs, img_idx), dim=1)[:, :-1]
        seq_len = input_idx.shape[1]
        att_mask = self.att_mask[:, :seq_len, :seq_len] # [B,L,L]
        pos_ids = self.pos_ids[:, :seq_len] # [B,L]
        logits = self.transformer(input_idx, att_mask, pos_ids)
        return logits

    @torch.no_grad()
    def sample(self, text_idx, temperature=1.0, sample=False, top_k=None, half=None):
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        steps = self.css**2//2 if half is not None else self.css**2
        x = torch.cat((text_idx+self.img_vbs, half), dim=1) if half is not None else text_idx+self.img_vbs
        for k in range(steps):
            assert x.size(1) <= block_size # make sure model can see conditioning
            seq_len = x.shape[1]
            att_mask = self.att_mask[:, :seq_len, :seq_len]
            pos_ids = self.pos_ids[:, :seq_len]
            logits = self.transformer(x, att_mask, pos_ids)
            # cut off the logits of text part, we are generating image tokens
            logits = logits[:, :, :self.img_vbs]  
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        # cut off conditioning
        x = x[:, text_idx.shape[1]:]
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=1.0, top_k=100, lr_interface=False, **kwargs):
        log = dict()
        text_idx, img_idx  = self.get_input(batch)
        
        # half sample
        half_sample = self.sample(text_idx, sample=True, top_k=top_k, 
            half=img_idx[:,:img_idx.shape[1]//2])
        x_half = self.decode_to_img(half_sample)
        log["half"] = x_half
        
        # det sample
        index_sample = self.sample(text_idx, sample=True, top_k=top_k)
        x_sample = self.decode_to_img(index_sample)
        log["Pred"] = x_sample

        # reconstruction
        x_rec = self.decode_to_img(img_idx)
        log["GT"] = x_rec
        text_list = self.textidx_to_text(text_idx)
        return log, text_list

class TrecARModel(ARModel):
    # Text Reconstruction Version
    def __init__(self,
                 css,
                 transformer_config,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 pkeep=1.0
                 ):
        super().__init__(css,
                         transformer_config,
                         first_stage_config,
                         ckpt_path,
                         ignore_keys,
                         pkeep)

    def shared_step(self, batch, batch_idx):
        text_idx, img_idx  = self.get_input(batch)
        logits = self(text_idx, img_idx)
        target_idx = torch.cat((text_idx+self.img_vbs, img_idx), dim=1)[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_idx.reshape(-1))
        return loss

class TrecARModelOnline(TrecARModel):
    def __init__(self,
                 css,
                 transformer_config,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 pkeep=1.0
                 ):
            super().__init__(css,
                         transformer_config,
                         first_stage_config,
                         ckpt_path,
                         ignore_keys,
                         pkeep)

    def get_input(self, batch):
        text_idx, images =  batch['text_idx'], batch['image']
        _, _, img_idx= self.first_stage_model.encode(images.to(self.dtype))
        img_idx = img_idx.view(images.shape[0], -1)
        if self.pkeep < 1 and self.training:
            img_idx = self._add_noise(img_idx, self.pkeep)
        return text_idx, img_idx
