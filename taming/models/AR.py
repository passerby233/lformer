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

    def process_logits(self, logits, top_k=None, top_p=0.9, temperature=1.0, greedy=False,):
        # cut off the logits of text part, we are generating image tokens
        logits = logits[:, :, :self.img_vbs]  
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1) #[B, V]
        # sample from the distribution or take the most likely
        if greedy:
            _, ix = torch.topk(probs, k=1, dim=-1)
        elif top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p   #[B,V]
            sorted_indices_to_remove[:, 0] = False # to ensure at least one token
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_idx_indice = sorted_probs.multinomial(1)
            ix = sorted_idx.gather(1, sorted_idx_indice)
        elif top_k is not None:
            ix = probs.multinomial(1)
        return ix

    @torch.no_grad()
    def sample(self, text_idx, 
               top_k=None, top_p=0.9, temperature=1.0, greedy=False, 
               half=None, use_cache=True):
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        steps = self.css**2//2 if half is not None else self.css**2
        x = torch.cat((text_idx+self.img_vbs, half), dim=1) if half is not None else text_idx+self.img_vbs

        # Prepare past if use cache
        if use_cache:
            config = self.transformer.config
            past_shape = [config.n_layer, 2, text_idx.shape[0], \
                config.n_head, config.block_size, config.n_embd//config.n_head]
            past = torch.empty(past_shape, dtype=self.dtype, device=text_idx.device)

        for k in range(steps):
            assert x.size(1) <= block_size # make sure model can see conditioning
            seq_len = x.shape[1]
            if use_cache:
                if k == 0:
                    pos_ids = self.pos_ids[:, :seq_len]
                    logits, past = self.transformer.forward_with_past(
                            x, pos_ids, past, past_length=0)
                else:
                    pos_ids = self.pos_ids[:, seq_len-1:seq_len]
                    logits, past = self.transformer.forward_with_past(
                            x[:, -1:], pos_ids, past, past_length=seq_len-1)
            else:
                att_mask = self.att_mask[:, :seq_len, :seq_len]
                pos_ids = self.pos_ids[:, :seq_len]
                logits = self.transformer(x, att_mask, pos_ids)

            ix = self.process_logits(logits, top_k, top_p, temperature, greedy) 
            x = torch.cat((x, ix), dim=1) # append to the sequence and continue

        # cut off conditioning
        x = x[:, text_idx.shape[1]:]
        return x

    @torch.no_grad()
    def log_images(self, batch, top_k=None, top_p=0.9, temperature=1.0,  **kwargs):
        log = dict()
        text_idx, img_idx  = self.get_input(batch)

        # half sample
        half_sample = self.sample(text_idx, top_k=top_k, top_p=top_p, temperature=temperature,
                                    half=img_idx[:,:img_idx.shape[1]//2])
        x_half = self.decode_to_img(half_sample)
        log["half"] = x_half

        # det sample
        index_sample = self.sample(text_idx, top_k=top_k, top_p=top_p, temperature=temperature,)
        x_sample = self.decode_to_img(index_sample)
        log["Pred"] = x_sample

        # reconstruction
        x_rec = self.decode_to_img(img_idx)
        log["GT"] = x_rec
        text_list = self.textidx_to_text(text_idx)
        return log, text_list

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

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

    def get_input(self, batch):
        text_idx, images =  batch['text_idx'], batch['image']
        _, _, img_idx= self.first_stage_model.encode(images.to(self.dtype))
        img_idx = img_idx.view(images.shape[0], -1)
        if self.pkeep < 1 and self.training:
            img_idx = self._add_noise(img_idx, self.pkeep)
        return text_idx, img_idx
