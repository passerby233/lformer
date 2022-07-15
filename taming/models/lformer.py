import torch
import torch.nn.functional as F

from .base import GenModel

import numpy as np
torch.set_printoptions(threshold=np.inf)

class Lformer(GenModel):
    def __init__(self,
                 css,
                 transformer_config,
                 first_stage_config,
                 text_len=77,
                 ckpt_path=None,
                 ignore_keys=[],
                 pkeep=1.0
                 ):
        super().__init__(css,
                         transformer_config,
                         first_stage_config,
                         pkeep
                         )
        self.text_len = text_len
        self.pad_id = self.img_vbs + self.text_vbs
        self.register_buffer("pos_ids", torch.arange(self.block_size).unsqueeze(0))
        self.register_buffer("att_mask", self.get_att_mask())

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def get_att_mask(self, css=None):
        css = self.css if css is None else css
        seq_len = css**2

        img_mask = torch.eye(seq_len, seq_len) # [S,S]
        for i in range(css):
            img_mask[i**2:(i+1)**2, :(i+1)**2] = 1.0
        partial_mask_ones = torch.ones(seq_len, self.text_len) #[S,T]
        img_mask = torch.cat((partial_mask_ones, img_mask), dim=1) #[S,T+S]

        text_mask = torch.tril(torch.ones(self.text_len, self.text_len)) #[T,T]
        partial_mask_zeros = torch.zeros((self.text_len, seq_len)) #[T,S]
        text_mask = torch.cat((text_mask, partial_mask_zeros), dim=1) # [T,T+S]
        
        att_mask = torch.cat((text_mask, img_mask), dim=0).unsqueeze(0)
        return att_mask

    def to_L_order(self, img_idx, css=None):
        css = self.css if css is None else css
        i, j = 0, 0
        L_order = []
        while i < css and j < css:
            L_order.append(i*css + j)
            if j == 0:
                j = i+1
                i = 0
            elif i<j:
                i += 1
            else:
                j -= 1    
        L_order = torch.tensor(L_order, requires_grad=False)
        return img_idx[:, L_order]

    def to_rs_order(self, img_idx, css=None):
        css = self.css if css is None else css
        rs_order = []
        for i in range(css):
            for j in range(css):
                if i<=j:
                    rs_order.append(j**2 + i)
                else:
                    rs_order.append(i**2 + 2*i - j)
        rs_order = torch.tensor(rs_order, requires_grad=False)
        return img_idx[:, rs_order]

    def pad_Ltoken(self, Ltoken, css=None):
        css = self.css if css is None else css
        batch_size = Ltoken.shape[0]
        pad_idx = (torch.ones(batch_size, 1)*self.pad_id).to(Ltoken)
        padded = torch.empty((batch_size, 0), dtype=Ltoken.dtype, device=Ltoken.device)
        for t in range(self.css-1):
            padded = torch.cat((padded, pad_idx, Ltoken[:, t**2:(t+1)**2], pad_idx), dim=1)
        return padded

    @torch.no_grad()
    def get_input(self, batch):
        text_idx, Ltoken, img_idx =  batch['text_idx'], batch['Ltoken'], batch['img_idx']
        if self.pkeep < 1:
            img_idx = self._add_noise(img_idx, self.pkeep)
        return text_idx, Ltoken, img_idx

    def shared_step(self, batch, batch_idx):
        text_idx, Ltoken, _ = self.get_input(batch)
        logits = self(text_idx, Ltoken)[:, text_idx.shape[1]:, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Ltoken.reshape(-1))
        return loss

    def forward(self, text_idx, Ltoken):
        padded = self.pad_Ltoken(Ltoken)
        #text_ctx, text_hidden = self.text_encoder(text_idx)
        pad_idx = (torch.ones(Ltoken.shape[0], 1) * self.pad_id).to(Ltoken)
        input_idx = torch.cat((text_idx+self.img_vbs, pad_idx, padded), dim=1)
        logits = self.transformer(input_idx, self.att_mask, self.pos_ids)
        return logits

    @torch.no_grad()
    def sample(self, text_idx, top_k=None, top_p=None, temperature=1.0, greedy=False):
        assert not self.transformer.training
        batch_size= text_idx.shape[0]
        pad_idx = (torch.ones(batch_size, 1)*self.pad_id).to(text_idx)
        #text_ctx, text_hidden = self.text_encoder(text_idx)
        Ltoken = torch.empty((batch_size, 0), dtype=torch.long, device=text_idx.device)
        padded = torch.cat((text_idx+self.img_vbs, pad_idx), dim=1)
        for t in range(self.css):
            seq_len = padded.shape[1]
            pos_ids = self.pos_ids[:, :seq_len]
            att_mask = self.att_mask[:, :seq_len, :seq_len]
            logits = self.transformer(padded, att_mask, pos_ids)
            # cut off the logits of text part, we are generating image tokens
            logits = logits[:, :, :self.img_vbs] 
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -(2*t+1):, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            if greedy:
                probs, pred_ids = torch.topk(probs, k=1, dim=-1)
            elif top_k is not None:
                B, L, V = probs.shape
                pred_ids = probs.view(-1, V).multinomial(1)
                pred_ids = pred_ids.reshape(B, L, 1) #[B, L]
                probs = probs.gather(2, pred_ids) # [B, L ,1]]
            elif top_p is not None:
                B, L, V = probs.shape
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1) 
                sorted_indices_to_remove = cumulative_probs > top_p   #[B,L,V]
                sorted_indices_to_remove[:, :, 0] = False # to ensure at least one token
                sorted_probs[sorted_indices_to_remove] = 0
                sorted_idx_indice = sorted_probs.view(-1, V).multinomial(1)
                sorted_idx_indice = sorted_idx_indice.reshape(B, L, 1)
                pred_ids = sorted_idx.gather(2, sorted_idx_indice)
                probs = probs.gather(2, pred_ids)
            pred_ids, probs = pred_ids.squeeze(-1), probs.squeeze(-1)
            Ltoken = torch.cat((Ltoken, pred_ids), dim=1)
            if t < self.css-1: # No need to cat at last step
                padded = torch.cat((padded, pad_idx, pred_ids, pad_idx), dim=1)
        
        return self.to_rs_order(Ltoken)

    @torch.no_grad()
    def log_images(self, batch, temperature=1.0, top_k=None, top_p=0.9, **kwargs):
        log = dict()
        text_idx, _, img_idx = self.get_input(batch)
        # det sample
        index_sample = self.sample(text_idx, top_k, top_p, temperature)
        x_sample = self.decode_to_img(index_sample)
        log["Pred"] = x_sample

        # reconstruction
        x_rec = self.decode_to_img(img_idx)
        log["GT"] = x_rec
        text_list = self.textidx_to_text(text_idx)
        return log, text_list


class TrecLformer(Lformer):
    def __init__(self,
                 css,
                 transformer_config,
                 first_stage_config,
                 text_len=77,
                 ckpt_path=None,
                 ignore_keys=[],
                 pkeep=1.0
                 ):
        super().__init__(css,
                         transformer_config,
                         first_stage_config,
                         text_len,
                         ckpt_path,
                         ignore_keys,
                         pkeep)

    def shared_step(self, batch, batch_idx):
        text_idx, Ltoken, _ = self.get_input(batch)
        target = torch.cat((text_idx[:, 1:]+self.img_vbs, Ltoken), dim=1)

        logits_out = self(text_idx, Ltoken)
        text_logits = logits_out[:, :text_idx.shape[1]-1, :]
        image_logits = logits_out[:, text_idx.shape[1]:, :]
        logits = torch.cat((text_logits, image_logits), dim=1)
        
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def forward(self, text_idx, Ltoken):
        padded = self.pad_Ltoken(Ltoken)
        #text_ctx, text_hidden = self.text_encoder(text_idx)
        pad_idx = (torch.ones(Ltoken.shape[0], 1) * self.pad_id).to(Ltoken)
        input_idx = torch.cat((text_idx+self.img_vbs, pad_idx, padded), dim=1)
        logits = self.transformer(input_idx, self.att_mask, self.pos_ids)
        return logits
