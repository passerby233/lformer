import torch
import torch.nn.functional as F
import math
from .base import GenModel

import numpy as np
torch.set_printoptions(threshold=np.inf)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Lformer(GenModel):
    def __init__(self,
                 css,
                 transformer_config,
                 #text_encoder_config,
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

        text_pos_ids = torch.arange(77)
        _pos_ids = torch.arange(css**2) + 77
        pos_ids = torch.cat((text_pos_ids, _pos_ids, _pos_ids)).unsqueeze(0)
        self.register_buffer('pos_ids', pos_ids)
        self.register_buffer("train_att_mask", self.get_train_att_mask())

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def get_train_att_mask(self, css=None):
        css = self.css if css is None else css
        seq_len = css**2
        text_mask = torch.cat((torch.tril(torch.ones(77, 77)), torch.zeros(77, 2*seq_len)), dim=1)
        img_mask = torch.eye(2*seq_len, 2*seq_len)
        for i in range(css):
            img_mask[i**2:(i+1)**2, :(i+1)**2] = 1.0
            img_mask[seq_len+i**2:seq_len+(i+1)**2, :i**2] = 1.0
        partial_mask = torch.cat((torch.ones(2*seq_len, 77), img_mask), dim=1)
        train_att_mask = torch.cat((text_mask, partial_mask), dim=0)
        return train_att_mask.unsqueeze(0)

    def get_infer_att_mask(self, input_idx):
        img_idx = input_idx[:, 77:]
        seq_len = img_idx.shape[1]
        cur_ss = int(math.sqrt(seq_len))-1
        assert seq_len % (cur_ss+1) == 0
        text_mask = torch.cat((torch.tril(torch.ones(77, 77)), torch.zeros(77, seq_len)), dim=1)
        img_mask = torch.zeros(seq_len, seq_len)
        for i in range(cur_ss+1):
            img_mask[i**2:(i+1)**2, :(i+1)**2] = 1.0
        img_mask[cur_ss**2:(cur_ss+1)**2, cur_ss**2:(cur_ss+1)**2] = 0.0
        for i in range(cur_ss**2, (cur_ss+1)**2):
            img_mask[i, i] =1.0
        partial_mask = torch.cat((torch.ones(seq_len, 77), img_mask), dim=1)
        infer_att_mask = torch.cat((text_mask, partial_mask), dim=0)
        return infer_att_mask.unsqueeze(0).to(input_idx.device)

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

    def get_input(self, batch):
        text_idx, Ltoken, img_idx =  batch['text_idx'], batch['Ltoken'], batch['img_idx']
        if self.pkeep < 1:
            img_idx = self._add_noise(img_idx, self.pkeep)
        return text_idx, Ltoken, img_idx

    def shared_step(self, batch, batch_idx):
        text_idx, Ltoken, _ = self.get_input(batch)
        logits = self(text_idx, Ltoken)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Ltoken.reshape(-1))
        return loss

    def forward(self, text_idx, Ltoken):
        input_idx = torch.cat((text_idx+self.img_vbs, Ltoken, 
            torch.ones_like(Ltoken)*self.pad_id), dim=1)
        logits = self.transformer(input_idx, self.train_att_mask, self.pos_ids)
        img_logits = logits[:, text_idx.shape[1]+Ltoken.shape[1]:]
        return img_logits

    @torch.no_grad()
    def sample(self, text_idx, temperature=1.0, sample=False, top_k=None):
        assert not self.transformer.training
        x = text_idx
        for t in range(self.css):
            pad_idx = (torch.ones((x.shape[0], 2*t+1))*self.pad_id).to(x)
            x = torch.cat((x, pad_idx), dim=1)
            pos_ids = self.pos_ids[:, :x.shape[1]]
            infer_att_mask = self.get_infer_att_mask(x)
            logits = self.transformer(x, infer_att_mask, pos_ids)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -(2*t+1):, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1], shape[2])
                pred_ids = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0], shape[1], shape[2])
                pred_ids = pred_ids.reshape(shape[0], shape[1], 1)
                probs = probs.gather(2, pred_ids)
            else:
                probs, pred_ids = torch.topk(probs, k=1, dim=-1)
            pred_ids, probs = pred_ids.squeeze(-1), probs.squeeze(-1)
            x[:, -(2*t+1):] = pred_ids
        # cut off conditioning
        x = x[:, text_idx.shape[1]:]
        x = self.to_rs_order(x)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=1.0, top_k=100, lr_interface=False, **kwargs):
        log = dict()
        text_idx, _, img_idx = self.get_input(batch)
        # det sample
        index_sample = self.sample(text_idx, sample=True, top_k=top_k)
        x_sample = self.decode_to_img(index_sample)

        # reconstruction
        x_rec = self.decode_to_img(img_idx)
        tex_img = self.textidx_to_img(text_idx)

        #log["inputs"] = batch['img_raw']
        log["GT"] = x_rec
        log['Text'] = tex_img
        log["Pred"] = x_sample
        return log

