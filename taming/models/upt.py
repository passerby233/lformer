import torch
import torch.nn.functional as F

from .base import GenModel

import numpy as np
torch.set_printoptions(threshold=np.inf)


class UPT(GenModel):
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
                         pkeep=pkeep
                         )

        att_mask = torch.ones((self.block_size, self.block_size))
        self.register_buffer('att_mask', att_mask.unsqueeze(0))
        self.register_buffer("pos_ids", torch.arange(self.block_size).unsqueeze(0))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def upsample(self, low_res):
        with torch.no_grad():
            x = self.decode_to_img(low_res)
            x = F.interpolate(x, scale_factor=2.0, mode="bicubic", align_corners=False)
            x = self.normalize(x)
            _, _, interp_idx = self.first_stage_model.encode(x)
            interp_idx = interp_idx.reshape(interp_idx.shape[0], -1)
        return interp_idx

    def get_input(self, batch):
        img16x, img32x = batch['imgidx_16x'], batch['imgidx_32x']
        return img16x, img32x

    def shared_step(self, batch, batch_idx):
        img16x, img32x = self.get_input(batch)
        interp_idx = self.upsample(img16x)
        logits = self(interp_idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), img32x.reshape(-1))
        return loss

    def forward(self, imgidx):
        logits = self.transformer(imgidx, self.att_mask, self.pos_ids)
        return logits

    def sample(self, imgidx, sample=True, top_k=100):
        logits = self(imgidx)
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
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
        return pred_ids #[B,L]

    @torch.no_grad()
    def log_images(self, batch, temperature=1.0, top_k=100, lr_interface=False, **kwargs):
        log = dict()
        img16x, img32x = self.get_input(batch)

        # reconstruction
        x_rec_16x = self.decode_to_img(img16x) # [C,H,W]
        log["GT_16x"] = x_rec_16x
        x_rec_32x = self.decode_to_img(img32x)
        log["GT_32x"] = x_rec_32x

        # interploate
        interp_idx = self.upsample(img16x)
        interp_img = self.decode_to_img(interp_idx)
        log["ITP_32x"] = interp_img

        # sample
        index_sample = self.sample(interp_idx, sample=True, top_k=top_k)
        x_sample = self.decode_to_img(index_sample)
        log["Pred"] = x_sample

        return log

