from email import contentmanager
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import GenModel, LformerBase

import numpy as np
torch.set_printoptions(threshold=np.inf)

class LformerLatent(GenModel, LformerBase):
    def __init__(self,
                 css,
                 w_layer,
                 classes,
                 transformer_config,
                 first_stage_config,
                 pkeep=1.0,
                 ckpt_path=None,
                 ignore_keys=[],
                 ):
        GenModel.__init__(self,
                            css,
                            transformer_config,
                            first_stage_config,
                            pkeep) 
        hs_cond = transformer_config.params.dim_cond
        hs_trans = transformer_config.params.n_embd

        self.pad_id = self.img_vbs
        self.label_embedding = nn.Embedding(classes, hs_cond)
        self.register_buffer("pos_ids", torch.arange(self.block_size).unsqueeze(0))
        self.register_buffer("att_mask", self.get_att_mask())

        self.linear_mu_c = nn.Linear(hs_cond, hs_cond)
        self.linear_mu_z = nn.Linear(hs_cond, hs_cond)
        self.linear_log_var = nn.Linear(hs_cond, hs_cond)
        self.linear_ctx = nn.Linear(hs_cond * 2, hs_trans)
        layers = []
        for _ in range(w_layer):
            layers.extend([nn.Linear(hs_cond, hs_cond), nn.LeakyReLU(inplace=True)])
        self.latent_to_w = nn.Sequential(*layers)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def get_input(self, batch):
        label, images =  batch['label'], batch['image']
        img_feature = batch['feature']
        _, _, img_idx= self.first_stage_model.encode(images.to(self.dtype))
        img_idx = img_idx.view(images.shape[0], -1)
        if self.pkeep < 1 and self.training:
            img_idx = self._add_noise(img_idx, self.pkeep)
        Ltoken = self.to_L_order(img_idx)
        return label, Ltoken, img_idx, img_feature

    def shared_step(self, batch, batch_idx):
        label, Ltoken, _, img_feature = self.get_input(batch)
        label_emb = self.label_embedding(label)
        padded = self.pad_Ltoken(Ltoken)
        logits, KLD = self(label_emb, padded, img_feature)
        CE = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Ltoken.reshape(-1))
        loss_weight = self.current_epoch / 100 
        loss = CE + KLD * loss_weight
        return loss, KLD

    def get_context(self, label_emb, img_feature=None):
        KLD = None
        mu_c = self.linear_mu_c(label_emb)
        if img_feature is not None:
            mu_z = self.linear_mu_z(img_feature)
            log_var = self.linear_log_var(img_feature)
            z = self.reparameterize(mu_z, log_var)
            KLD = -0.5 * torch.sum(1 + log_var - (mu_c - mu_z).pow(2) - log_var.exp(), dim=-1).mean()
        else:
            z = self.reparameterize(mu_c, torch.zeros_like(mu_c))
        w = self.latent_to_w(z)
        context = self.linear_ctx(torch.cat((label_emb, w), dim=1)).unsqueeze(1) # [B,1,H]
        return context, KLD

    def forward(self, label_emb, padded, img_feature=None):
        seq_len = padded.shape[1] + 1
        att_mask = self.att_mask[:, :seq_len, :seq_len]
        pos_ids = self.pos_ids[:, :seq_len]
        context, KLD = self.get_context(label_emb, img_feature)
        logits = self.transformer(padded, att_mask, pos_ids, embeddings=context)
        return logits, KLD

    @torch.no_grad()
    def sample(self, label,
               top_k=None, top_p=0.8, temperature=1.0, greedy=False, 
               return_feature=False, use_cache=False):
        # sample a token map from given text tokens
        assert not self.transformer.training
        batch_size = label.shape[0]
        pad_idx = (torch.ones(batch_size, 1)*self.pad_id).to(label)
        label_emb = self.label_embedding(label)
        context, _ = self.get_context(label_emb)
        Ltoken = torch.empty((batch_size, 0), dtype=torch.long, device=label.device)
        padded = torch.empty((batch_size, 0), dtype=torch.long, device=label.device)

        # Prepare past if use cache
        if use_cache:
            config = self.transformer.config
            past_shape = [config.n_layer, 2, batch_size, \
                config.n_head, config.block_size, config.n_embd//config.n_head]
            past = torch.empty(past_shape, dtype=label.dtype, device=label.device)

        for t in range(self.css):
            if use_cache:
                pred_ids, past = self.cached_single_step(
                                    padded, t, past, context,
                                    top_k, top_p, temperature, greedy)
                Ltoken = torch.cat((Ltoken, pred_ids), dim=1)
                if t < self.css-1: # No need to cat at last step
                    padded = torch.cat((pad_idx, pred_ids, pad_idx), dim=1)
            else:
                pred_ids, _ = self.infer_single_step(
                                    label_emb, padded, t,
                                    top_k, top_p, temperature, greedy)
                Ltoken = torch.cat((Ltoken, pred_ids), dim=1)
                if t < self.css-1: # No need to cat at last step
                    padded = torch.cat((padded, pad_idx, pred_ids, pad_idx), dim=1)

        return self.to_rs_order(Ltoken) 

    @torch.no_grad()
    def log_images(self, batch, top_k=512, top_p=0.8, temperature=1.0,  **kwargs):
        log = dict()
        label, _, img_idx, _ = self.get_input(batch)
        # det sample
        index_sample = self.sample(label, top_k, top_p, temperature)
        x_sample = self.decode_to_img(index_sample)
        log["Pred"] = x_sample

        # reconstruction
        x_rec = self.decode_to_img(img_idx)
        log["GT"] = x_rec
        text_list = [str(lb.item()) for lb in label]
        return log, text_list

    @torch.no_grad()
    def cached_single_step(self, padded, t, past=None, context=None,
                          top_k=None, top_p=0.9, temperature=1.0, greedy=False):
        assert past is not None or context is not None
        pos_ids = self.pos_ids[:, t**2:(t+1)**2]
        logits, past = self.transformer.forward_with_past(
                            padded, pos_ids, past, past_length=t**2,
                            embeddings=context if t==0 else None)
        assert logits.shape[1] == 2*t+1
        pred_ids, _ =  self.process_logits(logits, t, top_k, top_p, temperature, greedy)
        return pred_ids, past

    @torch.no_grad()
    def infer_single_step(self, label_emb, padded,  t, 
                          top_k=None, top_p=0.9, temperature=1.0, greedy=False):
        logits, _ = self(label_emb, padded)
        return self.process_logits(logits, t, top_k, top_p, temperature, greedy)

    def configure_optimizers(self):
        extra_modules = [self.label_embedding,
                         self.linear_mu_c, self.linear_mu_z, 
                         self.linear_log_var, self.linear_ctx, self.latent_to_w]
        return self._configure_optimizers(extra_modules)

    def training_step(self, batch, batch_idx):
        loss, KLD = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/KLD", KLD, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss