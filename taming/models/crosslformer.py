from contextvars import Context
from faulthandler import disable
from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from util import instantiate_from_config
from .base import GenModel, disabled_train
torch.set_printoptions(threshold=np.inf)

class Lformer(GenModel):
    def __init__(self,
                 css,
                 transformer_config,
                 text_encoder_config,
                 first_stage_config,
                 pkeep=1.0,
                 ):
        super().__init__(css,
                         transformer_config,
                         first_stage_config,
                         pkeep) 

        self.pad_id = self.img_vbs
        self.text_encoder = instantiate_from_config(text_encoder_config)
        if text_encoder_config.params.freeze:
            self.text_encoder.freeze()
            self.text_encoder.train = disabled_train

    def get_att_mask(self, css=None):
        css = self.css if css is None else css
        seq_len = css**2
        img_mask = torch.eye(seq_len, seq_len) # [S,S]
        for i in range(css):
            img_mask[i**2:(i+1)**2, :(i+1)**2] = 1.0
        return img_mask.unsqueeze(0)

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
        assert css > 0
        batch_size = Ltoken.shape[0]
        pad_idx = (torch.ones(batch_size, 1)*self.pad_id).to(Ltoken)
        padded = torch.empty((batch_size, 0), dtype=Ltoken.dtype, device=Ltoken.device)
        for t in range(css-1):
            padded = torch.cat((padded, pad_idx, Ltoken[:, t**2:(t+1)**2], pad_idx), dim=1)
        return padded

    @torch.no_grad()
    def get_input(self, batch):
        text_idx, images =  batch['text_idx'], batch['image']
        _, _, img_idx= self.first_stage_model.encode(images.to(self.dtype))
        img_idx = img_idx.view(images.shape[0], -1)
        if self.pkeep < 1 and self.training:
            img_idx = self._add_noise(img_idx, self.pkeep)
        Ltoken = self.to_L_order(img_idx)
        return text_idx, Ltoken, img_idx, None

    def shared_step(self, batch, batch_idx):
        text_idx, Ltoken, _, _ = self.get_input(batch)
        with torch.no_grad():
            text_feature, text_hidden = self.text_encoder(text_idx)
        padded = self.pad_Ltoken(Ltoken)
        logits, _ = self(text_feature, text_hidden, padded)
        CE = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Ltoken.reshape(-1))
        return CE, None

    def process_logits(self, logits, t, top_k=None, top_p=None, temperature=1.0, greedy=False):
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
        elif top_k is not None:
            B, L, V = probs.shape
            pred_ids = probs.view(-1, V).multinomial(1)
            pred_ids = pred_ids.reshape(B, L, 1) #[B, L]
            probs = probs.gather(2, pred_ids) # [B, L ,1]]
        pred_ids, probs = pred_ids.squeeze(-1), probs.squeeze(-1)
        return pred_ids, probs

    @torch.no_grad()
    def infer_single_step(self, text_feature, text_hidden, padded,  t, 
                          top_k=None, top_p=None, temperature=1.0, greedy=False):
        logits, _ = self(text_feature, text_hidden, padded)
        return self.process_logits(logits, t, top_k, top_p, temperature, greedy)
        
    @torch.no_grad()
    def cached_single_step(self, text_hidden, padded, t, past=None, context=None,
                          top_k=None, top_p=0.9, temperature=1.0, greedy=False):
        assert past is not None or context is not None
        pos_ids = self.pos_ids[:, t**2:(t+1)**2]
        logits, past = self.transformer.forward_with_past(
                            padded, pos_ids, past, past_length=t**2, eh=text_hidden, 
                            embeddings=context if t==0 else None)
        assert logits.shape[1] == 2*t+1
        pred_ids, _ =  self.process_logits(logits, t, top_k, top_p, temperature, greedy)
        return pred_ids, past

    @torch.no_grad()
    def sample(self, text_idx,
               top_k=None, top_p=0.9, temperature=1.0, greedy=False, 
               return_feature=False, use_cache=False):
        # sample a token map from given text tokens
        assert not self.transformer.training
        batch_size = text_idx.shape[0]
        pad_idx = (torch.ones(batch_size, 1)*self.pad_id).to(text_idx)
        text_feature, text_hidden = self.text_encoder(text_idx)
        context = self.get_context(text_feature)
        Ltoken = torch.empty((batch_size, 0), dtype=torch.long, device=text_idx.device)
        padded = torch.empty((batch_size, 0), dtype=torch.long, device=text_idx.device)

        # Prepare past if use cache
        if use_cache:
            config = self.transformer.config
            past_shape = [config.n_layer, 2, batch_size, \
                config.n_head, config.block_size, config.n_embd//config.n_head]
            past = torch.empty(past_shape, dtype=text_hidden.dtype, device=text_hidden.device)

        for t in range(self.css):
            if use_cache:
                pred_ids, past = self.cached_single_step(
                                    text_hidden, padded, t, past, context,
                                    top_k, top_p, temperature, greedy)
                Ltoken = torch.cat((Ltoken, pred_ids), dim=1)
                if t < self.css-1: # No need to cat at last step
                    padded = torch.cat((pad_idx, pred_ids, pad_idx), dim=1)
            else:
                pred_ids, _ = self.infer_single_step(
                                    text_feature, text_hidden, padded, t,
                                    top_k, top_p, temperature, greedy)
                Ltoken = torch.cat((Ltoken, pred_ids), dim=1)
                if t < self.css-1: # No need to cat at last step
                    padded = torch.cat((padded, pad_idx, pred_ids, pad_idx), dim=1)

        if return_feature:
            return self.to_rs_order(Ltoken), text_feature
        return self.to_rs_order(Ltoken)  

    @torch.no_grad()
    def log_images(self, batch, top_k=None, top_p=0.9, temperature=1.0,  **kwargs):
        log = dict()
        text_idx, _, img_idx, _ = self.get_input(batch)
        # det sample
        index_sample = self.sample(text_idx, top_k, top_p, temperature)
        x_sample = self.decode_to_img(index_sample)
        log["Pred"] = x_sample

        # reconstruction
        x_rec = self.decode_to_img(img_idx)
        log["GT"] = x_rec
        text_list = self.textidx_to_text(text_idx)
        return log, text_list

    def _configure_optimizers(self, extra_modules):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        # add extra params for vae to optimizer                                            
        extra_param = []
        for module in extra_modules:
            extra_param.extend(list(module.parameters())) 
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.04},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            {"params": extra_param, "weight_decay": 0.0}
        ]

        optimizer = self.get_optimizer(optim_groups)
        if self.scheduler_params is not None:
            scheduler = self.get_scheduler(optimizer, self.scheduler_params)
            return [optimizer], [scheduler]
        else:
            return optimizer
            
    def configure_optimizers(self):
        extra_modules = [self.linear_ctx]
        return self._configure_optimizers(extra_modules)

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    ### utilities for editing
    def get_css(self, token_idx):
        # return the css of a token [B,L]
        return int(sqrt(token_idx.shape[-1]))

    def crop_token(self, rs_token, crop_size):
        # input a rs_order token, return a cropped Ltoken
        css = self.get_css(rs_token)
        assert crop_size <= css
        L_token = self.to_L_order(rs_token, css)[:, :crop_size**2]
        return L_token
    
    def decode_cropped_to_img(self, cropped):
        # pad cropped Ltoken to full Ltoken and decode to image
        css = self.get_css(cropped)
        padded_cropped_L = F.pad(cropped, (0, self.css**2-css**2))
        padded_cropped_rs = self.to_rs_order(padded_cropped_L)
        padded_cropped_img = self.decode_to_img(padded_cropped_rs)
        return padded_cropped_img

    def repaint(self, text_idx, Ltoken, target_css, top_k=None, top_p=0.9, temperature=1.0):
        # infer from Ltoken of [cur_css x cur_css] to [target_css, target_css]
        # with condiction text_idx
        cur_css = self.get_css(Ltoken)
        padded = self.pad_Ltoken(Ltoken, cur_css+1)
        text_feature, text_hidden = self.text_encoder(text_idx)
        pad_idx = (torch.ones(Ltoken.shape[0], 1)*self.pad_id).to(text_idx)
        for t in range(cur_css, target_css):
            pred_ids, _ = self.infer_single_step(text_feature, text_hidden, padded, t, 
                                                top_k=top_k, top_p=top_p, temperature=temperature)
            Ltoken = torch.cat((Ltoken, pred_ids), dim=1)
            if t < target_css: # No need to cat at last step
                padded = torch.cat((padded, pad_idx, pred_ids, pad_idx), dim=1)
        return Ltoken

    def inpaint(self, text_idx, img_idx, x1, y1, x2, y2,
                top_k=None, top_p=0.9, temperature=1.0):
        assert x1 < x2 and y1 < y2
        # prepare inpaint mask
        css = self.get_css(img_idx)
        Ltoken = self.to_L_order(img_idx, css)
        mask = torch.zeros(css, css, dtype=Ltoken.dtype, device=Ltoken.device)
        mask[x1:x2, y1:y2] = 1
        L_mask = self.to_L_order(mask.reshape(1, -1))
        css_in, css_out = max(x1, y1), max(x2, y2)

        # crop the max Ltoken
        new_Ltoken =  Ltoken[:, :css_in**2]
        if css_in == 0:
            padded = torch.empty((text_idx.shape[0], 0), dtype=torch.long, device=text_idx.device)
        else:
            padded = self.pad_Ltoken(new_Ltoken, css_in+1)

        pad_idx = (torch.ones(Ltoken.shape[0], 1)*self.pad_id).to(img_idx)
        text_feature, text_hidden = self.text_encoder(text_idx)
        for t in range(css_in, css_out):
            pred_ids, _ = self.infer_single_step(text_feature, text_hidden, padded, t, 
                                                top_k=top_k, top_p=top_p, temperature=temperature)
            pred_mask = L_mask[:, t**2:(t+1)**2]
            L_raw = Ltoken[:, t**2:(t+1)**2]
            pred_ids = pred_ids * pred_mask + L_raw * (1 - pred_mask)
            new_Ltoken = torch.cat((new_Ltoken, pred_ids), dim=1)
            if t < css_out: # No need to cat at last step
                padded = torch.cat((padded, pad_idx, pred_ids, pad_idx), dim=1)
        # cat the residual
        new_Ltoken = torch.cat((new_Ltoken, Ltoken[:, css_out**2:]), dim=1)
        return self.to_rs_order(new_Ltoken)


class CrossLformer(Lformer):
    def __init__(self,
                 css,
                 transformer_config,
                 text_encoder_config,
                 first_stage_config,
                 pkeep=1.0,
                 ckpt_path=None,
                 ignore_keys=[],
                 ):
        super().__init__(css,
                         transformer_config,
                         text_encoder_config,
                         first_stage_config,
                         pkeep) 

        hs_cond = transformer_config.params.dim_cond
        hs_trans = transformer_config.params.n_embd
        self.linear_ctx = nn.Linear(hs_cond, hs_trans)

        self.register_buffer("pos_ids", torch.arange(self.css**2).unsqueeze(0))
        self.register_buffer("att_mask", self.get_att_mask())
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def get_context(self, text_feature):
        return self.linear_ctx(text_feature).unsqueeze(1) # [B,H]

    def forward(self, text_feature, text_hidden, padded):
        seq_len = padded.shape[1] + 1 # add one for text_feature
        att_mask = self.att_mask[:, :seq_len, :seq_len]
        pos_ids = self.pos_ids[:, :seq_len]
        context = self.linear_ctx(text_feature).unsqueeze(1) # [B,1,H]
        logits = self.transformer(padded, att_mask, pos_ids, 
            embeddings=context, eh=text_hidden)
        return logits, None


class CrossLformerLatent(Lformer):
    def __init__(self,
                 css,
                 w_layer,
                 transformer_config,
                 text_encoder_config,
                 first_stage_config,
                 pkeep=1.0,
                 ckpt_path=None,
                 ignore_keys=[],
                 ):
        super().__init__(css,
                         transformer_config,
                         text_encoder_config,
                         first_stage_config,
                         pkeep
                         )

        hs_cond = transformer_config.params.dim_cond
        hs_trans = transformer_config.params.n_embd
        self.linear_mu_c = nn.Linear(hs_cond, hs_cond)
        self.linear_mu_z = nn.Linear(hs_cond, hs_cond)
        self.linear_log_var = nn.Linear(hs_cond, hs_cond)
        self.linear_ctx = nn.Linear(hs_cond * 2, hs_trans)
        layers = []
        for _ in range(w_layer):
            layers.extend([nn.Linear(hs_cond, hs_cond), nn.LeakyReLU(inplace=True)])
        self.latent_to_w = nn.Sequential(*layers)

        self.register_buffer("pos_ids", torch.arange(self.css**2).unsqueeze(0))
        self.register_buffer("att_mask", self.get_att_mask())
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def get_input(self, batch):
        text_idx, images =  batch['text_idx'], batch['image']
        img_feature = batch['feature']
        _, _, img_idx= self.first_stage_model.encode(images.to(self.dtype))
        img_idx = img_idx.view(images.shape[0], -1)
        if self.pkeep < 1 and self.training:
            img_idx = self._add_noise(img_idx, self.pkeep)
        Ltoken = self.to_L_order(img_idx)
        return text_idx, Ltoken, img_idx, img_feature 

    def shared_step(self, batch, batch_idx):
        text_idx, Ltoken, _, img_feature = self.get_input(batch)
        text_feature, text_hidden = self.text_encoder(text_idx)
        padded = self.pad_Ltoken(Ltoken)
        logits, KLD = self(text_feature, text_hidden, padded, img_feature)
        CE = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Ltoken.reshape(-1))
        loss = CE + KLD
        return loss, KLD

    def get_context(self, text_feature):
        mu_c = self.linear_mu_c(text_feature)
        z = self.reparameterize(mu_c, torch.zeros_like(mu_c))
        w = self.latent_to_w(z)
        context = self.linear_ctx(torch.cat((text_feature, w), dim=1)) # [B,H]
        return context.unsqueeze(1)

    def forward(self, text_feature, text_hidden, padded, img_feature=None):
        mu_c = self.linear_mu_c(text_feature)
        KLD = None
        if img_feature is not None:
            mu_z = self.linear_mu_z(img_feature)
            log_var = self.linear_log_var(img_feature)
            z = self.reparameterize(mu_z, log_var)
            KLD = -0.5 * torch.sum(1 + log_var - (mu_c - mu_z).pow(2) - log_var.exp(), dim=-1).mean()
        else:
            z = self.reparameterize(mu_c, torch.zeros_like(mu_c))
        w = self.latent_to_w(z)
        context = self.linear_ctx(torch.cat((text_feature, w), dim=1)).unsqueeze(1) # [B,1,H]

        seq_len = padded.shape[1] + 1
        att_mask = self.att_mask[:, :seq_len, :seq_len]
        pos_ids = self.pos_ids[:, :seq_len]
        logits = self.transformer(padded, att_mask, pos_ids, 
            embeddings=context, eh=text_hidden)
        return logits, KLD

    def configure_optimizers(self):
        extra_modules = [self.linear_mu_c, self.linear_mu_z, 
                         self.linear_log_var, self.linear_ctx, self.latent_to_w]
        return self._configure_optimizers(extra_modules)

    def training_step(self, batch, batch_idx):
        loss, KLD = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/KLD", KLD, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss