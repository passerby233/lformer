from math import sqrt
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


from util import instantiate_from_config
from third_party.clip.clip import CLIPTokenizer

def convert_ckpt(state_dict):
    import re
    pattern = re.compile(r'module.(.*)')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore. 
    model.train() will apply reccursively thus we must disable the train() 
    of a submodel which needs to be frozen."""
    return self

class GenModel(pl.LightningModule):
    def __init__(self,
                 css,
                 transformer_config,
                 first_stage_config,
                 pkeep=1.0
                 ):
        super().__init__()
        self.css = css
        self.pkeep = pkeep
        self.block_size = transformer_config.params.block_size
        self.text_vbs = transformer_config.params.text_vbs
        self.img_vbs = transformer_config.params.img_vbs

        self.first_stage_model = self.init_frozen_from_ckpt(first_stage_config)
        self.transformer = instantiate_from_config(transformer_config)
        self.tokenizer = CLIPTokenizer()

    def init_from_ckpt(self, path, ignore_keys=list()):
        ckpt_dict = torch.load(path, map_location="cpu")
        if "state_dict" in ckpt_dict:
            sd = ckpt_dict["state_dict"]
        else:
            sd = convert_ckpt(ckpt_dict) # a state_dict with module.param
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_frozen_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model.freeze()
        model.train = disabled_train
        return model

    @torch.no_grad()
    def decode_to_img(self, index):
        img_t = self.first_stage_model.decode_to_img(index)
        return img_t

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def _add_noise(self, imgidx, pkeep):
        mask = torch.bernoulli(pkeep*torch.ones_like(imgidx))
        mask = mask.long().to(imgidx.device)
        r_indices = torch.randint_like(imgidx, self.img_vbs)
        imgidx = mask*imgidx + (1-mask)*r_indices
        return imgidx

    def textidx_to_text(self, text_idx):
        text_list = []
        for text_id in text_idx:
            text_id = text_id.detach().cpu().numpy()
            text = self.tokenizer.decode(text_id)
            text_list.append(text)
        return text_list

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
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

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.045},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = self.get_optimizer(optim_groups)
        if self.scheduler_params is not None:
            scheduler = self.get_scheduler(optimizer, self.scheduler_params)
            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def get_optimizer(self, optim_groups):
        if self.optimizer_params is not None:
            if self.optimizer_params.type == "Adafactor":
                from third_party.optimizer.optimizer import Adafactor
                optimizer = Adafactor(optim_groups, lr=self.learning_rate,
                                    scale_parameter=False, relative_step=False)
            elif self.optimizer_params.type == "DeepSpeedCPUAdam":
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                optimizer = DeepSpeedCPUAdam(optim_groups, lr=self.learning_rate, betas=(0.9, 0.96))
            else:
                raise Exception(f"Unrecognized optimizer {self.optimizer_params.type}")
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.96))
        print(f"=====Using Optimizer: {type(optimizer)}")
        return optimizer

    def get_scheduler(self, optimizer, opt):
        if opt.type == "OneCycle":    
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=opt.max_lr,
                total_steps=opt.total_steps if opt.total_steps is not None \
                            else self.trainer.estimated_stepping_batches,
                pct_start=opt.pct_start,
                cycle_momentum="momentum" in optimizer.defaults)
            scheduler = {"scheduler": scheduler, "interval":"step"}
        elif opt.type == "cos_warmup":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=opt.total_steps if opt.total_steps is not None \
                                 else self.trainer.estimated_stepping_batches,
                num_training_steps=opt.total_steps)
            scheduler = {"scheduler": scheduler, "interval":"step"}
        elif opt.type == 'ReduceLROnPlateauWithWarmup':
            from third_party.scheduler.scheduler import ReduceLROnPlateauWithWarmup
            scheduler = ReduceLROnPlateauWithWarmup(
                optimizer,
                patience=opt.patience,
                min_lr=opt.min_lr,
                threshold=opt.threshold,
                warmup_lr=opt.warmup_lr,
                warmup=opt.warmup
            )
            scheduler = {"scheduler": scheduler, "interval":"step", "monitor":"train/loss"}
        else:
            raise Exception(f"Not supported shceduler: {opt.type}")
        print(f"=====Using Scheduler: {type(scheduler['scheduler'])}")
        return scheduler

class LformerBase:
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