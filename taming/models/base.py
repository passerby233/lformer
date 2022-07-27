import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt 
from io import BytesIO
from PIL import Image
import numpy as np

from util import instantiate_from_config
from third_party.clip.clip import CLIPTokenizer

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
        sd = torch.load(path, map_location="cpu")["state_dict"]
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

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

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
        total_steps = opt.total_steps if opt.total_steps is not None \
            else self.trainer.estimated_stepping_batches
        if opt.type == "OneCycle":    
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=opt.max_lr,
                total_steps=total_steps,
                pct_start=opt.pct_start,
                cycle_momentum="momentum" in optimizer.defaults)
            scheduler = {"scheduler": scheduler, "interval":"step"}
        elif opt.type == "cos_warmup":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=total_steps,
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