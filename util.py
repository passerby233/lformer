import os
import argparse, importlib
from omegaconf import OmegaConf
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from PIL import Image
import numpy as np
import time

def download_data(source_dir, target_dir):
    import moxing as mox
    print(f"copying from {source_dir} to {target_dir}")
    start_t = time.time()
    mox.file.copy_parallel(source_dir, target_dir)
    print(f"copy succeeded, all done in {time.time()-start_t} s.")

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name",type=str, const=True, default="", 
                        nargs="?", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, const=True, default="",
                        nargs="?",help="resume from logdir or checkpoint in logdir",)
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", default=list(),
                        help="paths to base configs. Loaded from left-to-right. "
                        "Parameters can be overwritten or added with command-line options of the form `--key value`.")
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=True,
                        nargs="?", help="train")
    parser.add_argument("--no-test", type=str2bool, const=True, default=True,
                        nargs="?",help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, const=True, default=False,
                        nargs="?", help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23,
                        help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="",
                        help="post-postfix for default name")

    parser.add_argument('--deepspeed', type=int, default=0, help='whether to use deepspeed')
    parser.add_argument('--world_size', type=int, default=1, help='total number of nodes')
    parser.add_argument("--train_url", type=str, default=None,
                        help="data download folder for ModelArts")
    parser.add_argument("--data_url", type=str, default=None,
                        help="data upload folder for ModelArts")
    return parser
    
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config)
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config)
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=False):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, root, images,
                  global_step, current_epoch, batch_idx):
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = grid.permute(1, 2, 0) # [C,H,W] to [H,W,C]
            grid = (grid.numpy() * 255).astype(np.uint8) # [0,1] to [0,255]
            filename = "{}_gs-{:06}_e-{:03}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only        
    def log_text(self, root, text_list, 
                 global_step, current_epoch, batch_idx):
        filename = f"Text_gs{global_step:06}" +\
            f"_e-{current_epoch:03}" +\
            f"_b-{batch_idx:06}.txt"
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, 'w') as f:
            for text in text_list:
                f.write(text+'\n')

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            images, text_list = None, None
            with torch.no_grad():
                output = pl_module.log_images(batch, split=split, pl_module=pl_module)
            if isinstance(output, tuple):
                images, text_list = output
            else:
                images = output

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

            root = os.path.join(pl_module.logger.save_dir, "images", split)
            self.log_local(root, images, pl_module.global_step, 
                           pl_module.current_epoch, batch_idx)
            if text_list is not None:
                self.log_text(root, text_list, pl_module.global_step, 
                              pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


def get_default_cfgs(lightning_config, opt, nowname, now,
                     logdir, ckptdir, cfgdir, config):
    logger_cfg = get_logger_cfgs(lightning_config, nowname, logdir, opt)
    modelckpt_cfg = get_modelckpt_cfg(lightning_config, ckptdir)
    callbacks_cfg = get_callbacks_cfg(lightning_config, now, opt, logdir, ckptdir, cfgdir, config)
    callbacks_cfg['checkpoint_callback'] = modelckpt_cfg
    return logger_cfg, callbacks_cfg

def get_logger_cfgs(lightning_config, nowname, logdir, opt):
    """
    # default logger configs
    """
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir,
        }
    }
    
    logger_cfg = lightning_config.get('logger', OmegaConf.create())
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_modelckpt_cfg(lightning_config, ckptdir):
    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:04}-{step:06}",
            "verbose": True,
            "save_last": True,
            "every_n_epochs": 10,
            #"every_n_train_steps": ,
            #"save_top_k": -1
        }
    }
    modelckpt_cfg = lightning_config.get('modelcheckpoint', OmegaConf.create())
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    return modelckpt_cfg

def get_callbacks_cfg(lightning_config, now, opt, logdir, ckptdir, cfgdir, config):
    # add callback which sets up log directory
    default_callbacks_cfg = {
            "setup_callback": {
                "target": "util.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "util.ImageLogger",
                "params": {
                    "batch_frequency": 2000,
                    "max_images": 8,
                    "clamp": True,
                    "increase_log_steps": False
                }
            },
            "learning_rate_logger": {
                "target": "util.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
    callbacks_cfg = lightning_config.get('callbacks', OmegaConf.create())
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    return callbacks_cfg