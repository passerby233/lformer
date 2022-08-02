import os, sys, glob
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
import torchvision
import torchvision.transforms as T

from util import instantiate_from_config

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

size = 224
clip_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(size, interpolation=BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resume", type=str, nargs="?",
                        help="load from logdir or checkpoint in logdir")
    parser.add_argument("--gpu", type=str2bool, default=True, help="whether to use gpu")
    parser.add_argument("--bs", type=int, default=1, help="batch_size")
    parser.add_argument("--num_s", type=int, default=1, help="num_of_sample_for_each_text")
    parser.add_argument("--cdt", type=int, default=32, help="num_of_candidate")
    parser.add_argument("--fbs", type=int, default=32, help="num_of_forward_batch_size_per_step")
    parser.add_argument("--out", type=str, default="/home/ma-user/work/lijiacheng/logs/sample/", 
                        help="img_output_path")
    parser.add_argument("--cache", type=str2bool, default=True, help="whether to use cache")                    

    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list())
    parser.add_argument("-c", "--config", nargs="?", metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True, default="")
    parser.add_argument("--ignore_base_data", action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.")

    return parser

class SamplerWithCLIP(torch.nn.Module):
    def __init__(self, model, ranker, preprocess):
        super().__init__()
        self.model = model
        self.ranker = ranker
        self.preprocess = T.Compose([T.ToPILImage(), preprocess])

    def forward(self, text_idx, num_s=1, candidate=32, fbs=32, 
                 top_k=None, top_p=0.9, temperature=1.0, use_cache=True):
        batch_size = text_idx.shape[0]
        fbs = min(candidate, fbs)
        if batch_size > 1:
            assert candidate == fbs, "batch_size > 1, assert memory is enough"
        assert candidate % fbs == 0, f"candidate % fbs should be 0"
        img_idx = torch.empty((batch_size*candidate, self.model.block_size), 
                                dtype=text_idx.dtype, device=text_idx.device)

        if batch_size == 1:
            for t in range(candidate // fbs):
                ex_text = text_idx.expand(fbs, -1) # repeat text  [B, L] 
                img_idx[candidate*t:candidate*(t+1), :], text_feature = \
                    self.model.sample(ex_text, top_k, top_p, temperature, 
                        return_feature=True, use_cache=use_cache)
        else:
            # [B*cdt, L] ,repeat each text to cdt
            ex_text = text_idx[:,None,:].repeat(1, candidate, 1).view(-1, text_idx.shape[-1])
            img_idx[...], text_feature = \
                self.model.sample(ex_text, top_k, top_p, temperature, 
                                  return_feature=True, use_cache=use_cache)
     
        img_t = self.model.decode_to_img(img_idx) # [B,C,H,W]
        img_processed = torch.stack(
            [self.preprocess(s_img_t) for s_img_t in img_t]).to(text_idx.device)
        
        image_list = []
        img_t = img_t.view(batch_size, candidate, *img_t.shape[-3:])
        img_processed = img_processed.view(batch_size, candidate, *img_processed.shape[-3:])
        text_feature = text_feature.view(batch_size, candidate, -1)
        for batch_idx in range(batch_size):
            s_img_t = img_t[batch_idx]
            s_img_processed = img_processed[batch_idx]
            s_text_feature = text_feature[batch_idx, :1]
            logits_per_image, logits_per_text = self.ranker.clip_score(s_img_processed, s_text_feature)
            value, index = logits_per_text.squeeze().topk(num_s)
            image_list.append(s_img_t[index])
        image_sample = torch.cat(image_list, 0)
        return image_sample

def make_grid(img_t):
    # convert [B,]
    grid = torchvision.utils.make_grid(img_t, nrow=4)
    grid = grid.permute(1,2,0) # [C,H,W] -> [H,W,C]
    grid = (grid.detach().cpu().numpy() * 255).astype(np.uint8) # [0,1] to [0,255]
    return grid

def sample(model, *args, **kargs):
    img_idx = model.sample(*args, **kargs)
    img_t = model.decode_to_img(img_idx)
    return img_t

def load_model_and_data(config, ckpt, gpu=False, eval=True):
    data = instantiate_from_config(config.data)
    data.setup()
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    if gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model)
    if eval:
        model.eval()
    return model, data

def get_config(opt, unknown):
    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)
    return config, ckpt

def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.setup()
    return data

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