import argparse, os, sys, glob, math, time
dirname = os.path.dirname(__file__)
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance

from util import instantiate_from_config
from sample_utils import SamplerWithCLIP, get_config, get_data, convert_ckpt
from taming.models.custom_clip import clip_transform, VisualEncoder

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--resume", type=str, nargs="?",
        help="load from logdir or checkpoint in logdir")
    parser.add_argument(
        "-b", "--base", nargs="*", metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list())
    parser.add_argument(
        "-c", "--config", nargs="?", metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True, default="")
    parser.add_argument(
        "--ignore_base_data", action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.")
    return parser
 
if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config, ckpt = get_config(opt, unknown)
    
    # load data and model
    dsets = get_data(config) 

    ckpt_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in ckpt_dict:
        sd = ckpt_dict["state_dict"]
    else:
        sd = convert_ckpt(ckpt_dict) # a state_dict with module.param
    model = instantiate_from_config(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing = {missing}, unexpected={unexpected}")
    model.cuda().eval()

    # load clip visual encoder to get full sampler
    clip_ckpt_path = "/home/ma-user/work/lijiacheng/pretrained/clip/ViT-B-16.pt"
    print(f"Restored from {clip_ckpt_path}")
    ranker = VisualEncoder(clip_ckpt_path).cuda()
    sampler = SamplerWithCLIP(model, ranker, clip_transform)

    # prepare for evaluation
    _ = torch.manual_seed(23)
    fid = FrechetInceptionDistance(feature=2048)

