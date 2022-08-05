import os, sys, glob, time
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from sample_utils import get_parser, get_config, save_text
from sample_utils import load_model_and_data
from sample_utils import SamplerWithCLIP
from taming.models.custom_clip import clip_transform, VisualEncoder

def test_sample(sampler, dataloader, opt):
    device = "cuda" if opt.gpu else "cpu"
    if hasattr(sampler, 'module'):
        tokenizer = sampler.module.model.tokenizer
    else:
        tokenizer = sampler.model.tokenizer

    for i, batch in enumerate(dataloader):
        
        batch_text_idx = batch['text_idx'].to(device)

        print(f"Sampling for batch {i}")
        start_time = time.time()
        image_sample = sampler(batch_text_idx, opt.num_s, opt.cdt, opt.fbs).detach().cpu()
        print(f"batch_{i} uses time: {time.time() - start_time}s")
        
        img_path = os.path.join(opt.out, f"imgs_batch_{i}.png")
        save_image(image_sample, img_path)

        text_path = os.path.join(opt.out, f"text_batch_{i}.txt")
        save_text(batch_text_idx, tokenizer, text_path)

        if i>4:
            break

def sample_for_eval(sampler, dataloader, opt):
    device = "cuda" if opt.gpu else "cpu"
    batch_size =dataloader.batch_size
    if hasattr(sampler, 'module'):
        tokenizer = sampler.module.model.tokenizer
    else:
        tokenizer = sampler.model.tokenizer

    start = time.time()
    for batch_idx, batch in enumerate(dataloader):
        batch_text_idx = batch['text_idx'].to(device)
        image_sample = sampler(batch_text_idx, opt.num_s, opt.cdt, opt.fbs).detach().cpu()
        image_batch = image_sample.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(torch.uint8)
        for local_idx, img_t in enumerate(image_batch):
            global_idx = batch_idx * batch_size + local_idx
            imgpath = os.path.join(opt.out, f"{global_idx}.png")
            txtpath = os.path.join(opt.out, f"{global_idx}.txt")
            Image.fromarray(img_t.numpy()).save(imgpath)
            save_text(batch_text_idx, tokenizer, txtpath)
        if batch_idx > 3:
            break
    print(f"Generating {len(dataloader.dataset)} images uses {time.time()-start:.4}s")


if __name__ == "__main__":
    # Get args, detail options see sample_utils.py
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config, ckpt = get_config(opt, unknown)

    # Load Model and Data
    model, data = load_model_and_data(config, ckpt)
    dataloader = DataLoader(data.datasets['validation'], batch_size=opt.bs, pin_memory=True)

    # Wrap CLIP model for ranking
    clip_ckpt_path = "/home/ma-user/work/lijiacheng/pretrained/ViT-B-16.pt"
    print(f"Restored from {clip_ckpt_path}")
    ranker = VisualEncoder(clip_ckpt_path)
    sampler = SamplerWithCLIP(model, ranker, clip_transform)
    sampler = nn.DataParallel(sampler)
    if opt.gpu:
        sampler = sampler.cuda()

    # Create target folder
    if not os.path.exists(opt.out):
        os.mkdir(opt.out)

    if opt.eval:
        sample_for_eval(sampler, dataloader, opt)
    else:
        test_sample(sampler, dataloader, opt)
