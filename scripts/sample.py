import os, sys, glob, time
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from sample_utils import get_parser, get_config, save_text, load_model
from sample_utils import SamplerWithCLIP
from taming.models.custom_clip import clip_transform, VisualEncoder
from util import instantiate_from_config

def test_sample(sampler, dataloader, args):
    device = "cuda" if args.gpu else "cpu"
    if hasattr(sampler, 'module'):
        tokenizer = sampler.module.model.tokenizer
    else:
        tokenizer = sampler.model.tokenizer

    for i, batch in enumerate(dataloader):
        
        batch_text_idx = batch['text_idx'].to(device)

        print(f"Sampling for batch {i}")
        start_time = time.time()
        image_sample = sampler(batch_text_idx, args.num_s, args.cdt, args.fbs,
                                args.top_k, args.top_p).detach().cpu()
        print(f"batch_{i} uses time: {time.time() - start_time}s")
        
        img_path = os.path.join(args.out, f"imgs_batch_{i}.png")
        save_image(image_sample, img_path, nrow=4)

        text_path = os.path.join(args.out, f"text_batch_{i}.txt")
        save_text(batch_text_idx, tokenizer, text_path)

        if i>4:
            break

def sample_for_eval(sampler, dataloader, args):
    device = "cuda" if args.gpu else "cpu"
    batch_size = dataloader.batch_size
    if hasattr(sampler, 'module'):
        tokenizer = sampler.module.model.tokenizer
    else:
        tokenizer = sampler.model.tokenizer

    start = time.time()
    for batch_idx, batch in enumerate(dataloader):
        batch_text_idx = batch['text_idx'].to(device)
        image_sample = sampler(batch_text_idx, args.num_s, args.cdt, args.fbs,
                               args.top_k, args.top_p).detach().cpu()
        image_batch = image_sample.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(torch.uint8)
        for local_idx, img_t in enumerate(image_batch):
            global_idx = batch_idx * args.bs + local_idx
            if global_idx >= args.num_a:
                break
            caption = tokenizer.decode(batch_text_idx[local_idx].detach().cpu().numpy())
            caption = caption.strip('<|startoftext|>').rstrip('<|endoftext|>')
            caption = caption.replace('/', ' ')
            imgpath = os.path.join(args.out, f"{caption[:200]}.png")
            Image.fromarray(img_t.numpy()).save(imgpath)
        if global_idx >= args.num_a:
            break
    print(f"Generating {len(dataloader.dataset)} images uses {time.time()-start:.4}s")


if __name__ == "__main__":
    # Get args, detail options see sample_utils.py
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    config, ckpt = get_config(args, unknown)

    # Load Model and Data
    model = load_model(config, ckpt)
    dataset = instantiate_from_config(config.data.params.validation)
    dataloader = DataLoader(dataset, batch_size=args.bs, pin_memory=True, shuffle=True)

    # Wrap CLIP model for ranking
    clip_ckpt_path = "/home/ma-user/work/lijiacheng/pretrained/ViT-B-16.pt"
    print(f"Restored from {clip_ckpt_path}")
    ranker = VisualEncoder(clip_ckpt_path)
    sampler = SamplerWithCLIP(model, ranker, clip_transform)
    sampler = nn.DataParallel(sampler)
    if args.gpu:
        sampler = sampler.cuda()

    # Create a new target folder
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    """
    if os.path.exists(args.out):
        os.removedirs(args.out)
    os.mkdir(args.out)
    """

    if args.eval:
        sample_for_eval(sampler, dataloader, args)
        del sampler
        torch.cuda.empty_cache()
        #os.system(f"python scripts/eval.py --path2={args.out} --gpus=0")
    else:
        test_sample(sampler, dataloader, args)
