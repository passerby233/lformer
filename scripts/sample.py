import os, sys, glob, time
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.append(os.getcwd())
import torch
from torch.utils.data import DataLoader
from PIL import Image
from third_party.clip import clip
from sample_utils import get_parser, get_config
from sample_utils import load_model_and_data, make_grid
from sample_utils import SamplerWithCLIP

if __name__ == "__main__":
    # Get args, detail options see sample_utils.py
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config, ckpt = get_config(opt, unknown)

    # Load Model and Data
    model, data = load_model_and_data(config, ckpt, opt.gpu)
    dataloader = DataLoader(data.datasets['validation'], batch_size=opt.bs, pin_memory=True)

    # Load CLIP model for ranking
    clip_ckpt_path = "/home/ma-user/work/lijiacheng/pretrained/clip/ViT-B-16.pt"
    print(f"Restored from {clip_ckpt_path}")
    clip_model, preprocess = clip.load(clip_ckpt_path, device='cuda' if opt.gpu else 'cpu')
    sampler = SamplerWithCLIP(model, clip_model, preprocess)

    # Create target folder
    if not os.path.exists(opt.out):
        os.mkdir(opt.out)

    for i, batch in enumerate(dataloader):
        batch_text_idx = batch['text_idx']
        if opt.gpu:
            batch_text_idx = batch_text_idx.cuda()

        start_time = time.time()
        sample_list = []
        for k in range(batch_text_idx.shape[0]):
            print(f"Sampling for text {k}")
            image_sample = sampler(batch_text_idx[k:k+1], opt.num_s, opt.cdt, opt.fbs)
            sample_list.append(image_sample)
        samples = torch.cat(sample_list, dim=0)
        time_used = time.time() - start_time
        print(f"batch_{i} uses time: {time_used}s")
        
        # Make image gird and save to path
        img_grid = make_grid(samples)
        img_path = os.path.join(opt.out, f"imgs_batch_{i}.png")
        Image.fromarray(img_grid).save(img_path)

        # Save text
        text_idx = batch_text_idx.detach().cpu().numpy()
        text_path = os.path.join(opt.out, f"text_batch_{i}.txt")
        with open(text_path, 'w') as f:
            for text_id in text_idx:
                text = model.tokenizer.decode(text_id)
                f.write(text+'\n')

        if i>4:
            break