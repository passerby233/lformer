import os, sys, time
dirname = os.path.dirname(__file__)
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from PIL import Image

from sample_utils import get_parser, get_config, load_model_and_data
from sample_utils import sample, make_grid

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config, ckpt = get_config(opt, unknown)

    if opt.out is None:
        opt.out = "/mnt/lijiacheng/logs/samplep/"
    os.makedirs(opt.out, exist_ok=True)

    model, data = load_model_and_data(config, ckpt, opt.gpu)
    dataloader = DataLoader(data.datasets['validation'], batch_size=opt.bs, pin_memory=True)

    print("Testing on time:")
    for i, batch in enumerate(dataloader):
        text_idx = batch['text_idx']
        if opt.gpu:
            text_idx = text_idx.cuda()
        start_time = time.time()
        img_idx = model.sample(text_idx, top_k=1024, top_p=0.9, use_cache=opt.cache)
        code_time_point = time.time()
        img_t = model.decode_to_img(img_idx)
        image_time_point = time.time()
        print(f"batch_{i} uses time: Total {image_time_point-start_time:.3}s, " +\
            f"Sample {code_time_point-start_time:.3}s, " +\
            f"Decode {image_time_point-code_time_point:.3}s")
        
        img_grid = make_grid(img_t)
        img_path = os.path.join(opt.out, f"imgs_batch_{i}.png")
        Image.fromarray(img_grid).save(img_path)

        # Save text
        text_idx = text_idx.detach().cpu().numpy()
        text_path = os.path.join(opt.out, f"text_batch_{i}.txt")
        with open(text_path, 'w') as f:
            for text_id in text_idx:
                text = model.tokenizer.decode(text_id)
                f.write(text+'\n')
        
        if i>10:
            break