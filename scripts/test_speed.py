import os, sys, time
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from PIL import Image

from sample_utils import get_parser, get_config, load_model_and_data
from sample_utils import de_normalize, sample, make_grid

if __name__ == "__main__":
    out_path = "/home/ljc/lformer/outputs/"
    
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config, ckpt = get_config(opt, unknown)

    model, data = load_model_and_data(config, ckpt, opt.gpu)
    dataloader = DataLoader(data.datasets['validation'], batch_size=opt.bs, pin_memory=True)

    print("Testing on time:")
    for i, batch in enumerate(dataloader):
        text_idx = batch['text_idx']
        if opt.gpu:
            text_idx = text_idx.cuda()
        start_time = time.time()
        img_t = sample(model, text_idx, sample=True, top_k=100)
        time_used = time.time() - start_time
        print(f"batch_{i} uses time: {time_used}s")

        img_grid = make_grid(img_t)
        img_path = os.path.join(out_path, f"imgs_batch_{i}.png")
        Image.fromarray(img_grid).save(img_path)

        text_grid = make_grid(de_normalize(model.textidx_to_img(text_idx)))
        text_path = os.path.join(out_path, f"text_batch_{i}.png")
        Image.fromarray(text_grid).save(text_path)

        if i>5:
            break