import os, sys, time
dirname = os.path.dirname(__file__)
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from sample_utils import get_parser, get_config, load_model_and_data, save_text

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config, ckpt = get_config(opt, unknown)

    if opt.out is None:
        opt.out = "/mnt/lijiacheng/logs/sample/"
    os.makedirs(opt.out, exist_ok=True)

    model, data = load_model_and_data(config, ckpt, opt.gpu)
    dataloader = DataLoader(data.datasets['validation'], batch_size=opt.bs, pin_memory=True)
    time_list = []

    print("Testing on time:")
    for i, batch in enumerate(dataloader):
        text_idx = batch['text_idx'].to(model.device)

        start_time = time.time()
        img_idx = model.sample(text_idx, top_k=opt.top_k, top_p=opt.top_p, use_cache=opt.cache)
        code_time_point = time.time()
        img_t = model.decode_to_img(img_idx)
        image_time_point = time.time()

        if i > 0: # omit the first step
            time_list.append(image_time_point-start_time)
        print(f"batch_{i} uses time: Total {image_time_point-start_time:.4}s, " +\
            f"Sample {code_time_point-start_time:.3}s, " +\
            f"Decode {image_time_point-code_time_point:.4}s")
        
        img_path = os.path.join(opt.out, f"imgs_batch_{i}.png")
        save_image(img_t, img_path)

        text_path = os.path.join(opt.out, f"text_batch_{i}.txt")
        save_text(text_idx, model.tokenizer, text_path)

        if i>9:
            break

    print(f"Average time per batch:{sum(time_list) / len(time_list):.4}")