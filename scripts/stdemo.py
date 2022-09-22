import argparse, os, sys, math, time
dirname = os.path.dirname(__file__)
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data.dataloader import default_collate
import streamlit as st

from sample_utils import SamplerWithCLIP, get_config, get_data, load_model
from taming.models.custom_clip import clip_transform, VisualEncoder

rescale = lambda x: (x + 1.) / 2.

def bchw_to_st(x):
    return x.detach().cpu().numpy().transpose(0,2,3,1)

def save_img(xstart, fname):
    I = (xstart.clip(0,1)[0]*255).astype(np.uint8)
    Image.fromarray(I).save(fname)

def get_interactive_image(resize=False):
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"])
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        print("upload image shape: {}".format(image.shape))
        img = Image.fromarray(image)
        if resize:
            img = img.resize((256, 256))
        image = np.array(img)
        return image

def single_image_to_torch(x, permute=True):
    assert x is not None, "Please provide an image through the upload function"
    x = np.array(x)
    x = torch.FloatTensor(x/255.*2. - 1.)[None,...]
    if permute:
        x = x.permute(0, 3, 1, 2)
    return x

def pad_to_M(x, M):
    hp = math.ceil(x.shape[2]/M)*M-x.shape[2]
    wp = math.ceil(x.shape[3]/M)*M-x.shape[3]
    x = torch.nn.functional.pad(x, (0,wp,0,hp,0,0,0,0))
    return x

@torch.no_grad()
def run_conditional(sampler, dsets):
    if len(dsets.datasets) > 1:
        split = st.sidebar.radio("Split", sorted(dsets.datasets.keys()))
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))
    batch_size = 1
    start_index = st.sidebar.number_input(f"Example Index (Size: {len(dset)})", 
                                          value=0, min_value=0,
                                          max_value=len(dset)-batch_size)
    indices = list(range(start_index, start_index+batch_size))
    example = default_collate([dset[i] for i in indices])

    gt_img = st.empty()
    gt_txt = st.empty()
    time_txt = st.empty()
    output = st.empty()

    animate = st.checkbox("animate")
    if animate:
        import imageio
        outvid = "sampling.mp4"
        writer = imageio.get_writer(outvid, fps=25)

    ### condition config

    temperature = st.sidebar.number_input("Temperature", value=1.0)
    top_k = st.sidebar.number_input("Top k", value=512)
    top_p = st.sidebar.number_input("Top p", value=0.9)
    candidate = st.sidebar.number_input("candidate", value=32)
    fbs = st.sidebar.number_input("forward_batch_size", value=32)
    num_out = st.sidebar.number_input("num_out", value=4)
    lambda_ = st.sidebar.number_input("lambda_", value=0.0)
    #greedy = st.checkbox("Greedy", value=False)

    device = next(sampler.parameters()).device
    if hasattr(sampler, "module"):
        tokenizer = sampler.module.model.tokenizer
    else:
        tokenizer = sampler.model.tokenizer

    ### Prepare input text
    text_input = st.text_input("Text_Input", "")
    if len(text_input) > 0 :
        text_idx= tokenizer(text_input).to(device)
        raw_text = text_input
    else:
        text_idx = example['text_idx'].to(device)
        raw_text = tokenizer.decode(text_idx[0].detach().cpu().numpy())
    gt_txt.write(f"input_text: {raw_text}")

    ### Show raw image
    x = rescale(example['image'])
    scale_factor = st.sidebar.slider("Scale Factor", min_value=0.5, max_value=4.0, step=0.25, value=1.00)
    if scale_factor != 1.0:
        x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="bicubic")
    gt_img.image(bchw_to_st(x), clamp=True, output_format="PNG")

    ### Sample a batch of images
    if st.button("Sample"):
        output = st.empty()
        # Sample images
        start_t = time.time()
        xout = sampler(text_idx, num_out, candidate, 
                        fbs, top_k, top_p, temperature, lambda_)
        time_txt.text(f"Time: {time.time() - start_t} seconds")
        xout = bchw_to_st(xout)
        output.image(xout, clamp=True, output_format="PNG")

    if animate:
        writer.append_data((xout[0]*255).clip(0, 255).astype(np.uint8))
    #save_img(xstart, "full_res_sample.png")
    if animate:
        writer.close()
        st.video(outvid)

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

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model_and_dset(config, ckpt, gpu, eval_mode=True):
    # get data
    dsets = get_data(config)   # calls data.config ...
    model = load_model(config, ckpt, gpu, eval_mode)
    clip_ckpt_path = "/home/ma-user/work/lijiacheng/pretrained/ViT-B-16.pt"
    print(f"Restored from {clip_ckpt_path}")
    ranker = VisualEncoder(clip_ckpt_path)
    sampler = SamplerWithCLIP(model, ranker, clip_transform)
    #sampler = nn.DataParallel(sampler)
    if gpu:
        sampler = sampler.cuda()
    sampler.eval()
    return dsets, sampler


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config, ckpt = get_config(opt, unknown)

    st.sidebar.text(ckpt)
    gs = st.sidebar.empty()
    st.sidebar.text("Options")
    #gpu = st.sidebar.checkbox("GPU", value=True)
    gpu = True
    #eval_mode = st.sidebar.checkbox("Eval Mode", value=True)
    eval_mode = True
    #show_config = st.sidebar.checkbox("Show Config", value=False)
    show_config = False
    if show_config:
        st.info("Checkpoint: {}".format(ckpt))
        st.json(OmegaConf.to_container(config))

    # Load data
    dsets, sampler = load_model_and_dset(config, ckpt, gpu, eval_mode)

    run_conditional(sampler, dsets)
