import argparse, os, sys, glob, math, time
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.append(os.getcwd())
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from main import instantiate_from_config
from torch.utils.data.dataloader import default_collate
import streamlit as st
from streamlit import caching

from sample_utils import SamplerWithCLIP
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
    top_k = st.sidebar.number_input("Top k", value=100)
    top_p = st.sidebar.number_input("Top p", value=0.9)
    candidate = st.sidebar.number_input("candidate", value=64)
    fbs = st.sidebar.number_input("forward_batch_size", value=32)
    num_out = st.sidebar.number_input("num_out", value=4)
    #greedy = st.checkbox("Greedy", value=False)

    ### Prepare input text
    text_input = st.text_input("Text_Input", "")
    if len(text_input) > 0 :
        text_idx= sampler.model.tokenizer(text_input).to(next(sampler.parameters()).device)
        raw_text = text_input
    else:
        text_idx = example['text_idx'].to(next(sampler.parameters()).device)
        raw_text = sampler.model.tokenizer.decode(example['text_idx'][0].detach().cpu().numpy())
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
                        fbs, top_k, top_p, temperature)
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

def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        st.warning("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        st.warning("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            st.warning("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            st.warning("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        st.info(f"Missing Keys in State Dict: {missing}")
        st.info(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}

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

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    ckpt_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in ckpt_dict:
        sd = ckpt_dict["state_dict"]
    else:
        sd = convert_ckpt(ckpt_dict) # a state_dict with module.param
    model = load_model_from_config(config.model,
                                   sd,
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    clip_ckpt_path = "/home/ma-user/work/lijiacheng/pretrained/clip/ViT-B-16.pt"
    print(f"Restored from {clip_ckpt_path}")
    ranker = VisualEncoder(clip_ckpt_path)
    if gpu:
        ranker.cuda()
    sampler = SamplerWithCLIP(model, ranker, clip_transform)
    return dsets, sampler


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

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