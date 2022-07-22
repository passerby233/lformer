import os, sys
import sys
# To ensure we run scripts in the correct dir.
dirname = os.path.dirname(os.path.abspath(__file__))
proj_path = os.path.join(dirname, os.pardir)
os.chdir(proj_path)
sys.path.insert(0, os.getcwd())
from util import instantiate_from_config
from omegaconf import OmegaConf
import torch
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
from third_party.clip import clip

class WrappedModel(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, img_t):
        img_features = self.clip.encode_image(img_t)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features

def get_args():
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
    parser.add_argument('-b', '--cfg_path',type=str,
                        default="configs/extract/extract_cub_feature.yaml",
                        help="extract config file")
    parser.add_argument('-s', '--save_prefix',type=str,
                        help="extracted_pickle_save_path")
    parser.add_argument('-m', '--mode', type=str,
                        default='val', help="dataset mode",)
    parser.add_argument('-g','--gpu',  type=str2bool, default=True,
                        help="use gpu or not")
    parser.add_argument('-c','--cc3m',  type=str2bool, default=False,
                        help="whether to download cc data")
    parser.add_argument("--train_url", type=str, default=None,
                        help="data download folder for ModelArts")
    parser.add_argument("--init_method", default="tcp://127.0.0..1:6666", 
                        help="tcp_port")
    args = parser.parse_args()
    return args

def get_model(config, args):
    model, preprocess = clip.load(config.clip_ckpt_path, device='cpu')
    model = WrappedModel(model)
    if args.gpu:
        model.cuda()
        model = torch.nn.DataParallel(model)
    model = model.eval()
    return model

def extract(model, data, mode):
    if mode == 'train':
        dataloader = data.train_dataloader()
    elif mode == 'val':
        dataloader = data.val_dataloader()

    data_dict = defaultdict(dict)
    pbar = tqdm(dataloader)
    pbar.set_description(f'Extracting image features')
    for batch in pbar:
        img_input = batch['image']
        img_features = model(img_input)
        img_features = img_features.detach().cpu().numpy()
        indexes = batch['index'].detach().numpy()
        for batch_idx in range(len(indexes)):
            global_idx = indexes[batch_idx]
            img_id = dataloader.dataset.img_ids[global_idx]
            data_dict['img_id_to_feature'][img_id] = img_features[batch_idx]
    return data_dict

def extract_and_save(model, data, args, mode):
    data_dict = extract(model, data, mode)
    save_path = f'{args.save_prefix}_{mode}.pkl'
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(save_path,'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Features successfully dumped to {save_path}')    

def main():
    args = get_args()
    config = OmegaConf.load(args.cfg_path)
    if config.save_prefix is not None:
        args.save_prefix = config.save_prefix
    if args.cc3m:
        os.system("python prepare_cc3m.py")
    # Init model and data
    model = get_model(config, args)
    data = instantiate_from_config(config.data)
    data.setup()

    extract_and_save(model, data, args, args.mode)
    # cc3m needs dataupload, thus defaultly extract training set
    if args.cc3m: 
        extract_and_save(model, data, args, 'train')

if __name__ == "__main__":
    main()
