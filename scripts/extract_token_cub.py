import os, sys
sys.path.append(os.getcwd())
# add project dir to the path
from util import instantiate_from_config
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
import torch
import pickle
import argparse

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
                        default="configs/extract/extract_vqgan.yaml",
                        help="extract config file")
    parser.add_argument('-s', '--save_prefix',type=str,
                        help="extract config file")
    parser.add_argument('-m', '--mode', type=str,
                        default='val', help="dataset mode",)
    parser.add_argument('-g','--gpu', type=str2bool,
                        default=False, help="use gpu or not")
    args = parser.parse_args()
    return args

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def init_first_stage_from_ckpt(model_config, gpu_dp):
    model = instantiate_from_config(model_config)
    model = WrappedModel(model)
    if gpu_dp:
        model.cuda()
        model = torch.nn.DataParallel(model)
    model = model.eval()
    model.train = disabled_train
    return model

class WrappedModel(nn.Module):
    def __init__(self, fs_model):
        super().__init__()
        self.vqmodel = fs_model

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[..., None]
        _, _, img_tokens = self.vqmodel.encode(x)
        img_tokens = img_tokens.view(x.shape[0], -1)
        return img_tokens

class Extractor:
    def __init__(self, model_config, gpu_dp=False):
        self.vqmodel = init_first_stage_from_ckpt(model_config, gpu_dp)

    def extract(self, data, mode='val', save_prefix=None):
        save_path = f'{save_prefix}_{mode}.pkl'
        if mode == 'train':
            dataloader = data.train_dataloader()
            dataset = data.datasets["train"]
        elif mode == 'val':
            dataloader = data.val_dataloader()
            dataset = data.datasets["validation"]

        data_dict = defaultdict(dict)
        img_key_list = ['image_128', 'image_256']
        data_key_list = ['img_id_to_16x', 'img_id_to_32x']

        pbar = tqdm(dataloader)
        pbar.set_description(f'Extracting image tokens')
        for batch in pbar:
            for img_key, data_key in zip(img_key_list, data_key_list):
                img_input = batch[img_key]
                img_tokens = self.vqmodel(img_input)
                img_tokens = img_tokens.detach().cpu().numpy()
                indexes = batch['index'].detach().numpy()
                for batch_idx in range(len(indexes)):
                    global_idx = indexes[batch_idx]
                    img_id = dataset.img_ids[global_idx]
                    data_dict[data_key][img_id] = img_tokens[batch_idx]

        data_dict['img_ids'] = dataset.img_ids
        data_dict['img_id_to_captions'] = dataset.img_id_to_captions
        data_dict['img_id_to_filename'] = dataset.img_id_to_filename 
        with open(save_path,'wb') as f:
            pickle.dump(data_dict, f)
        print(f'Indices successfully dumped to {save_path}')
 
def main():
    args = get_args()
    config = OmegaConf.load(args.cfg_path)
    extractor = Extractor(config.first_stage_config, args.gpu)
    if config.save_prefix is not None:
        args.save_prefix = config.save_prefix
    print(args)
    data = instantiate_from_config(config.data)
    data.setup()
    extractor.extract(data, args.mode, args.save_prefix)

if __name__ == "__main__":
    main()
