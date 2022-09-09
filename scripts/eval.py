import argparse, os, sys, time
dirname = os.path.dirname(__file__)
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--path1", type=str, default="/mnt/lijiacheng/data/coco/val2017/")
    parser.add_argument("--path2", type=str, help="path contains images")
    parser.add_argument("--gpus", type=str, help="which gpus to use")
    return parser
 
class Imageset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path, transform=None):
        'Initialization'
        self.file_names = self.get_filenames(path)
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        filename = self.file_names[index]
        img = Image.open(filename).convert('RGB')
        caption = filename.strip('.png').strip('.jpg')
        # Convert image and label to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        example = {'img': img, 'caption': caption}
        return example

    def get_filenames(self, data_path):
        images = []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                    filename = os.path.join(path, name)
                    if os.path.isfile(filename):
                        images.append(filename)
        return images

if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(f"path1={args.path1}")
    print(f"path2={args.path2}")

    dataset1 = Imageset(args.path1, transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.PILToTensor(),
        ]))
    dataset2 = Imageset(args.path2, transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.PILToTensor(),
        ]))
    dataloader1 = DataLoader(dataset=dataset1, batch_size=args.batch, shuffle=False,
                            drop_last=False, num_workers=16, pin_memory=True)
    dataloader2 = DataLoader(dataset=dataset2, batch_size=args.batch, shuffle=False,
                            drop_last=False, num_workers=16, pin_memory=True)
    print(f"num of dataset1: {len(dataset1)}, num of dataset2: {len(dataset2)}")
    torch.manual_seed(23)
    fid = FrechetInceptionDistance(feature=2048).cuda()
    inception = InceptionScore(feature=2048).cuda()

    for batch in tqdm(dataloader1):
        fid.update(batch['img'].cuda(), real=True)
    for batch in tqdm(dataloader2):
        imgs = batch['img'].cuda()
        fid.update(imgs, real=False)
        inception.update(imgs)
    fid_score = fid.compute()
    inception_score = inception.compute()
    print(f"FID score={fid_score.item()}, Inception Score={inception_score}")
