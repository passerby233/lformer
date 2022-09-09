import os, sys
dirname = os.path.dirname(__file__)
os.chdir(os.path.join(dirname, os.path.pardir))
sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from eval import get_parser
from eval import Imageset
from third_party.clip.clip import _transform, CLIPTokenizer

if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    transform = _transform(224)
    dataset = Imageset(args.path2, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False,
                            drop_last=False, num_workers=16, pin_memory=True)
    print(f"num of dataset1: {len(dataset)}, num of dataset2: {len(dataset)}")
    torch.manual_seed(23)

    clip_ckpt_path = "/home/ma-user/work/lijiacheng/pretrained/ViT-B-16.pt"
    tokenizer = CLIPTokenizer()
    with open(clip_ckpt_path, 'rb') as opened_file:
        clip = torch.jit.load(opened_file, map_location="cpu").cuda().eval()

    scores = []
    for batch in tqdm(dataloader):
        imgs = batch['img'].cuda()
        caption = batch['caption']
        for img, caption in zip(imgs, caption):
            text_idx= tokenizer(caption).cuda()
            logits_per_image, logits_per_text = clip(img.unsqueeze(0), text_idx)
            scores.append(logits_per_text.item())

    print(f"CLIP_SCORE: {sum(scores) / len(scores) / 100}")
