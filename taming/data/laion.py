import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import lmdb
import numpy as np
from  torchvision import transforms as T
from third_party.clip.clip import CLIPTokenizer


class LAION(Dataset):
    def __init__(self, img_root=None, meta=None,
                img_size=256, crop_size=256,
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                split='train', tokenize=True):
        super().__init__()
        # cap_path need to be prepared in ma_entry.py
        # cap_path = '/cache/mm_en_filename_caption.lmdb'
        print(f"Loading from {meta}")
        self.env = lmdb.open(meta, readonly=True, lock=False, readahead=False, meminit=False)
        
        with self.env.begin() as txn:
            self.dataset_len = txn.stat()['entries']

        self.img_root = img_root
        self.tokenize = tokenize
        if tokenize:
            self.tokenizer = CLIPTokenizer()
        self.transform = self.get_transform(img_size, crop_size, mean, std, split)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, i):
        key = str(i).encode('utf-8')
        with self.env.begin() as txn:
            filename, caption = txn.get(key).decode('utf-8').split('\t', 1)
        example = dict(
            index = i,
            caption = caption,
            img_path = os.path.join(self.img_root, filename)
        )
        try:
            example = self._get_example(example)
        except Exception as err:
            print(err)
            print(f'corrupted image: {example["img_path"]}')
            ritem = torch.randint(0, len(self), (1,)).item()
            return self[ritem]
        return example

    def _get_example(self, example):
        raw_pil_img = self._img_loader(example['img_path'])
        image = self.transform(raw_pil_img)
        example['image'] = image
        if self.tokenize:
            text_idx = self.tokenizer(example['caption']).squeeze()
            example['text_idx'] = text_idx
        return example

    def get_transform(self, img_size, crop_size, mean, std, split):
        transform = T.Compose([
            T.Resize(img_size), 
            T.RandomCrop(crop_size) if split == 'train' else T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        return transform   
        
    def _img_loader(self, obj):
        image = Image.open(obj) # read from filepath by default
        if not image.mode == "RGB": # convert to RGB
            image = image.convert("RGB")
        return image

import hashlib
class LAIONFeature(LAION):
    def __init__(self, fea=None, *arg, **kwargs):
        self.fea = fea
        super().__init__(*arg, **kwargs)

    def __getitem__(self, i):
        example = super().__getitem__(i)
        
        hash_object = hashlib.sha1(example['img_path'].encode('utf-8'))
        hex_dig = hash_object.hexdigest()
        example['feature'] = np.load(os.path.join(self.fea, hex_dig+f'.npy'));
        return example