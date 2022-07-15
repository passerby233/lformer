import os
import pickle
import numpy as np
from PIL import Image
from .utils import WrappedDataset

class CUB(WrappedDataset):
    def __init__(self, data_root, img_root=None, meta=None, fea=None,
                img_size=304, crop_size=256,
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                split='train', cache=None):
        super().__init__(data_root, img_root, meta, fea, 
                         img_size, crop_size, mean, std, split)
        cache_path = os.path.join(data_root, cache)
        with open(cache_path, 'rb') as f:
            self.img_id_to_images = pickle.load(f)

        self.img_id_to_filename = self.metadata['img_id_to_filename']
        self.img_ids = list(self.img_id_to_images.keys())

    def _img_loader(self, obj):
        image = Image.fromarray(obj) # CUB read from numpy
        if not image.mode == "RGB": # convert to RGB
            image = image.convert("RGB")
        return image

    def get_example(self, i):
        img_id = self.img_ids[i]
        raw_pil_img = self._img_loader(self.img_id_to_images[img_id])
        image = self.transform(raw_pil_img)
        # randomly draw one of all available captions per image
        captions = self.img_id_to_captions[img_id]
        caption = captions[np.random.randint(0, len(captions))]
        text_idx = self.tokenizer(caption).squeeze()
        example = {'index': i, 
                   'image': image,
                   'text_idx': text_idx}
        return example

    def __getitem__(self, i):
        example = self.get_example(i)
        return example

class CUBFeature(CUB):
    def __init__(self, data_root, img_root=None, meta=None, fea=None,
                img_size=304, crop_size=256,
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                split='train', cache=None):
        super().__init__(data_root, img_root, meta, fea, 
                         img_size, crop_size, mean, std, split, cache)

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        example = self.get_example(i)
        example['feature'] = self.img_id_to_feature[img_id]
        return example

"""
deprecated

class CUBImageAndCaptions(Dataset):
    def __init__(self, cache_path, meta_path, 
                resolution=[128, 256], 
                crop_size=[128, 256],
                mean=None, std=None):
        super().__init__()
        with open(cache_path, 'rb') as f:
            self.img_id_to_images = pickle.load(f)
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
        self.img_id_to_captions = self.meta['img_id_to_captions']
        self.img_id_to_filename = self.meta['img_id_to_filename']
        self.img_ids = list(self.img_id_to_images.keys())
        for res, crop in zip(resolution, crop_size):
            setattr(self, f'trans_{res}', self.get_transform(res, crop, mean, std))
        self.resolution = resolution

    def get_transform(self, img_size, crop_size, mean, std):
        transform = T.Compose([
            T.Resize(img_size), 
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        return transform

    def __len__(self):
        return len(self.img_id_to_images)

    def _img_loader(self, array):
        image = Image.fromarray(array)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        captions = self.img_id_to_captions[img_id]
        # randomly draw one of all available captions per image
        caption = captions[np.random.randint(0, len(captions))]
        example = {"index": i, "caption": str(caption)}

        for res in self.resolution:
            image_key = f'image_{res}'
            transform = getattr(self, f'trans_{res}')
            raw_pil_img = self._img_loader(self.img_id_to_images[img_id])
            image = transform(raw_pil_img)
            example[image_key] = image
    
        return example

class CUBClipTokens(Dataset):
    def __init__(self, cache_path, img_keys=['16x', '32x'],
            root="/home/ma-user/work/lijiacheng/data/birds/CUB_200_2011/images"):
        self.root= root
        self.tokenizer = CLIPTokenizer()
        with open(cache_path, 'rb') as f:
            metadata = pickle.load(f)   
        self.img_ids = metadata['img_ids']
        self.img_id_to_captions = metadata['img_id_to_captions']
        self.img_id_to_filename = metadata['img_id_to_filename']
        for key in img_keys:
            dict_name = f'img_id_to_{key}'
            setattr(self, dict_name, metadata[dict_name])
        self.img_keys = img_keys

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        captions = self.img_id_to_captions[img_id]
        caption = captions[np.random.randint(0, len(captions))]
        text_idx = self.tokenizer(caption).squeeze()
        example = {'index': i, 'text_idx': text_idx}
        for key in self.img_keys:
            dict_name = f'img_id_to_{key}'
            example[f'imgidx_{key}'] = getattr(self, dict_name)[img_id]
        return example

class Ltoken(CUBClipTokens):
    def __init__(self, cache_path, css=32,
            root="/home/ma-user/work/lijiacheng/data/birds/CUB_200_2011/images"):
        super().__init__(cache_path, root=root)
        self.css = css # content spatial size

    def to_L_order(self, img_idx, css=None):
        css = self.css if css is None else css
        i, j = 0, 0
        L_order = []
        while i < css and j < css:
            L_order.append(i*css + j)
            if j == 0:
                j = i+1
                i = 0
            elif i<j:
                i += 1
            else:
                j -= 1    
        return img_idx[L_order]  

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        captions = self.img_id_to_captions[img_id]
        caption = captions[np.random.randint(0, len(captions))]
        text_idx = self.tokenizer(caption).squeeze()
        img_idx = getattr(self, f'img_id_to_{self.css}x')[img_id]
        L_token = self.to_L_order(img_idx)
        # img_id = image identification, imgid = image indices
        example = {'index': i,
                   'img_idx': img_idx,
                   'Ltoken': L_token,
                   'text_idx': text_idx}
        return example

class LtokenFeature(Ltoken):
    def __init__(self, cache_path, fea_path, css=32, 
            root="/home/ma-user/work/lijiacheng/data/birds/CUB_200_2011/images"):
        super().__init__(cache_path, css=css, root=root)
        with open(fea_path, 'rb') as f:
            self.img_id_to_feature = pickle.load(f)['img_id_to_feature']

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        captions = self.img_id_to_captions[img_id]
        caption = captions[np.random.randint(0, len(captions))]
        text_idx = self.tokenizer(caption).squeeze()
        img_idx = getattr(self, f'img_id_to_{self.css}x')[img_id]
        L_token = self.to_L_order(img_idx)
        feature = self.img_id_to_feature[img_id]
        # img_id = image identification, imgid = image indices
        example = {'index': i,
                   'img_idx': img_idx,
                   'Ltoken': L_token,
                   'text_idx': text_idx,
                   'feature': feature}
        return example

"""