import os
import numpy as np
from .utils import WrappedDataset

class CelebA(WrappedDataset):
    def __init__(self, data_root, img_root=None, meta=None, fea=None,
                 img_size=288, crop_size=256, 
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], 
                 split='train'):
        super().__init__(data_root, img_root, meta, fea, 
                         img_size, crop_size, mean, std, split)
        self.img_ids = self.metadata[split]

    def get_example(self, i):
        img_id = self.img_ids[i]
        img_path = os.path.join(self.img_root, f"{img_id}.jpg")
        raw_pil_img = self._img_loader(img_path)
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


class CelebAFeature(CelebA):
    def __init__(self, data_root, img_root=None, meta=None, fea=None,
                 img_size=288, crop_size=256, 
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], 
                 split='train'):
        super().__init__(data_root, img_root, meta, fea, 
                         img_size, crop_size, mean, std, split)

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        example = self.get_example(i)
        example['feature'] = self.img_id_to_feature[img_id]
        return example
        