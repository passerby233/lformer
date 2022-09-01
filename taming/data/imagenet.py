import imp
import os
import pickle
from torch.utils.data import Dataset
from  torchvision import transforms as T
from PIL import Image

class ImageNet(Dataset):
    def __init__(self, img_root=None, meta=None,
                img_size=256, crop_size=256,
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                split='train'):
        super().__init__()

        with open(meta, 'rb') as f:
            metadata = pickle.load(f)
        self.samples = metadata['samples']
        self.label_to_wid = metadata['label_to_wid']
        self.wid_to_label = metadata['wid_to_label']
        self.name_to_wid = metadata['name_to_wid']
        self.wid_to_name = metadata['wid_to_name']

        self.img_ids = list(range(len(self.samples)))
        self.img_root = img_root
        self.split = split
        self.transform = self.get_transform(img_size, crop_size, mean, std, split)

    def __len__(self):
        return len(self.img_ids)

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

    def get_example(self, i):
        rel_path, label = self.samples[i]
        img_path = os.path.join(self.img_root, self.split, rel_path)
        raw_pil_img = self._img_loader(img_path)
        image = self.transform(raw_pil_img)
        example = {'index': i, 
                   'image': image,
                   'label': int(label)}
        return example

    def __getitem__(self, i):
        example = self.get_example(i)
        return example

class ImageNetFeature(ImageNet):
    def __init__(self, img_root=None, meta=None, fea=None,
                 img_size=256, crop_size=256, 
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], 
                 split='train'):
        super().__init__(img_root, meta, 
                         img_size, crop_size, mean, std, split)
        with open(fea, 'rb') as f:
            self.img_id_to_feature = pickle.load(f)['img_id_to_feature']

    def __getitem__(self, i):
        example = self.get_example(i)
        example['feature'] = self.img_id_to_feature[i]
        return example
