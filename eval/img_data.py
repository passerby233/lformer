import os, pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class Rawdataset(Dataset):
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
        img = Image.open(self.file_names[index]).convert('RGB')
        # Convert image and label to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_filenames(self, data_path):
        def is_image_file(filename):
            if filename.rfind('jpg') != -1 or filename.rfind('png') != -1:
                return True
        images = [os.path.join(data_path, x) for x in os.listdir(data_path) if is_image_file(x)]
        return images


class IndexedSet(Dataset):
    def __init__(self, img_dir, transform=None, meta_path=None, with_cap=False):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.with_cap = with_cap
        if meta_path is None:
            meta_path = os.path.join(img_dir, 'meta.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    self.meta = pickle.load(f)
            else:
                filenames = self.get_filenames(img_dir)
                self.meta = [(filename, "") for filename in filenames]

    def get_filenames(self, data_path):
        def is_image_file(filename):
            if filename.rfind('jpg') != -1 or filename.rfind('png') != -1:
                return True
        images = [os.path.join(data_path, x) for x in os.listdir(data_path) if is_image_file(x)]
        return images

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        filename, caption = self.meta[index]
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.with_cap:
            return {'img': img, 'caption': caption}
        else:
            return img