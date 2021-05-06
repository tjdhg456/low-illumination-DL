from torchvision.transforms import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from random import shuffle
import torch
from glob import glob
from PIL import Image

## Paired LOL dataset
class LOL_dataset(Dataset):
    def __init__(self, option, transform, type):
        self.data_dir = os.path.join(option.result['data_dir'], type)
        self.data_name = [path.split('/')[-1] for path in glob(os.path.join(self.data_dir, 'high', '*.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        high_img = self.load_image(path=os.path.join(self.data_dir, 'high', self.data_name[index]))
        low_img = self.load_image(path=os.path.join(self.data_dir, 'low', self.data_name[index]))
        return high_img, low_img

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img