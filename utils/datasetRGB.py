from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class ImageFolder(Dataset):

    def __init__(self, root, transform=None, split='train'):
        splitdir = Path(root) / split
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert('RGB')
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)