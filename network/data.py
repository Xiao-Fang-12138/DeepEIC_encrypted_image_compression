"""RGB datasets with the original crop and padding conventions."""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .runtime import local_path


class Images(Dataset):
    def __init__(self, folder, crop=0, training=False):
        self.root = local_path(folder)
        self.files = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
        if not self.files:
            raise ValueError(f"No images found in {folder}; place RGB images inside this release")
        for path in self.files:
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Dataset symlinks must not leave the dataset directory")
        operations = []
        if training:
            operations = [transforms.RandomCrop(crop), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        elif crop:
            operations = [transforms.CenterCrop(crop)]
        self.transform = transforms.Compose(operations + [transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.relative_to(self.root).as_posix()
