import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class VariableSizeDataset(Dataset):

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (img, self.img_list[idx])

def variable_collate(batch):
    assert len(batch) == 1, '本实现仅支持 batch_size=1'
    return batch[0]
if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = VariableSizeDataset(img_dir='G:/dataset/urban100/bicubic_4x/val/HR', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=variable_collate, pin_memory=True)
    for (img_tensor, filename) in dataloader:
        print(f'图像 {filename} 的尺寸: {img_tensor.shape}')