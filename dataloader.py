import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class CellDataset(Dataset):
    def __init__(self, imgs, masks, args = None):
        self.imgs = imgs
        self.masks = masks
        self.args = args
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]
        img = self.transform(img)
        mask = self.mask_transform(mask)
        
        return img, mask
