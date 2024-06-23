import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


# Create custom PyTorch for files 
class CellDataset(Dataset):
    # Images and masks are stored in imgs and masks
    def __init__(self, imgs, masks, args = None):
        self.imgs = imgs
        self.masks = masks
        self.args = args
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.ToTensor()
        
    # Return number of samples in the dataset
    def __len__(self):
        return len(self.imgs)

    # Load a sample at a given index
    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]
        img = self.transform(img)
        mask = self.mask_transform(mask)
        
        return img, mask
