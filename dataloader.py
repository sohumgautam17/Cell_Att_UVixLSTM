import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
from utils import get_args  # Import the args object from main.py
args = get_args()

# Create custom PyTorch for files 
class CellDataset(Dataset):
    # Images and masks are stored in imgs and masks
    def __init__(self, imgs, masks, args = None):
        self.imgs = imgs
        self.masks = masks
        self.args = args
        # Random batch trasforms
        if args.augfly:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(size=(args.patch_size, args.patch_size)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
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
        trans_img = self.transform(Image.fromarray(img))

        
        if args.dataset == './Data/all_data.npy':
            mask = self.mask_transform(mask)
        else:
            mask = torch.Tensor(mask.squeeze())
            mask = mask.unsqueeze(0)
            mask = (mask > 0).float() # turns mask into 0 and 1

        # print(mask)
        # print(np.unique(mask))
        # input('testing')

        return trans_img, mask, img
