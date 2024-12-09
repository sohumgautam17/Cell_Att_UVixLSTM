import torch

from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


import json

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
        img = (img * 255).astype(np.uint8)
        mask = self.masks[index]
        trans_img = self.transform(Image.fromarray(img))
        # print('before', np.unique(mask))
        # mask = mask-1
        # print('after', np.unique(mask))
        assert mask.min() >= 0 and mask.max() <= 5
        mask = torch.tensor(mask, dtype=torch.int64) 
        mask = mask.squeeze(2).unsqueeze(0) # from 256 256 c --> c 256 256 

        return trans_img, mask, img


class ECGCLIPPretrain(Dataset):
    def __init__(self, all_signals_path, all_texts_path, clip_tokenizer = None, processor = None, max_length=77):
        self.signals_path = glob.glob(all_signals_path)[:1727]
        self.texts_path = glob.glob(all_texts_path)[:1727]
        self.clip_tokenizer = clip_tokenizer
        self.processor = processor
        self.max_length = max_length

        assert len(self.signals_path) == len(self.texts_path), "The number of images and texts should match."

    def __len__(self):
        return len(self.signals_path)

    def __getitem__(self, idx):
        signal_path = self.signals_path[idx]
        signal_image = Image.open(signal_path).convert("RGB") # sing images are png

        with open(self.texts_path[idx], 'r') as f:
            text = f.read().strip()

        inputs_text = self.processor(text=text, return_tensors="pt", padding='max_length', 
                                     truncation=True, max_length=self.max_length)

        inputs_image = self.processor(images=signal_image, return_tensors="pt")

        inputs = {
            'input_ids': inputs_text.input_ids.squeeze(0),  # Remove batch dimension
            'attention_mask': inputs_text.attention_mask.squeeze(0),  # Remove batch dimension
            'pixel_values': inputs_image.pixel_values.squeeze(0)  # Remove batch dimension
        }


        # print(inputs.values())

        return inputs
