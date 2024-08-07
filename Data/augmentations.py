import os
os.environ['ALBUMENTATIONS_DISABLE_UPDATE_CHECK'] = '1'
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from typing import List, Tuple


def HorizontalFlip():
    return A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=1.0),
        #ToTensorV2()
    ], is_check_shapes=False)  # Disable shape check

def VerticalFlip():
    return A.Compose([
        A.Resize(1024, 1024),
        A.VerticalFlip(p=.75),
        #ToTensorV2()
    ])

def RandomCrop():
    return A.Compose([
        A.Resize(1024, 1024),
        A.RandomCrop(height=200, width=200, p=.75),
        A.Resize(1024,1024)
    ])

def Brightness():
    return A.Compose([
        A.Resize(1024,1024),
        A.ColorJitter(brightness=(0.8, 1.2), p=1.0)
    ])

def Contrast():
    return A.Compose([
        A.Resize(1024, 1024),
        A.ColorJitter(contrast=.5, hue=0.8, p=1.0)
    ])

def Gaussian_Noise():
    return A.Compose([
        A.Resize(1024,1024),
        A.GaussNoise(var_limit=(10.0,50.0), p=1.0)
    ])

# def Affine():
#     return A.Compose([
#         A.RandomAffine()
#     ])

'''-> Tuple[np.ndarray, np.ndarray]'''

def apply_aug(images: np.ndarray, masks: np.ndarray):

    horiz_trans = HorizontalFlip()
    vert_trans = VerticalFlip()
    random_crop = RandomCrop()
    brightness = Brightness()
    contrast = Contrast()
    gaussian_noise = Gaussian_Noise()

    all_augs = [horiz_trans,vert_trans,random_crop, brightness, contrast, gaussian_noise]

    aug_images = []
    aug_masks = []
    
    for aug in all_augs:
        aug_result = [aug(image=image, mask=mask) for image, mask in zip(images, masks)]
        aug_images.extend(aug['image'] for aug in aug_result)
        aug_masks.extend(aug['mask'] for aug in aug_result)
    return aug_images, aug_masks

# def test():
#     image = np.random.rand(1024, 1024, 3) * 255 
#     mask = np.random.rand(1024, 1024) * 255  

#     image = image.astype(np.uint8)
#     mask = mask.astype(np.uint8)

#     aug_images, aug_masks = apply_aug([image], [mask])
#     print(f'Number of augmented images: {len(aug_images)}')
#     print(f'Number of augmented masks: {len(aug_masks)}')

#     # Example of checking shapes
#     print(f'Augmented image shape: {aug_images[0].shape}')
#     print(f'Augmented mask shape: {aug_masks[0].shape}')

