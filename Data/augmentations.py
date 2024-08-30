import os
os.environ['ALBUMENTATIONS_DISABLE_UPDATE_CHECK'] = '1'
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


def HorizontalFlip():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=1.0),
        #ToTensorV2()
    ], is_check_shapes=False)  # Disable shape check

def VerticalFlip():
    return A.Compose([
        A.Resize(256, 256),
        A.VerticalFlip(p=.75),
        #ToTensorV2()
    ])

def RandomCrop():
    return A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(height=200, width=200, p=.75),
        A.Resize(256,256)
    ])

def Brightness():
    return A.Compose([
        A.Resize(256,256),
        A.ColorJitter(brightness=(0.8, 1.2), p=1.0)
    ])

def Contrast():
    return A.Compose([
        A.Resize(256, 256),
        A.ColorJitter(contrast=.5, hue=0.8, p=1.0)
    ])

def Gaussian_Noise():
    return A.Compose([
        A.Resize(256,256),
        A.GaussNoise(var_limit=(10.0,50.0), p=1.0)
    ])


### These are new

def Zoom_Blur():
    return A.Compose([
        A.RandomScale(scale_limit=(0.1, 0.2), p=1.0), 
        A.MotionBlur(blur_limit=15, p=1.0),       
        A.Resize(256, 256),
    ])

def Rotate():
    return A.Compose([
        A.Resize(256, 256),
        A.Rotate(limit=(-90, 90), p=0.75),
        A.Resize(256, 256),

    ])

def Elastic_Transform():
    return A.Compose({
        A.Resize(256, 256),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        A.Resize(256, 256),

    })


def apply_aug(images, masks):

    horiz_trans = HorizontalFlip()
    vert_trans = VerticalFlip()
    random_crop = RandomCrop()
    brightness = Brightness()
    contrast = Contrast()
    gaussian_noise = Gaussian_Noise()
    zoom_blur = Zoom_Blur()
    rotate = Rotate()
    elastic = Elastic_Transform()

    all_augs = [horiz_trans,vert_trans,random_crop, brightness, contrast, gaussian_noise, zoom_blur, rotate, elastic]

    aug_images = []
    aug_masks = []
    
    for aug in all_augs:
        aug_result = [aug(image=image, mask=mask) for image, mask in tqdm(zip(images, masks), total=len(images), desc="Applying augmentations")]
        aug_images.extend(aug['image'] for aug in aug_result)
        aug_masks.extend(aug['mask'] for aug in aug_result)
    return aug_images, aug_masks


    # for image, mask in tqdm(zip(data['original_image'], data['mask']), total=len(data['original_image']), desc="Applying augmentations"):
    #     augmented_image, augmented_mask = apply_aug([image], [mask])
    #     data['aug_images'].extend(augmented_image)
    #     data['aug_masks'].extend(augmented_mask)
