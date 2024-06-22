import os
os.environ['ALBUMENTATIONS_DISABLE_UPDATE_CHECK'] = '1'

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

def HorizontalFlip():
    return A.Compose([
        A.Resize(1000, 1000),
        A.HorizontalFlip(p=1.0),
        #ToTensorV2()
    ], is_check_shapes=False)  # Disable shape check

def VerticalFlip():
    return A.Compose([
        A.Resize(1000, 1000),
        A.VerticalFlip(p=.75),
        #ToTensorV2()
    ])

def RandomCrop():
    return A.Compose([
        A.Resize(1000, 1000),
        A.RandomCrop(height=200, width=200, p=.75),
        A.Resize(1000,1000)
    ])

def Brightness():
    return A.Compose([
        A.Resize(1000,1000),
        A.ColorJitter(brightness=(0.8, 1.2), p=1.0)
    ])

def Contrast():
    return A.Compose([
        A.Resize(1000, 1000),
        A.ColorJitter(contrast=.5, hue=0.8, p=1.0)
    ])

def Gaussian_Noise():
    return A.Compose([
        A.Resize(1000,1000),
        A.GaussNoise(var_limit=(10.0,50.0), p=1.0)
    ])

# def Affine():
#     return A.Compose([
#         A.RandomAffine()
#     ])


def apply_aug(image, mask):
    # # Ensure image and mask are converted to uint8 and maintain dimensions
    # if image.ndim == 2:
    #     image = np.expand_dims(image, axis=-1)
    # if mask.ndim == 2:
    #     mask = np.expand_dims(mask, axis=-1)

    # # Ensure image has 3 channels and mask has 1 channel
    # if image.shape[-1] == 1:
    #     image = np.repeat(image, 3, axis=-1)
    # if mask.shape[-1] == 1:
    #     mask = mask[..., 0]

    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    
    try:
        horiz_trans = HorizontalFlip()
        vert_trans = VerticalFlip()
        random_crop = RandomCrop()
        brightness = Brightness()
        contrast = Contrast()
        gaussian_noise = Gaussian_Noise()


        horiz_transformed = horiz_trans(image=image, mask=mask)
        vert_transformed = vert_trans(image=image, mask=mask)
        random_crop = random_crop(image=image, mask=mask)
        brightness = brightness(image=image, mask=mask)
        contrast = contrast(image=image, mask=mask)
        gaussian_noise = gaussian_noise(image=image, mask=mask)



        transformed_image_1 = horiz_transformed["image"]
        transformed_mask_1 = horiz_transformed["mask"]

        transformed_image_2 = vert_transformed["image"]
        transformed_mask_2 = vert_transformed["mask"]

        transformed_image_3 = random_crop["image"]
        transformed_mask_3 = random_crop["mask"]

        transformed_image_4 = brightness["image"]
        transformed_mask_4 = brightness["mask"]

        transformed_image_5 = contrast["image"]
        transformed_mask_5 = contrast["mask"]

        transformed_image_6 = gaussian_noise["image"]
        transformed_mask_6 = gaussian_noise["mask"]


        return {
            "images": [image, transformed_image_1, transformed_image_2, transformed_image_3,
                       transformed_image_4, transformed_image_5, transformed_image_6],
            "masks": [mask, transformed_mask_1, transformed_mask_2, transformed_mask_3,
                      transformed_mask_4, transformed_mask_5, transformed_mask_6]
        }

    except Exception as e:
        print("Error:", e)
        raise