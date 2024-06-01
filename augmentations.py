import os
os.environ['ALBUMENTATIONS_DISABLE_UPDATE_CHECK'] = '1'

import albumentations as A
from albumentations.pytorch import ToTensorV2

def HorizontalFlip():
    return A.Compose([
        A.Resize(1000, 1000),
        A.HorizontalFlip(p=1),
    ])

def VerticalFlip():
    return A.Compose([
        A.Resize(1000, 1000),
        A.VerticalFlip(p=1)
    ])

def Resize():
    return A.Compose({
        A.Resize(1000, 1000) 
    })


def apply_aug(image, mask):
    try:
        horiz_trans = HorizontalFlip()
        vert_trans = VerticalFlip()


        horiz_transformed = horiz_trans(image=image, mask=mask)
        vert_transformed = vert_trans(image=image, mask=mask)


        transformed_image_1 = horiz_transformed["image"]
        transformed_mask_1 = horiz_transformed["mask"]

        transformed_image_2 = vert_transformed["image"]
        transformed_mask_2 = vert_transformed["mask"]

        return [transformed_image_1, transformed_mask_1, transformed_image_2, transformed_mask_2]
    
    except Exception as e:
        print("Error")
        raise