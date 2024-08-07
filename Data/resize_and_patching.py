import torch
import torch.nn.functional as F
import numpy as np
import argparse
from typing import List, Tuple

def split_image_into_patches(image, patch_size: int):
    # print(f'image before patching: {image.shape} and patch size: {patch_size}')
    # Check if the image dimensions are divisible by the patch size
    assert image.shape[0] % patch_size == 0, "Image height must be divisible by the patch size"
    assert image.shape[1] % patch_size == 0, "Image width must be divisible by the patch size"
    
    m, n, _ = image.shape

    patches = image.reshape(m // patch_size, patch_size, n // patch_size, patch_size, -1)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, image.shape[-1])
    # print(f'Patches shape: {patches.shape}')
    return patches

def split_mask_into_patches(mask, patch_size: int):
    # print(f'mask before patching: {mask.shape} and patch size: {patch_size}')
    assert mask.shape[0] % patch_size == 0, "Mask height must be divisible by the patch size"
    assert mask.shape[1] % patch_size == 0, "Mask width must be divisible by the patch size"

    m, n = mask.shape
    patches = mask.reshape(m // patch_size, patch_size, n // patch_size, patch_size)
    patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)

    # print(f'Mask patches shape: {patches.shape}')
    return patches

def patch_imgs(images: np.ndarray, masks: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    all_img_patches = []
    all_mask_patches = []
    
    # Ensure inputs are lists of images and masks
    if not isinstance(images, list):
        images = [images]
    if not isinstance(masks, list):
        masks = [masks]
    
    for img, mask in zip(images, masks):
        # print(f'Processing image with shape: {img.shape}')
        # print(f'Processing mask with shape: {mask.shape}')
        img_patches = split_image_into_patches(img, patch_size)
        mask_patches = split_mask_into_patches(mask, patch_size)
        
        all_img_patches.extend(img_patches)
        all_mask_patches.extend(mask_patches)

    img_patches = np.array(all_img_patches)
    mask_patches = np.array(all_mask_patches)

    return img_patches, mask_patches

# Example usage
# image = np.random.rand(1024, 1024, 3)  # Example large color image
# mask = np.random.rand(1024, 1024)  # Example large mask

# print(f"Initial image shape: {image.shape}")
# print(f"Initial mask shape: {mask.shape}")

# img_patches, mask_patches = patch_imgs(image, mask, 256)
# print(f'Image patches shape: {img_patches.shape}')
# print(f'Mask patches shape: {mask_patches.shape}')
