import time 
import glob
import argparse
from process_data import load_cryo, resize_cryo, load_pannuke
from augmentations import apply_aug
from PIL import Image
from resize_and_patching import patch_imgs
import numpy as np
from tqdm import tqdm
# from typing import List, Tuple

start_time = time.process_time()

def main():    
    images, masks = load_pannuke()

    data = {
        "original_image": [],
        "mask": [],
        'aug_images': [],
        'aug_masks': [],
        'all_images': [],
        'all_masks': [],
        'train_patched_images': [],
        'train_patched_masks': [],
        'val_patched_images': [],
        'val_patched_masks': [],
        'test_patched_images': [],
        'test_patched_masks': []
    }
    print('hello1')

    for image in tqdm(images):
        data['original_image'].append(image)
    print("Length of images_list:", len(data['original_image']))
    print("Shape of first element in images_list:", data['original_image'][0].shape)
    data['mask'].extend(masks)
    print("Length of mask_list:", len(data['mask']))
    print("Shape of first element in mask_list:", data['mask'][0].shape)
    
    aug_images, aug_masks = apply_aug(data['original_image'], data['mask'])

    print(f'length of aug_images: {len(aug_images)}')

    data['aug_images'].extend(aug_images) # data['original_image']
    data['aug_masks'].extend(aug_masks) # data['mask']
    data['all_images'].extend(data['aug_images'])
    data['all_images'].extend(data['original_image'])
    data['all_masks'].extend(data['aug_masks'])
    data['all_masks'].extend(data['mask'])

    # print(f'Length of all images : {len(data['all_images'])}')

    total_patch_images = len(data['all_masks'])
    print(total_patch_images)
    train_cutoff = int(0.85 * total_patch_images)
    print(f'train_cutoff: {train_cutoff}')
    val_cutoff = int(0.93 * total_patch_images)
    print(f'train_cutoff: {val_cutoff}')
    
    data['train_patched_images'] = data['all_images'][:train_cutoff]
    data['train_patched_masks'] = data['all_masks'][:train_cutoff]
    data['val_patched_images'] = data['all_images'][train_cutoff:val_cutoff] 
    data['val_patched_masks'] = data['all_masks'][train_cutoff:val_cutoff] 
    data['test_patched_images'] = data['all_images'][val_cutoff:]
    data['test_patched_masks'] = data['all_masks'][val_cutoff:] 

    print(len(data['train_patched_images']))
    print(len(data['val_patched_images']))
    print(len(data['test_patched_images']))


    print(len(data['train_patched_images']))
    print(len(data['val_patched_images']))
    print(len(data['test_patched_images']))

    end_time = time.process_time()
    print(float(end_time-start_time))
    
    np.save('./pannuke_6c_augs', data)

if __name__ == "__main__":
    print('Preprocessing Data...')
    print('#' * 60)
    main()
