import time 
import glob
import argparse
from process_data import load_cryo, resize_cryo, load_monuseg
from augmentations import apply_aug
from PIL import Image
from resize_and_patching import patch_imgs
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

start_time = time.process_time()

def main(args):    
    cryo_images, cryo_annotations = load_cryo()
    cryo_images, cryo_annotations = resize_cryo(cryo_images=cryo_images, cryo_annotations=cryo_annotations)
    # print(cryo_annotations[0])
   
    monu_images, monu_annotations = load_monuseg()

    # original images here
    all_imgs = cryo_images + monu_images
    all_masks = cryo_annotations + monu_annotations
    assert len(all_imgs) == len(all_masks)

    data = {
        "original_image": all_imgs,
        "mask": all_masks,
        'aug_images': [],
        'aug_masks': [],
        'all_patched_img': [],
        'all_patched_msk': [],
        'train_patched_images': [],
        'train_patched_masks': [],
        'val_patched_images': [],
        'val_patched_masks': [],
        'test_patched_images': [],
        'test_patched_masks': []
    }

    aug_images, aug_masks = apply_aug(all_imgs, all_masks)
    data['aug_images'].extend(aug_images)
    data['aug_masks'].extend(aug_masks)

    aug_patch_images, aug_patch_masks = patch_imgs(data['aug_images'], data['aug_masks'], args.patch_size)
    data['all_patched_img'].extend(aug_patch_images)
    data['all_patched_msk'].extend(aug_patch_masks)


    orig_patch_images, orig_mask_images = patch_imgs(data['original_image'], data['mask'], args.patch_size)

    data['all_patched_img'].extend(orig_patch_images)
    data['all_patched_msk'].extend(orig_mask_images)



    total_patch_images = len(data['all_patched_img'])
    train_cutoff= int(0.85 * total_patch_images)
    val_cutoff = int(0.93 * total_patch_images)

    data['train_patched_images'] = data['all_patched_img'][:train_cutoff]
    data['train_patched_masks'] = data['all_patched_msk'][:train_cutoff]
    data['val_patched_images'] = data['all_patched_img'][train_cutoff:val_cutoff] 
    data['val_patched_masks'] = data['all_patched_msk'][train_cutoff:val_cutoff] 
    data['test_patched_images'] = data['all_patched_img'][val_cutoff:]
    data['test_patched_masks'] = data['all_patched_msk'][val_cutoff:] 

    print(len(data['train_patched_images']))
    print(len(data['val_patched_images']))
    print(len(data['test_patched_images']))

    end_time = time.process_time()
    print(float(end_time-start_time))

    # np.save('./all_data.npy', data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocesseing images and masks')
    parser.add_argument('--patch_size', type=int, default=256, help='Choose a patch_size')
    args = parser.parse_args()

    print('Preprocessing Data...')
    print('#' * 60)

    main(args)
