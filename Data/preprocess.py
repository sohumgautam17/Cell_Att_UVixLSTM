import glob
import argparse
from process_data import load_cryo, resize_cryo, load_monuseg
from augmentations import HorizontalFlip, Brightness, Contrast, Gaussian_Noise, RandomCrop, VerticalFlip
from PIL import Image
from resize_and_patching import split_image_into_patches, split_mask_into_patches
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description=None)
    return parser.parse_args()

def main(args):
    # list_of_annotations_mon = glob.glob('../MoNuSeg_Annotations/*.xml')
    # list_of_imgs_mon = glob.glob('../MoNuSeg_Images/*.tif')
    # print('Length of MoNuSeg Annotations', len(list_of_annotations_mon))
    # print('Length of MoNuSeg Imgs', len(list_of_imgs_mon))
    # assert len(list_of_annotations_mon) == len(list_of_imgs_mon)
    # list_of_annotations_cryo = glob.glob('../Cryo_Annotater_1/*.png')
    # list_of_imgs_cryo = glob.glob('../CryoNuSeg_Images/*.tif')
    # print('Length of Cryo Annotations', len(list_of_annotations_cryo))
    # print('Length of Cryo Imgs', len(list_of_imgs_cryo))
    # assert len(list_of_annotations_cryo) == len(list_of_imgs_cryo)
    cryo_images, cryo_annotations = load_cryo()
    cryo_images, cryo_annotations = resize_cryo(cryo_images=cryo_images, cryo_annotations=cryo_annotations)
    print(len(cryo_images))
    print(len(cryo_annotations))
    print(cryo_images[0].shape)
    print(cryo_annotations[0].shape)

    monu_images, monu_annotations = load_monuseg()
    print(len(monu_images))
    print(len(monu_annotations))
    print(monu_images[0].shape)
    print(monu_annotations[0].shape)

    all_imgs = cryo_images + monu_images
    all_masks = cryo_annotations + monu_annotations
    assert len(all_imgs) == len(all_masks)

    horiz_trans = HorizontalFlip()
    vert_trans = VerticalFlip()
    random_crop = RandomCrop()
    brightness = Brightness()
    contrast = Contrast()
    gaussian_noise = Gaussian_Noise()

    data = {
        "original_image": all_imgs,
        "mask": all_masks,
        'aug_images': [],
        'aug_masks': [],
        'patched_orig_images': [],
        'patched_orig_masks': [],
        'patched_aug_images': [],
        'patched_aug_masks': [],
        'train_patched_images': [],
        'train_patched_masks': [],
        'val_patched_images': [],
        'val_patched_masks': [],
        'test_patched_images': [],
        'test_patched_masks': []
    }

    for i in tqdm(range(len(all_imgs)), desc = 'Augmenting and Patching Data'):
        
        # Geometric
        horiz = horiz_trans(image = all_imgs[i], mask = all_masks[i])
        vert = vert_trans(image = all_imgs[i], mask = all_masks[i])
        crop = random_crop(image = all_imgs[i], mask = all_masks[i])

        # Intensity
        bright = brightness(image = all_imgs[i], mask = all_masks[i])
        cont = contrast(image = all_imgs[i], mask = all_masks[i])
        gauss = gaussian_noise(image = all_imgs[i], mask = all_masks[i])

        orig_image = all_imgs[i]
        orig_mask = all_masks[i]
        horiz_image = horiz["image"]
        horiz_mask = horiz["mask"]
        vert_image = vert["image"]
        vert_mask = vert["mask"]
        crop_image = crop["image"]
        crop_mask = crop["mask"]
        bright_image = bright["image"]
        bright_mask = bright["mask"]
        cont_image = cont["image"]
        cont_mask = cont["mask"]
        gauss_image = gauss["image"]
        gauss_mask = gauss["mask"]

        data['aug_images'].append(horiz_image)
        data['aug_masks'].append(horiz_mask)
        data['aug_images'].append(vert_image)
        data['aug_masks'].append(vert_mask)
        data['aug_images'].append(crop_image)
        data['aug_masks'].append(crop_mask)
        data['aug_images'].append(bright_image)
        data['aug_masks'].append(bright_mask)
        data['aug_images'].append(cont_image)
        data['aug_masks'].append(cont_mask)
        data['aug_images'].append(gauss_image)
        data['aug_masks'].append(gauss_mask)

        orig_patches = split_image_into_patches(orig_image)
        orig_mask_patches = split_mask_into_patches(orig_mask)
        horiz_patches = split_image_into_patches(horiz_image)
        horiz_mask_patches = split_mask_into_patches(horiz_mask)
        vert_patches = split_image_into_patches(vert_image)
        vert_mask_patches = split_mask_into_patches(vert_mask)
        crop_patches = split_image_into_patches(crop_image)
        crop_mask_patches = split_mask_into_patches(crop_mask)
        bright_patches = split_image_into_patches(bright_image)
        bright_mask_patches = split_mask_into_patches(bright_mask)
        cont_patches = split_image_into_patches(cont_image)
        cont_mask_patches = split_mask_into_patches(cont_mask)
        gauss_patches = split_image_into_patches(gauss_image)
        gauss_mask_patches = split_mask_into_patches(gauss_mask)

        for j in range(len(orig_patches)):
            data['patched_orig_images'].append(orig_patches[j])
            data['patched_orig_masks'].append(orig_mask_patches[j])
            data['patched_aug_images'].append(horiz_patches[j])
            data['patched_aug_masks'].append(horiz_mask_patches[j])
            data['patched_aug_images'].append(vert_patches[j])
            data['patched_aug_masks'].append(vert_mask_patches[j])
            data['patched_aug_images'].append(crop_patches[j])
            data['patched_aug_masks'].append(crop_mask_patches[j])
            data['patched_aug_images'].append(bright_patches[j])
            data['patched_aug_masks'].append(bright_mask_patches[j])
            data['patched_aug_images'].append(cont_patches[j])
            data['patched_aug_masks'].append(cont_mask_patches[j])
            data['patched_aug_images'].append(gauss_patches[j])
            data['patched_aug_masks'].append(gauss_mask_patches[j])
    
    for i in range(len(data['patched_orig_images'])):
        if i < 0.4*len(data['patched_orig_images']):
            data['train_patched_images'].append(data['patched_orig_images'][i])
            data['train_patched_masks'].append(data['patched_orig_masks'][i])
        elif i < 0.5*len(data['patched_orig_images']):
            data['val_patched_images'].append(data['patched_orig_images'][i])
            data['val_patched_masks'].append(data['patched_orig_masks'][i])
        else:
            data['test_patched_images'].append(data['patched_orig_images'][i])
            data['test_patched_masks'].append(data['patched_orig_masks'][i])
    
    for i in range(len(data['patched_aug_images'])):
        if i < 0.8*len(data['patched_aug_images']):
            data['train_patched_images'].append(data['patched_aug_images'][i])
            data['train_patched_masks'].append(data['patched_aug_masks'][i])
        else:
            data['val_patched_images'].append(data['patched_aug_images'][i])
            data['val_patched_masks'].append(data['patched_aug_masks'][i])
        
    np.save('./all_data.npy', data)


if __name__ == "__main__":
    args = get_args()
    print('Preprocessing Data...')
    print('#'*50)
    main(args)
