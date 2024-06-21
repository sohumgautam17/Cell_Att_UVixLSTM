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

    horiz_trans = HorizontalFlip()
    vert_trans = VerticalFlip()
    random_crop = RandomCrop()
    brightness = Brightness()
    contrast = Contrast()
    gaussian_noise = Gaussian_Noise()

    data = {
        "original_image": all_imgs,
        "mask": all_masks,
        'horiz_images': [],
        'horiz_masks': [],
        'vert_images': [],
        'vert_masks': [],
        'crop_images': [],
        'crop_masks': [],
        'bright_images': [],
        'bright_masks': [],
        'cont_images': [],
        'cont_masks': [],
        'gauss_images': [],
        'gauss_masks': [],
        'patched_orig_images': [],
        'patched_orig_masks': [],
        'patched_horiz_images': [],
        'patched_horiz_masks': [],
        'patched_vert_images': [],
        'patched_vert_masks': [],
        'patched_crop_images': [],
        'patched_crop_masks': [],
        'patched_bright_images': [],
        'patched_bright_masks': [],
        'patched_cont_images': [],
        'patched_cont_masks': [],
        'patched_gauss_images': [],
        'patched_gauss_masks': []
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

        data['horiz_images'].append(horiz_image)
        data['horiz_masks'].append(horiz_mask)
        data['vert_images'].append(vert_image)
        data['vert_masks'].append(vert_mask)
        data['crop_images'].append(crop_image)
        data['crop_masks'].append(crop_mask)
        data['bright_images'].append(bright_image)
        data['bright_masks'].append(bright_mask)
        data['cont_images'].append(cont_image)
        data['cont_masks'].append(cont_mask)
        data['gauss_images'].append(gauss_image)
        data['gauss_masks'].append(gauss_mask)

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
            data['patched_horiz_images'].append(horiz_patches[j])
            data['patched_horiz_masks'].append(horiz_mask_patches[j])
            data['patched_vert_images'].append(vert_patches[j])
            data['patched_vert_masks'].append(vert_mask_patches[j])
            data['patched_crop_images'].append(crop_patches[j])
            data['patched_crop_masks'].append(crop_mask_patches[j])
            data['patched_bright_images'].append(bright_patches[j])
            data['patched_bright_masks'].append(bright_mask_patches[j])
            data['patched_cont_images'].append(cont_patches[j])
            data['patched_cont_masks'].append(cont_mask_patches[j])
            data['patched_gauss_images'].append(gauss_patches[j])
            data['patched_gauss_masks'].append(gauss_mask_patches[j])
    
    np.save('./all_data.npy', data)


if __name__ == "__main__":
    args = get_args()
    print('Preprocessing Data...')
    print('#'*50)
    main(args)
