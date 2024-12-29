import time 
import glob
import argparse
from process_data import load_pannuke
from augmentations import apply_aug
import matplotlib.patches as mpatches
from matplotlib.legend import Legend

from PIL import Image
from resize_and_patching import patch_imgs
import numpy as np
from tqdm import tqdm
import os
# ln -s ../gpt_utils.py gpt_utils.py
from gpt_utils import ensure_directory_exists
import matplotlib.pyplot as plt

start_time = time.process_time()

def main():    
    print("Starting Data Load")
    images, masks = load_pannuke()
    print('Data loaded')

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
    
    data['original_image'].extend(images)
    print("Length of images_list:", len(data['original_image']))
    print("Shape of first element in images_list:", data['original_image'][0].shape)
    data['mask'].extend(masks)
    print("Length of mask_list:", len(data['mask']))
    print("Shape of first element in mask_list:", data['mask'][0].shape)
    
    # aug_images, aug_masks = apply_aug(data['original_image'], data['mask'])

    # print(f'length of aug_images: {len(aug_images)}')

    # data['aug_images'].extend(aug_images) # data['original_image'] 
    # data['aug_masks'].extend(aug_masks) # data['mask'] 
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
    print(f'validation_cutoff: {val_cutoff}')
    
    data['train_patched_images'] = data['all_images'][:train_cutoff]
    data['train_patched_masks'] = data['all_masks'][:train_cutoff]
    data['val_patched_images'] = data['all_images'][train_cutoff:val_cutoff] 
    data['val_patched_masks'] = data['all_masks'][train_cutoff:val_cutoff] 
    data['test_patched_images'] = data['all_images'][val_cutoff:]
    data['test_patched_masks'] = data['all_masks'][val_cutoff:] 

    print(len(data['train_patched_images']))
    print(len(data['val_patched_images']))
    print(len(data['test_patched_images']))

    def xlstm_load(data):
        ensure_directory_exists('./Data/images_npy/')
        ensure_directory_exists('./Data/masks_npy/')
        for idx, (image, mask) in enumerate(zip(data['all_images'], data['all_masks'])):
            image_path = f'./Data/images_npy/image_{idx}'
            mask_path = f'./Data/masks_npy/mask_{idx}'

            np.save(image_path, image)
            np.save(mask_path, mask)
            print('Done loading')
            # input()
        
        return None

    def clip_load(data, gpt_load=bool):
        if gpt_load:
            print("Saving images, masks, and generating text files...")
            ensure_directory_exists('./Data/masks/')
            ensure_directory_exists('./Data/images/')
            ensure_directory_exists('./Data/joined_images/')
            ensure_directory_exists('./Data/texts/')

            for idx, (image, mask) in enumerate(zip(data['all_images'], data['all_masks'])):
                image_path = f'./Data/images/image_{idx}.png'
                mask_path = f'./Data/masks/mask_{idx}.png'
                joined_path = f'./Data/joined_images/join_{idx}.png'
                text_path = f'./Data/texts/text_{idx}.txt'

                # img = Image.fromarray((image * 255).astype(np.uint8))  # Assuming images are normalized between 0-1
                # img = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
                if image.max() > 1.0:  # If image is in range [0, 255], normalize to [0, 1]
                    image = image / 255.0
                if image.max() <= 1.0:  # If already in range [0, 1], scale to [0, 255]
                    image = (image * 255).astype(np.uint8)
                img = Image.fromarray(image)  # Convert to PIL Image
                img.save(image_path)

                mask = mask.astype(np.uint8) 

                colors = [
                    (0, 255, 255),     # Class 5 - Cyan
                    (0, 0, 0),         # Class 0 - Black
                    (255, 0, 0),       # Class 1 - Red
                    (0, 255, 0),       # Class 2 - Green
                    (0, 0, 255),       # Class 3 - Blue
                    (255, 255, 0),     # Class 4 - Yellow
                ]
                # ----------------------------
                squeezed_mask = mask.squeeze(-1)
                colored_mask = np.zeros((*squeezed_mask.shape, 3), dtype=np.uint8)
                for num in range(6):
                    colored_mask[squeezed_mask == num] = colors[num]  # Assign colors to corresponding mask classes
                plt.figure(figsize=(8, 8))  # Adjust the figure size if necessary
                plt.imshow(colored_mask)
                plt.axis('off')  # Turn off axis
                plt.tight_layout()  # Adjust layout

                plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)  # Save mask
                plt.close()  # Close the figure to avoid overlapping in the next iteration

                # ----------------------------
                colors = [(r/255, g/255, b/255) for r, g, b in colors]  # Example color list
                legend_labels = ['Background', 'Neoplastic cells', 'Inflammatory', 'Connective/Soft tissue cells', 'Dead Cells', 'Epithelial']

                fig, axes = plt.subplots(1, 2, figsize=(10, 5.75))  # 1 row, 2 columns

                # Display the images
                axes[0].imshow(img, aspect='auto')
                axes[0].set_title('Original Whole Slide Image', fontsize=14)
                axes[0].axis('off')
                axes[0].grid(False)

                axes[1].imshow(colored_mask, aspect='auto')
                axes[1].set_title('Instance Segmentation Mask', fontsize=14)
                axes[1].axis('off')
                axes[1].grid(False)

                legend_patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(colors))]
                fig.legend(handles=legend_patches, loc='lower center', ncol=6, title='Classes', fontsize=9.5, bbox_to_anchor=(0.5, -0.))

                plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.02)  

                plt.savefig(joined_path, bbox_inches='tight', dpi=300)
                plt.show()
                # ----------------------------

                with open(text_path, 'w') as file:
                    file.write(f"Placeholder_text_for_image_{idx}.png")  # Replace with GPT-generated text if available

                print(f"Saved image_{idx}.png, mask_{idx}.png, join_{idx}.png, and text_{idx}.txt")

            print("All images, masks, and text files saved.")
        
        else:
            print('Not saving images, masks for CLIP')

    end_time = time.process_time()
    print(float(end_time-start_time))
    xlstm_load(data)
    # clip_load(data, gpt_load=False)
    # np.save('./pannuke_6c', data)

if __name__ == "__main__":
    print('Preprocessing Data...')
    print('#' * 60)
    main()

