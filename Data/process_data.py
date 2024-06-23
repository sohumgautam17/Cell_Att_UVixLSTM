from matplotlib.patches import Polygon
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import requests
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from torch import nn
from tqdm import tqdm
from skimage.draw import polygon2mask
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import glob
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
import torchvision.transforms as T


# way to get the list of files in a directory without all the for loops
list_of_annotations = glob.glob('./MoNuSeg_Annotations/*.xml')
list_of_imgs = glob.glob('./MoNuSeg_Images/*.tif')
# print(len(list_of_annotations))
# print(len(list_of_imgs))
assert len(list_of_annotations) == len(list_of_imgs)

# sorting part
sorted_list_of_annotations = sorted(list_of_annotations)
sorted_list_of_imgs = sorted(list_of_imgs)

def he_to_binary_mask(filename, visualize = False):
    im_file = '../MoNuSeg_Images/' + filename + '.tif'
    xml_file = '../MoNuSeg_Annotations/' + filename + '.xml'

    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    regions = root.findall('.//Region')

    xy = []
    for region in regions:
        vertices = region.findall('.//Vertex')
        coords = np.zeros((len(vertices), 2))
        for i, vertex in enumerate(vertices):
            x = float(vertex.get('X'))
            y = float(vertex.get('Y'))
            coords[i] = [x, y]
        xy.append(coords)

    # Load image
    image = Image.open(im_file)
    nrow, ncol = image.size

    binary_mask = np.zeros((nrow, ncol), dtype=np.uint8)
    color_mask = np.zeros((nrow, ncol, 3), dtype=np.uint8)

    for i, coords in enumerate(xy):
        rr, cc = polygon2mask((nrow, ncol), coords).nonzero()
        binary_mask[rr, cc] += 255  # Update binary mask

        # Update color mask with random colors (If we do not use we can comment)
        # color = [random.randint(0, 255) for _ in range(3)]
        # for j in range(3):
        #     color_mask[rr, cc, j] = np.clip(color_mask[rr, cc, j] + color[j], 0, 255)

    # visualize masks
    if visualize:
        plt.imsave('./visualized_imgs/' + filename + '_binary.png', binary_mask, cmap='gray')
        plt.close()
        # plt.imsave('./visualized_imgs/' + filename + '_color.png', color_mask)
        # plt.close()

    return {
        'original_image': np.array(image),
        'binary_mask': binary_mask,
        # 'color_mask': color_mask
    }

# image_annot_data_struct = {}


def load_monuseg():
    files_path = "../MoNuSeg_Images"
    # filenames = []
    imgs = []
    binary_masks = []

    for each_file in tqdm(os.listdir(files_path), desc='Loading MoNuSeg Images'):
        each_file = each_file[:-4]
        data = he_to_binary_mask(each_file)
        image_array = np.array(data['original_image'])
        img_flip_ud = cv2.flip(image_array, 0)
        img_rotated = np.rot90(img_flip_ud, k=3)
        imgs.append(img_rotated)
        binary_masks.append(data['binary_mask'])

    return imgs, binary_masks



###____________________________________________________________________________________________###

cyro_annotations = glob.glob('./Cryo_Annotater_1/*.png')
cryo_images = glob.glob('./CryoNuSeg_Images/*.tif')

sorted_cryo_annotations = sorted(cyro_annotations)
sorted_cryo_image = sorted(cryo_images)

assert len(sorted_cryo_annotations) == len(sorted_cryo_image)

def pdf_to_binary(image):
    storing_list = []

    # Load image
    img = cv2.imread(image)
    storing_list.append(img)

    return storing_list



def load_cryo():
    image_path = '../CryoNuSeg_Images'
    annotations_path = '../Cryo_Annotater_1'

    image_array = []
    annotation_array = []

    # Get lists of image and annotation file paths
    image_files = sorted(os.listdir(image_path))
    annotation_files = sorted(os.listdir(annotations_path))

    # Iterate over both lists simultaneousl using zip()
    for image_file in image_files:
        image_path_full = os.path.join(image_path, image_file)
        image_numpy = pdf_to_binary(image_path_full)
        image_array.append(np.squeeze(image_numpy))

    for annotation_file in annotation_files:
        annotation_path_full = os.path.join(annotations_path, annotation_file)
        annotation_numpy = pdf_to_binary(annotation_path_full)
        annotation_array.append(np.squeeze(annotation_numpy))

    return image_array, annotation_array

def resize_cryo(cryo_images, cryo_annotations):
    resized_img_array = []
    resized_mask_array = []

    for image, mask in tqdm(zip(cryo_images, cryo_annotations), desc = 'Loading Cryo Images') :
        pil_image = Image.fromarray(image)
        resized_img = T.Resize(size=(1000,1000))(pil_image)
        resized_img = np.array(resized_img)
        resized_img_array.append(resized_img)

        pil_mask = Image.fromarray(mask).convert("L")
        resized_mask = T.Resize(size=(1000, 1000), interpolation=Image.NEAREST)(pil_mask)
        resized_mask = np.array(resized_mask)
        resized_mask_array.append(resized_mask)

    return resized_img_array, resized_mask_array