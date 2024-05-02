from matplotlib.patches import Polygon
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

# way to get the list of files in a directory without all the for loops
list_of_annotations = glob.glob('./Annotations/*.xml')
list_of_imgs = glob.glob('./Tissue Images/*.tif')
print(len(list_of_annotations))
print(len(list_of_imgs))
assert len(list_of_annotations) == len(list_of_imgs)

# sorting part
sorted_list_of_annotations = sorted(list_of_annotations)
sorted_list_of_imgs = sorted(list_of_imgs)



def he_to_binary_mask(filename, visualize = False):
    im_file = './Tissue Images/' + filename + '.tif'
    xml_file = './Annotations/' + filename + '.xml'
    
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
        binary_mask[rr, cc] += 1  # Update binary mask
        
        # Update color mask with random colors
        color = [random.randint(0, 255) for _ in range(3)]
        for j in range(3):
            color_mask[rr, cc, j] = np.clip(color_mask[rr, cc, j] + color[j], 0, 255)
    
    # visualize masks
    if visualize:
        plt.imsave('./visualized_imgs/' + filename + '_binary.png', binary_mask, cmap='gray')
        plt.close()
        plt.imsave('./visualized_imgs/' + filename + '_color.png', color_mask)
        plt.close()

    return {
        'original_image': np.array(image),
        'binary_mask': binary_mask,
        'color_mask': color_mask
    }
    
image_annot_data_struct = {}

# Example usage:
# to assert that the files are same. 
for i in tqdm(range(len(sorted_list_of_annotations)), desc = 'Processing'):
  annot = sorted_list_of_annotations[i]
  img_path = sorted_list_of_imgs[i]
  assert Path(annot).stem == Path(img_path).stem ## you can print out Path(annot).stem to see what this operation is doing
  # essentially, it is getting the filename without the extension
  # we are checking if the filename is the same for both the annotation and the image

  # saving the output of our function to a dict
  image_annot_data_struct[Path(annot).stem] = he_to_binary_mask(Path(annot).stem)

# Now we have a dictionary with the filename as the key and the value is a dictionary with the original image, binary mask and color mask
np.save('image_annot_data_struct.npy', image_annot_data_struct)
# way to load data
data = np.load('./image_annot_data_struct.npy', allow_pickle = True).item()


