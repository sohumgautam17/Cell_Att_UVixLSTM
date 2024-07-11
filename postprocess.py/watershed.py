import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import albumentations as A
from postprocess.watershed_utils import Resize
from postprocess.watershed_utils import process, watershed, visualize
import os

import torch
from torch.utils.data import DataLoader


def inference_watershed(model, test_loader, device, args):
    model.eval()
    
    # Want to utilize both pred_mask and gt_mask for visualization
    with torch.no_grad():
        for idx, batch in tqdm.tqdm(enumerate(test_loader, 1), desc="Watershedding"):
            img, mask, gt_img = batch
            img, mask = img.to(device), mask.to(device)
            output = model(img)

            output = torch.sigmoid(output)
            pred_output = (output > 0.5).float() # This creates a usable mask for watershed
            pred_num_cells, pred_image = watershed(gt_img[0].detach().cpu().numpy().astype(np.uint8))


            ## the visualize doesnt really make sense since we only need the watershed to annotate how many cells. this can also be done separately during preprocess
