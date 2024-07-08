import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import os

def Resize():
    return A.Compose([
        A.Resize(3000, 3000),
    ], is_check_shapes=False)

def process(img, ):
    if not isinstance(img, np.ndarray):
        raise TypeError("Image must be a numpy array type...")

    resize = Resize() # Resizing the image may help with accuracy
    resized_img = resize(image=img)
    new_image = resized_img['image']
    new_image = np.array(new_image)
    print(f"Image datatype: {new_image.dtype}")
    return new_image


def watershed(image: np.array, lower_thresh: float, visualize: bool):
    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv.morphologyEx(gray_image, cv.MORPH_DILATE, kernel)

    disTrans = cv.distanceTransform(imgDilate, cv.DIST_L2, 5)
    _, distThresh = cv.threshold(disTrans, lower_thresh * disTrans.max(), 255, cv.THRESH_BINARY)
    # print(f'distThresh datatype: {distThresh.dtype}')

    distThresh = np.uint8(distThresh)
    _, markers = cv.connectedComponents(distThresh)

    markers = markers + 1

    # Mark the unknown region as 0
    markers[distThresh == 0] = 0

    # Apply watershed
    markers = cv.watershed(imgRGB, markers)
    unique_segments = len(np.unique(markers)) - 1 
    image[markers == -1] = [255, 0, 0]  # Mark boundaries with red

    number_cells = unique_segments
    processed_image = image

    return number_cells, processed_image

def visualize(gt_output, pred_output, gt_cells, pred_cells, args, inst): 
    gt_output = gt_output.squeeze().cpu().numpy().transpose(1, 2, 0) 
    pred_output = pred_output.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax[0].imshow(gt_output, cmap='gray')
    ax[0].set_title(f'Ground Truth Mask (Cell Count: {gt_cells})')
    ax[1].imshow(pred_output, cmap='gray')
    ax[1].set_title(f'Predicted Mask (Cell Count: {pred_cells})')
    plt.savefig(f'./runs/watershed/{args.checkpoint}/visualization_{inst}.png')



