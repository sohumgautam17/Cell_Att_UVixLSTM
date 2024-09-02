from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def one_hot_encoding(tensor, num_classes):
    """ Convert tensor to one-hot encoding. """
    tensor = tensor.to(torch.int64)  # Ensure tensor is of type int64
    shape = list(tensor.shape)
    shape.append(num_classes)
    one_hot = torch.zeros(shape, device=tensor.device)
    # Expand tensor dimensions and use scatter_
    one_hot.scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot


## Functionized Training, Val, and Test Loops

def trainer(model, train_loader, optimizer, device, args, dc_loss, bce_loss, jacc_loss, focal_loss):
    model.train()
    losses = 0
    len_of_batch = 0
    
    for batch in tqdm(train_loader, desc = 'Training'):
        optimizer.zero_grad()
        img, mask, _ = batch
        img, mask = img.to(device), mask.to(device)
        # print(img)        
        ####vizualising 
        # img_cpu = img[0].cpu().numpy()  # Move the tensor to the CPU and convert it to a NumPy array
        # img_cpu = img_cpu.transpose(1, 2, 0)  # Rearrange dimensions if necessary (C, H, W) -> (H, W, C)
        # plt.imshow(img_cpu)
        # plt.axis('off')  # Turn off axis
        # plt.savefig(f'./train_img.png', bbox_inches='tight', pad_inches=0)
        # plt.close()  # Close the figure to free memory

        # mask_cpu = mask[0].cpu().numpy()
        # mask_cpu = mask_cpu.transpose(1, 2, 0)  # Rearrange dimensions if necessary (C, H, W) -> (H, W, C)
        # plt.imshow(mask_cpu)
        # plt.axis('off')
        # plt.savefig(f'./train_mask.png', bbox_inches='tight', pad_inches=0)
        # plt.close()  # Close the figure to free memory
        ####

        output = model(img)
        if args.loss == 'dice':
            loss_value = dc_loss(output, mask)
        elif args.loss == 'bce':
            loss_value = bce_loss(output, mask)
        elif args.loss == 'all':
            loss_value = dc_loss(output, mask) + bce_loss(output, mask) + jacc_loss(output, mask) + focal_loss(output, mask)


        loss_value.backward()
        optimizer.step_and_update_lr()
        # optimizer.step()

        losses += loss_value.item()
        len_of_batch += 1
        
        if args.dev:
            if len_of_batch == 10:
                break
    
    average_loss = losses/len_of_batch
    
    return average_loss

def validater(model, val_loader, device, args, dc_loss, bce_loss, jacc_loss, focal_loss):
    model.eval()
    losses = 0
    len_of_batch = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc = 'Vaidating'):
            img, mask, _ = batch
            img, mask = img.to(device), mask.to(device)

            output = model(img)
            if args.loss == 'dice':
                loss_value = dc_loss(output, mask)
            elif args.loss == 'bce':
                loss_value = bce_loss(output, mask)
            elif args.loss == 'all':
                loss_value = dc_loss(output, mask) + bce_loss(output, mask) + jacc_loss(output, mask) + focal_loss(output, mask)


            losses += loss_value.item()
            len_of_batch += 1
            
            if args.dev:
                if len_of_batch == 10:
                    break
    average_loss = losses/len_of_batch
    return average_loss

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return dice.mean()

def iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.mean()


def visualize(img, mask, output, args, inst):
    # Change to visualizable shape (h w c) not (c h w)
    img = img.squeeze().cpu().numpy().transpose(1, 2, 0) 
    mask = mask.squeeze().cpu().numpy()
    output = output.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[2].imshow(output, cmap='gray')
    axes[2].set_title('Predicted Mask')
    plt.savefig(f'./runs/checkpoint/{args.checkpoint}/visualization_{inst}.png')

def tester(model, test_loader, device, args):
    model.eval()
    len_of_batch = 0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            img, mask, _ = batch
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            
            output = torch.sigmoid(output)  # Assuming sigmoid for binary classification
            output = (output > 0.5).float() # True is > 0.5 else false -> True -> 1.0 False -> 0.0

            dice = dice_coefficient(output, mask)
            iou_score = iou(output, mask)
            dice_scores.append(dice.item())
            iou_scores.append(iou_score.item())
            
            if len_of_batch >20:
                pass
            else:
                visualize(img, mask, output, args, inst = len_of_batch)
            if args.dev:
                if len_of_batch == 10:
                    break

            len_of_batch += 1

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f'Average Dice Coefficient: {avg_dice}')
    print(f'Average IoU: {avg_iou}')
    
