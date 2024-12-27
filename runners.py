from tqdm import tqdm
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

## Functionized Training, Val, and Test Loops

def trainer(model, train_loader, optimizer, device, args, dc_loss, bce_loss, jacc_loss, focal_loss):
    model.train()
    losses = 0
    len_of_batch = 0
    
    for batch in tqdm(train_loader, desc = 'Training'):
        optimizer.zero_grad()
        img, mask, _ = batch
        img, mask = img.to(device), mask.to(device)
        print(f'Input image shape: {img.shape} input mask shape: {mask.shape} \n')
        output = model(img)
        print(f'Model output shape: {output.shape}')

        if args.loss == 'dice':
            loss_value = dc_loss(output, mask.squeeze(1))
        elif args.loss == 'bce':
            loss_value = bce_loss(output, mask.squeeze(1))
        elif args.loss == 'all':
            targets = mask.squeeze(1).long()

            dice_loss = dc_loss(output, targets)
            ce_loss = bce_loss(output, targets)
            jac_loss = jacc_loss(output, targets)
            foc_loss = focal_loss(output, targets)

            loss_value = dice_loss + ce_loss + jac_loss + foc_loss

        loss_value.backward()
        optimizer.step_and_update_lr()
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
                loss_value = dc_loss(output, mask.squeeze(1))
            elif args.loss == 'bce':
                loss_value = bce_loss(output, mask.squeeze(1))
            elif args.loss == 'all':
                loss_value = dc_loss(output, mask.squeeze(1)) + bce_loss(output, mask.squeeze(1)) + jacc_loss(output, mask.squeeze(1)) + focal_loss(output, mask.squeeze(1))
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

    intersection = (pred * target).sum(dim=2).sum(dim=2)  # (B, C)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)  # (B, C)
    
    return dice.mean(dim=1)


def iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=2).sum(dim=2)  # (B, C)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection  # (B, C)
    iou = (intersection + smooth) / (union + smooth)  # (B, C)
    
    return iou.mean(dim=1)



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

def tester(model, test_loader, device, args, num_classes=6):
    model.eval()
    len_of_batch = 0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            img, mask, _ = batch
            img, mask = img.to(device), mask.to(device)

            output = model(img)
            output = torch.softmax(output, dim=1)
            pred_classes = torch.argmax(output, dim=1)
            mask_one_hot = F.one_hot(mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
            pred_one_hot = F.one_hot(pred_classes, num_classes=num_classes).permute(0, 3, 1, 2).float()
            dice = dice_coefficient(pred_one_hot, mask_one_hot)
            iou_score = iou(pred_one_hot, mask_one_hot)
            
            dice_scores.append(dice.mean().item())
            iou_scores.append(iou_score.mean().item())

            if args.dev and len_of_batch == 10:
                break

            len_of_batch += 1

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f'Average Dice Coefficient: {avg_dice}')
    print(f'Average IoU: {avg_iou}')