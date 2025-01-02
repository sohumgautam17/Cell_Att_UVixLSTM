from tqdm import tqdm
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from add_losses import dice_coefficient, iou
import os
import matplotlib.patches as mpatches


## Functionized Training, Val, and Test Loops
def trainer(model, train_loader, optimizer, device, args, dc_loss, bce_loss, jacc_loss, focal_loss):
    model.train()
    losses = 0
    len_of_batch = 0
    
    for batch in tqdm(train_loader, desc = 'Training'):
        optimizer.zero_grad()
        xlstm_img, mask, clip_input = batch
        xlstm_img, mask = xlstm_img.to(device), mask.to(device)
        
        mask = mask.permute(0, 3, 1, 2)
        mask = mask.squeeze(1).long()

        clip_input = {
            'input_ids': clip_input['input_ids'].to(device),
            'pixel_values': clip_input['pixel_values'].to(device)
        }

        model_inputs = {
            'clip_data': clip_input,
            'xlstm_image': xlstm_img
        }

        output = model(model_inputs)
        

        if args.loss == 'dice':
            loss_value = dc_loss(output, mask)  # No need for squeeze now
        elif args.loss == 'bce':
            loss_value = bce_loss(output, mask)  # No need for squeeze now
        elif args.loss == 'all':
            targets = mask.long()  # No need for squeeze now

            dice_loss = dc_loss(output, targets)
            ce_loss = bce_loss(output, targets)
            jac_loss = jacc_loss(output, targets)
            foc_loss = focal_loss(output, targets)

            loss_value = dice_loss + ce_loss + jac_loss + foc_loss
            
            # Debug print
            # print("\nLoss values:")
            # print(f"dice: {dice_loss.item():.4f}")
            # print(f"ce: {ce_loss.item():.4f}")
            # print(f"jac: {jac_loss.item():.4f}")
            # print(f"focal: {foc_loss.item():.4f}")
            # print(f"total: {loss_value.item():.4f}")
            # print(f"requires grad: {loss_value.requires_grad}")

        # Make sure inputs are on the right device
        if not output.is_cuda:
            print("Warning: output not on CUDA")
        if not mask.is_cuda:
            print("Warning: mask not on CUDA")

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
            xlstm_img, mask, clip_input = batch
            xlstm_img, mask = xlstm_img.to(device), mask.to(device)
            
            mask = mask.permute(0, 3, 1, 2)
            mask = mask.squeeze(1).long()

            clip_input = {
                'input_ids': clip_input['input_ids'].to(device),
                'pixel_values': clip_input['pixel_values'].to(device)
            }

            model_inputs = {
                'clip_data': clip_input,
                'xlstm_image': xlstm_img
            }

            output = model(model_inputs)

            if args.loss == 'dice':
                loss_value = dc_loss(output, mask)  # No need for squeeze now
            elif args.loss == 'bce':
                loss_value = bce_loss(output, mask)  # No need for squeeze now
            elif args.loss == 'all':
                targets = mask.long()  # No need for squeeze now
                dice_loss = dc_loss(output, targets)
                ce_loss = bce_loss(output, targets)
                jac_loss = jacc_loss(output, targets)
                foc_loss = focal_loss(output, targets)
                
                loss_value = dice_loss + ce_loss + jac_loss + foc_loss

            losses += loss_value.item()
            len_of_batch += 1
            
            if args.dev:
                if len_of_batch == 10:
                    break

    average_loss = losses/len_of_batch
    return average_loss


def tester(model, test_loader, device, args, num_classes=6):
    model.eval()
    len_of_batch = 0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            xlstm_img, mask, clip_input = batch
            xlstm_img, mask = xlstm_img.to(device), mask.to(device)
            
            mask = mask.permute(0, 3, 1, 2)
            mask = mask.squeeze(1).long()

            clip_input = {
                'input_ids': clip_input['input_ids'].to(device),
                'pixel_values': clip_input['pixel_values'].to(device)
            }

            model_inputs = {
                'clip_data': clip_input,
                'xlstm_image': xlstm_img
            }

            output = model(model_inputs)

            output = torch.softmax(output, dim=1)
            pred_classes = torch.argmax(output, dim=1)
            mask_one_hot = F.one_hot(mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
            pred_one_hot = F.one_hot(pred_classes, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            dice = dice_coefficient(pred_one_hot, mask_one_hot)
            iou_score = iou(pred_one_hot, mask_one_hot)
            
            dice_scores.append(dice.mean().item())
            iou_scores.append(iou_score.mean().item())

            if batch_idx % 10 == 0:
                visualize(xlstm_img[0], mask[0], pred_classes[0], args, batch_idx)
            
            if args.dev and batch_idx == 10:
                break

            len_of_batch += 1

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f'Average Dice Coefficient: {avg_dice}')
    print(f'Average IoU: {avg_iou}')

    metrics = {
        'dice_scores': dice_scores,
        'iou_scores': iou_scores,
        'avg_dice': avg_dice,
        'avg_iou': avg_iou
    }
    
    np.save(f'./runs/checkpoint/{args.model_checkpoint}/test_metrics.npy', metrics)

def visualize(img, mask, output, args, inst):
    img = img.cpu().numpy().transpose(1, 2, 0)
    mask = mask.cpu().numpy()
    output = output.cpu().numpy()
    
    colors = [
        (0, 255, 255),     # Background - Cyan
        (0, 0, 0),         # Neoplastic - Black  
        (255, 0, 0),       # Inflammatory - Red
        (0, 255, 0),       # Connective - Green
        (0, 0, 255),       # Dead Cells - Blue
        (255, 255, 0),     # Epithelial - Yellow
    ]
    colors = [(r/255, g/255, b/255) for r, g, b in colors]
    custom_cmap = plt.cm.colors.ListedColormap(colors)
    labels = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead Cells', 'Epithelial']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[1].imshow(mask, cmap=custom_cmap)
    axes[2].imshow(output, cmap=custom_cmap)

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    fig.legend(handles=patches, bbox_to_anchor=(1.15, 0.5))
    
    [ax.set_title(t) for ax, t in zip(axes, ['Input Image', 'Ground Truth', 'Prediction'])]
    plt.savefig(f'./runs/checkpoint/{args.model_checkpoint}/visualizations/vis_{inst}.png', bbox_inches='tight')
    plt.close()

# def visualize(img, mask, output, args, inst):
#     # Change to visualizable shape (h w c) not (c h w)
#     img = img.cpu().numpy().transpose(1, 2, 0)
#     mask = mask.cpu().numpy()
#     output = output.cpu().numpy()
    
#     # Create color maps for multi-class visualization
#     cmap = plt.cm.get_cmap('tab10')  # Use tab10 for up to 10 classes
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     axes[0].imshow(img)
#     axes[0].set_title('Input Image')
#     axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=5)
#     axes[1].set_title('Ground Truth Mask')
#     axes[2].imshow(output, cmap=cmap, vmin=0, vmax=5)
#     axes[2].set_title('Predicted Mask')
    
#     plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes[1])
#     plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes[2])
    
#     save_dir = f'./runs/checkpoint/{args.model_checkpoint}/visualizations'
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(f'{save_dir}/vis_{inst}.png')
#     plt.close()