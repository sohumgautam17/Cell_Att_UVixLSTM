import torch
import matplotlib as plt

def visualize_segmentation(images, masks, preds, num_images=3):
    
    images = images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NHWC for visualization
    print(masks.shape)
    masks = masks.cpu().numpy()
    print(preds.shape)
    preds = torch.argmax(preds, dim=1).cpu().numpy()  # Convert predictions to class indices
    
    fig, axs = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    
    for i in range(num_images):
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1] for displaying

        ax = axs[i, 0]
        ax.imshow(img)
        ax.set_title("Original Image")
        ax.axis('off')

        ax = axs[i, 1]
        ax.imshow(img)
        ax.imshow(masks[i], alpha=0.3, cmap='jet')  # Overlay mask
        ax.set_title("Ground Truth Mask")
        ax.axis('off')

        ax = axs[i, 2]
        ax.imshow(img)
        ax.imshow(preds[i], alpha=0.3, cmap='jet')  # Overlay prediction
        ax.set_title("Predicted Mask")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('./test.png')
    plt.close()

# batch_size = 7
# images = torch.randn(batch_size, 3, 512, 512).to(device)  # Random images
# targets = torch.randint(0, 2, (batch_size, 512, 512)).to(device)  # Random ground truth masks
# outputs = model(images)  # Model predictions

# # Visualize the first 3 images, masks, and predictions
# visualize_segmentation(images.detach().cpu(), targets.detach().cpu(), outputs.detach().cpu(), num_images=3)
