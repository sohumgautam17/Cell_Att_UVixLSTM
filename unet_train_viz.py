import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + (in_channels // 2), out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + (in_channels // 2), out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


device = torch.device("cuda")
    
    
num_classes = 2  
model = UNet(n_channels=3, n_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy inputs and labels for demonstration
# Input should be of the shape [batch_size, channels, H, W]
inputs = torch.randn(7, 3, 512, 512).to(device)  # 7 images in the batch

# Targets should be of the shape [batch_size, H, W] and contain class indices
# Assuming the number of classes is 10, class indices range from 0 to 9
targets = torch.randint(0, 2, (7, 512, 512)).to(device)  # Randomly generated indices for each image in the batch

# Forward pass
outputs = model(inputs)

loss = criterion(outputs, targets) 

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")




def visualize_segmentation(images, masks, preds, num_images=3):
    
    images = images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NHWC for visualization
    masks = masks.cpu().numpy()
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

batch_size = 7
images = torch.randn(batch_size, 3, 512, 512).to(device)  # Random images
targets = torch.randint(0, 2, (batch_size, 512, 512)).to(device)  # Random ground truth masks
outputs = model(images)  # Model predictions

# Visualize the first 3 images, masks, and predictions
visualize_segmentation(images.detach().cpu(), targets.detach().cpu(), outputs.detach().cpu(), num_images=3)