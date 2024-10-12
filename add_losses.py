
import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Compute Jaccard loss for each class
        intersection = torch.sum(predictions * targets, dim=(0, 2, 3))
        union = torch.sum(predictions, dim=(0, 2, 3)) + torch.sum(targets, dim=(0, 2, 3)) - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - jaccard
        
        # Average loss over all classes
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        
        # predictions and targets should be of shape [B, C, H, W]
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Average loss over all dimensions except the class dimension
        return focal_loss.mean(dim=(0, 2, 3)).sum()
