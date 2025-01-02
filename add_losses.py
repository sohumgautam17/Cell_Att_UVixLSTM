
import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-5, num_classes=6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=1)
        
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = torch.sum(predictions * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(predictions, dim=(0, 2, 3)) + torch.sum(targets_one_hot, dim=(0, 2, 3)) - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - jaccard
        
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=1)
        
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        bce_loss = F.binary_cross_entropy(predictions, targets_one_hot, reduction='none')
        
        pt = torch.exp(-bce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


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
