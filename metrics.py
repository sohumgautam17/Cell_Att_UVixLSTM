import torch
from torch import nn
import torch.nn.Functional as F
from sklearn.metrics import recall_score, average_precision_score


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice



def accuracy_metric(pred, gt):
  with torch.no_grad():
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    correct = torch.eq(pred, gt).int() #True is 1 False is 0
    accuracy = (float(correct.sum()))/float(correct.numel())
  return accuracy



def recall(y_true, y_pred):
  recall_metric = recall_score(y_true, y_pred, average='binary')




def MAP(y_true, y_score):
  average_precision = average_precision_score(y_true, y_score)




def iou_implementation(outputs, truth):
  smoothing = 1e-6 # prevent division by 0

  print(outputs.shape), print(truth.shape)
  outputs = outputs.squeeze() # depends on shape of the tensors AND all tensors have to be torch tensor

  intersection = (outputs & truth).float.sum(1, 2)
  union = (output | truth).float.sum(1, 2)

  iou = (intersection+smoothing)/(union+smoothing)
  print(iou)

  #    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

  return iou #Return IOU or threshold?
