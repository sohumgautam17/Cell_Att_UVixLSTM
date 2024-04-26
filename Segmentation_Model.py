import torch 
from torch import nn


class CellSegmenter_fn(nn.Module):
  def __init__(self, input_size:(int, int int), ) 
    super().__init__()
    self.unet = nn.Sequential(
            nn.Conv2d(..., ..., kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(..., ..., kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(...),
            nn.ReLU()
    )
  def forward(self, x):
    return self.unet(x)

