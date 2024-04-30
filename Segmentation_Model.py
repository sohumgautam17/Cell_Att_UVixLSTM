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



***

  (architecture): Unet(
    (encoder): EfficientNetEncoder(
      (conv_stem): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNormAct2d(
        64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        (drop): Identity()
        (act): Swish()
      )
      (blocks): Sequential(
        (0): Sequential(
          (0): DepthwiseSeparableConv(
            (conv_dw): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn1): BatchNormAct2d(
              64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pw): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn2): BatchNormAct2d(
              32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (1): DepthwiseSeparableConv(
            (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn1): BatchNormAct2d(
              32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn2): BatchNormAct2d(
              32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.004)
          )
          (2): DepthwiseSeparableConv(
            (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn1): BatchNormAct2d(
              32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn2): BatchNormAct2d(
              32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.007)
          )
          (3): DepthwiseSeparableConv(
            (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn1): BatchNormAct2d(
              32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn2): BatchNormAct2d(
              32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.011)
          )
        )
        (1): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
            (bn2): BatchNormAct2d(
              192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.015)
          )
          (1): InvertedResidual(
            (conv_pw): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
            (bn2): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.018)
          )
          (2): InvertedResidual(
            (conv_pw): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
            (bn2): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.022)
          )
          (3): InvertedResidual(
            (conv_pw): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
            (bn2): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.025)
          )
          (4): InvertedResidual(
            (conv_pw): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
            (bn2): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.029)
          )
          (5): InvertedResidual(
            (conv_pw): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
            (bn2): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.033)
          )
          (6): InvertedResidual(
            (conv_pw): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
            (bn2): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.036)
          )
        )
        (2): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
            (bn2): BatchNormAct2d(
              288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(288, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.040)
          )
          (1): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
            (bn2): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.044)
          )
          (2): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
            (bn2): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.047)
          )
          (3): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
            (bn2): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.051)
          )
          (4): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
            (bn2): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.055)
          )
          (5): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
            (bn2): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.058)
          )
          (6): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
            (bn2): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.062)
          )
        )
        (3): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=480, bias=False)
            (bn2): BatchNormAct2d(
              480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(480, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.065)
          )
          (1): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.069)
          )
          (2): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.073)
          )
          (3): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.076)
          )
          (4): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.080)
          )
          (5): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.084)
          )
          (6): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.087)
          )
          (7): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.091)
          )
          (8): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.095)
          )
          (9): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.098)
          )
        )
        (4): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
            (bn2): BatchNormAct2d(
              960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(960, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.102)
          )
          (1): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.105)
          )
          (2): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.109)
          )
          (3): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.113)
          )
          (4): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.116)
          )
          (5): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.120)
          )
          (6): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.124)
          )
          (7): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.127)
          )
          (8): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.131)
          )
          (9): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.135)
          )
        )
        (5): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(1344, 1344, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=1344, bias=False)
            (bn2): BatchNormAct2d(
              1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.138)
          )
          (1): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.142)
          )
          (2): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.145)
          )
          (3): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.149)
          )
          (4): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.153)
          )
          (5): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.156)
          )
          (6): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.160)
          )
          (7): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.164)
          )
          (8): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.167)
          )
          (9): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.171)
          )
          (10): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.175)
          )
          (11): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.178)
          )
          (12): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.182)
          )
        )
        (6): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(2304, 2304, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2304, bias=False)
            (bn2): BatchNormAct2d(
              2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.185)
          )
          (1): InvertedResidual(
            (conv_pw): Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              3840, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(3840, 3840, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3840, bias=False)
            (bn2): BatchNormAct2d(
              3840, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.189)
          )
          (2): InvertedResidual(
            (conv_pw): Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              3840, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(3840, 3840, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3840, bias=False)
            (bn2): BatchNormAct2d(
              3840, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.193)
          )
          (3): InvertedResidual(
            (conv_pw): Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              3840, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (conv_dw): Conv2d(3840, 3840, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3840, bias=False)
            (bn2): BatchNormAct2d(
              3840, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Swish()
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))
              (act1): Swish()
              (conv_expand): Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): DropPath(drop_prob=0.196)
          )
        )
      )
      (conv_head): Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNormAct2d(
        2560, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        (drop): Identity()
        (act): Swish()
      )
      (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
    )
    (decoder): UnetDecoder(
      (center): Identity()
      (blocks): ModuleList(
        (0): DecoderBlock(
          (conv1): Conv2dReLU(
            (0): Conv2d(864, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention1): Attention(
            (attention): Identity()
          )
          (conv2): Conv2dReLU(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention2): Attention(
            (attention): Identity()
          )
        )
        (1): DecoderBlock(
          (conv1): Conv2dReLU(
            (0): Conv2d(336, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention1): Attention(
            (attention): Identity()
          )
          (conv2): Conv2dReLU(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention2): Attention(
            (attention): Identity()
          )
        )
        (2): DecoderBlock(
          (conv1): Conv2dReLU(
            (0): Conv2d(176, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention1): Attention(
            (attention): Identity()
          )
          (conv2): Conv2dReLU(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention2): Attention(
            (attention): Identity()
          )
        )
        (3): DecoderBlock(
          (conv1): Conv2dReLU(
            (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention1): Attention(
            (attention): Identity()
          )
          (conv2): Conv2dReLU(
            (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention2): Attention(
            (attention): Identity()
          )
        )
        (4): DecoderBlock(
          (conv1): Conv2dReLU(
            (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention1): Attention(
            (attention): Identity()
          )
          (conv2): Conv2dReLU(
            (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (attention2): Attention(
            (attention): Identity()
          )
        )
      )
    )
    (segmentation_head): SegmentationHead(
      (0): Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): Identity()
      (2): Activation(
        (activation): Identity()
      )
    )
  )
)
***
