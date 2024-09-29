import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape

####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1, padding=1)  # Changed padding to 1
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 3, stride=2, padding=1)  # Changed padding to 1
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 3, stride=2, padding=1)  # Changed padding to 1
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2, padding=1)  # Changed padding to 1

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            # Change out_ch to 1 for single-channel output
            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder


        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=1)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=1)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):
        imgs = imgs / 255.0  # to 0-1 range to match XY
        print(f'input image shape {imgs.shape}')
    
        if self.training:
            d0 = self.conv0(imgs)
            print(f'after conv0 {d0.shape}')
            d0 = self.d0(d0, self.freeze)
            print(f'after downsample 0 {d0.shape}')
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        print(f'd0 shape: {d0.shape}')
        print(f'd1 shape: {d1.shape}')
        print(f'd2 shape: {d2.shape}')
        print(f'd3 shape: {d3.shape}')

        # Crop operations based on mode
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1], target_shape=d[-2].shape) + d[-2]
            u3 = branch_desc[0](u3)
            
            u2 = self.upsample2x(u3, target_shape=d[-3].shape) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2, target_shape=d[-4].shape) + d[-4]  # Ensured shape consistency
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict['np']



        # if self.mode == 'original':
        #     d[0] = crop_op(d[0], [184, 184])
        #     d[1] = crop_op(d[1], [72, 72])
        # else:
        #     d[0] = crop_op(d[0], [92, 92])
        #     d[1] = crop_op(d[1], [36, 36])

        # out_dict = OrderedDict()
        # print("Starting decoder processing...")

        # for branch_name, branch_desc in self.decoder.items():
        #     u3 = self.upsample2x(d[-1])
        #     print(f'upsample u3 shape {u3.shape}')
            
        #     # Ensure the shapes match
        #     if u3.size(2) != d[-2].size(2) or u3.size(3) != d[-2].size(3):
        #         pad_height = d[-2].size(2) - u3.size(2)
        #         pad_width = d[-2].size(3) - u3.size(3)
        #         u3 = F.pad(u3, (0, pad_width, 0, pad_height))

        #     u3 = u3 + d[-2]
        #     u3 = branch_desc[0](u3)

        #     u2 = self.upsample2x(u3)
        #     if u2.size(2) != d[-3].size(2) or u2.size(3) != d[-3].size(3):
        #         pad_height = d[-3].size(2) - u2.size(2)
        #         pad_width = d[-3].size(3) - u2.size(3)
        #         u2 = F.pad(u2, (0, pad_width, 0, pad_height))

        #     u2 = u2 + d[-3]
        #     u2 = branch_desc[1](u2)

        #     u1 = self.upsample2x(u2)
        #     if u1.size(2) != d[-4].size(2) or u1.size(3) != d[-4].size(3):
        #         pad_height = d[-4].size(2) - u1.size(2)
        #         pad_width = d[-4].size(3) - u1.size(3)
        #         u1 = F.pad(u1, (0, pad_width, 0, pad_height))

        #     u1 = u1 + d[-4]
        #     u1 = branch_desc[2](u1)

        #     u0 = branch_desc[3](u1)
        #     out_dict[branch_name] = u0

        # return out_dict['np']
