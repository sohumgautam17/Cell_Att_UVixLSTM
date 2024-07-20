import torch
import torch.nn as nn
from torchsummary import summary
from einops import rearrange
from monai.networks.blocks import PatchEmbeddingBlock
import einops
from enum import Enum
import math
import torch.nn.functional as F
from models.vLSTM import *
from models.attGateUtils import AttentionBlock

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        # interconnected residual connections
        x = x + x_down
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels,
                 depth=24,
                 dim=1024,
                 drop_path_rate=0.0,
                 stride=None,
                 alternation="bidirectional",
                 drop_path_decay=False,
                 legacy_norm=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)
        self.patch_embed = PatchEmbeddingBlock(in_channels=out_channels * 8,
                                               img_size=img_dim // 16,
                                               patch_size=2,
                                               hidden_size=256,
                                               num_heads=1,
                                               proj_type='perceptron',
                                               spatial_dims=2)

        self.conv2 = nn.Conv2d(out_channels * 8, 512,
                               kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.alternation = alternation
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # directions of img traversals
        directions = []
        if alternation == "bidirectional":
            for i in range(depth):
                if i % 2 == 0:
                    directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
                else:
                    directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        else:
            raise NotImplementedError(f"invalid alternation '{alternation}'")

        # blocks
        self.blocks = nn.ModuleList(
            [
                ViLBlock(
                    dim=dim,
                    drop_path=dpr[i],
                    direction=directions[i],
                )
                for i in range(depth)
            ]
        )
        # LEGACY: only norm after pooling is needed, norm after blocks is not needed but was used for training
        if legacy_norm:
            self.legacy_norm = LayerNorm(dim, bias=False)
        else:
            self.legacy_norm = nn.Identity()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # head

        # no head -> use as feature extractor
        self.output_shape = ((img_dim // 16) // 2, dim)

    def load_state_dict(self, state_dict, strict=True):
        # interpolate pos_embed for different resolution (e.g. for fine-tuning on higher-resolution)
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed.embed"}

    def forward(self, x):
        print(f'encoder x og size {x.shape}')
        x = self.conv1(x) # after conv1 torch.Size([1, 64, 48, 48])
        
        x = self.norm1(x)
        x1 = self.relu(x)
        print(f'encoder x after conv1 {x1.shape}')

        x2 = self.encoder1(x1) # after encoder1 torch.Size([1, 128, 24, 24])
        # input()
        print(f'encoder x after encoder1 {x2.shape}')

        x3 = self.encoder2(x2)
        print(f'encoder x after encoder2 {x3.shape}')


        x4 = self.encoder3(x3)
        print(f'encoder x after encoder3 {x4.shape}')
       

        x4 = self.patch_embed(x4)
        print(f'size after embeddings')


        x4= einops.rearrange(x4, "b ... d -> b (...) d")
        # print(x.size())

        for block in self.blocks:
            x4 = block(x4)
        x4 = self.legacy_norm(x4)
        x4 = self.norm(x4) # torch.Size([1, 9, 256])
        print(f'size after norm {x4.shape}')
        x4 = rearrange(x4, "b (x y) c -> b c x y", x=self.output_shape[0], y=self.output_shape[0])
        print(f'after rearrange {x4.shape}')
        # input()
        return x1, x2, x3, x4


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor*2, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # def forward(self, x, x_concat=None):
    #     print(x.shape, x_concat.shape)
    #     print('----')
    #     if x.shape[2] == 3:
    #         x = self.upsample1(x)
    #     else:
    #         x = self.upsample(x)
    #     print(x.size(), x_concat.size())
    #     if x_concat is not None:
    #         x = torch.cat([x_concat, x], dim=1)

    #     x = self.layer(x)
    #     return x
    def forward(self, x, x_concat=None):
        # print(x.size(), x_concat.size() if x_concat is not None else None)
        
        if x_concat is not None:
            target_size = x_concat.shape[2:]
            x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=True)
        else:
            x = self.upsample(x)
        
        # print(x.size(), x_concat.size() if x_concat is not None else None)
        
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
        
        x = self.layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()
        
        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1, stride = 2)

        self.attention_block1 = AttentionBlock(out_channels * 8, out_channels * 8, out_channels * 4)
        self.attention_block2 = AttentionBlock(out_channels * 4, out_channels * 4, out_channels * 2)
        self.attention_block3 = AttentionBlock(out_channels * 2, out_channels * 2, out_channels)
        self.attention_block4 = AttentionBlock(out_channels, out_channels, out_channels // 2)


    def forward(self, x1, x2, x3, x4):
        print(f'decoder  x1: {x1.shape} | x2: {x2.shape} | x3: {x3.shape} | x: {x4.shape},')

        x4 = self.attention_block1(gate=x4, skip_connection=x4)
        print(f'After attention_block1, x: {x.shape} | x4: {x4.shape}')
        x = self.decoder1(x4, x4)
        print(f'first decoder: {x.shape}')

        x3 = self.attention_block2(gate=x, skip_connection=x3)
        print(f'After attention_block2, x: {x.shape} | x3: {x3.shape}')
        x = self.decoder2(x, x3)
        print(f'second decoder: {x.shape}')

        x2 = self.attention_block3(gate=x, skip_connection=x2)
        print(f'After attention_block3, x: {x.shape} | x2: {x1.shape}')
        x = self.decoder3(x, x2)
        print(f'third decoder: {x.shape}')

        x1 = self.attention_block4(gate=x, skip_connection=x1)
        print(f'After attention_block4, x: {x.shape} | x1: {x1.shape}')
        x = self.decoder4(x, x1)
        print(f'fourth decoder: {x.shape}')

        x = self.conv1(x)
        print(f'last {x.shape}')

        return x

class UVixLSTM(nn.Module):
    def __init__(self, class_num, img_dim=96,
                     in_channels=1,
                     out_channels=64,
                     depth=12,
                     dim=256):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                                   depth, dim)

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        print(x1.size(), x2.size(), x3.size(), x4.size())
        x = self.decoder(x1, x2, x3, x4)
        
        return x
