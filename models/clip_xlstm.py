import torch
import torch.nn as nn
from einops import rearrange
from monai.networks.blocks import PatchEmbeddingBlock
import einops
from enum import Enum
import math
import os
import torch.nn.functional as F
from vLSTM import *
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel, CLIPTokenizer
from PIL import Image
import requests

def clip_load(checkpoint_path, device):
    print(f'Loading weights from finetuned clip model from {checkpoint_path}')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    clip_model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model Successfully loaded')
    return clip_model

def freeze_clip_weights(model):
    for params in model.parameters():
        params.requires_grad = False

def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
    
    # Print high-level architecture
    print("Model Architecture:")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")

def encoder(checkpoint_path, device):
    clip_model = clip_load(checkpoint_path, device)
    freeze_clip_weights(clip_model)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image_path = '../Data/Data/images/image_0.png'
    text_path = '../Data/Data/texts/text_0.txt'

    signal_image = Image.open(image_path).convert("RGB")

    with open(text_path, 'r') as f:
        text = f.read().strip()

    inputs_text = processor(text=text, return_tensors="pt", padding='max_length', 
                          truncation=True, max_length=77)
    inputs_image = processor(images=signal_image, return_tensors="pt")

    inputs = {
        'input_ids': inputs_text.input_ids.to(device),
        'attention_mask': inputs_text.attention_mask.to(device),
        'pixel_values': inputs_image.pixel_values.to(device)
    }

    print_model_summary(clip_model)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        print("\nTest forward pass successful")
        print("Output keys:", outputs.keys())
        print(outputs['image_embeds'].shape)
        
    return clip_model, outputs

device = torch.device('cuda:2')    
model, inputs = encoder('../runs/checkpoint/best_clip_pretrain_checkpoint.pt/best_checkpoint.chkpt', device)


class xlstm_bottle(nn.Module):
    def __init__(image_embebds, dim=512, drop_path_rate=0.0,
    alternation='bidirectional', drop_path_decay=False, legacy_norm=False):
        super().__init__()

        self.alternation = alternation
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        directions = []
        if alternation == "bidirectional":
            for i in range(depth):
                if i % 2 == 0:
                    directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
                else:
                    directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        else:
            raise NotImplementedError(f"invalid alternation '{alternation}'")


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
        for block in self.blocks:
            x = block(x)
        return x


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
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
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


    def forward(self, x, x1, x2, x3):

        x = self.decoder1(x, x3)
        # print(f'shape after decoder 1: {x.shape}')

        x = self.decoder2(x, x2)
        # print(f'shape after decoder 2: {x.shape}')

        x = self.decoder3(x, x1)
        # print(f'shape after decoder 3: {x.shape}')

        x = self.decoder4(x)
        # print(f'shape after decoder 4: {x.shape}')

        x = self.conv1(x)
        # print(f'shape after decoder 5: {x.shape}')


        return x

class clip_xlstm(nn.Module):
