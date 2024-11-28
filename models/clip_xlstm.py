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
from torchvision import transforms

def clip_load(checkpoint_path, device):
        print(f'Loading weights from finetuned clip model from {checkpoint_path}')
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        clip_model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model Successfully loaded')
        return clip_model

def freeze_clip_weights(model):
        for params in model.parameters():
            params.requires_grad = False # gradients dont need to be computed

def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters()) # total elements
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # trainable elements
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
    
    # Print high-level architecture
    print("Model Architecture:")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")

class clip_encoder(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.clip_model = clip_load(checkpoint_path, device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   

    def forward(self, x):
        model = self.clip_model
        freeze_clip_weights(model)

        signal_image = Image.open(x['image']).convert("RGB")

        with open(x['text'], 'r') as f:
            text = f.read().strip()

        processor = self.processor
        inputs_text = processor(text=text, return_tensors="pt", padding='max_length', 
                            truncation=True, max_length=77)
        inputs_image = processor(images=signal_image, return_tensors="pt")

        inputs = {
            'input_ids': inputs_text.input_ids.to(device),
            'pixel_values': inputs_image.pixel_values.to(device)
        }

        # print_model_summary(model)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # print("Output keys:", outputs.keys())
            # print(outputs['image_embeds'].shape)
            return outputs['image_embeds'] # (1, 512)

class xlstm_bottle(nn.Module):
    def __init__(self, image=None, img_dim=256, dim=512, depth=24, drop_path_rate=0.0, 
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
                    direction=directions[i],)
                for i in range(depth)
            ]
        )

        if legacy_norm:
            self.legacy_norm = LayerNorm(dim, bias=False)
        else:
            self.legacy_norm = nn.Identity()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # no head -> use as feature extractor
        self.output_shape = ((img_dim // 16) // 2, dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.legacy_norm(x)
        x = self.norm(x)

        print(f'xlstm pre rearrange output shape: {x.shape}')
        b, c, _ = x.shape
        x = rearrange(x, "b c (x y) -> b c x y", x=32, y=16)  # Ensure correct dimensions for x and y

        print(f'Shape of x is {x.shape}')
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


    def forward(self, batch):

        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x

class clip_xlstm(nn.Module):
    def __init__(self, checkpoint_path, device, shape2=512, shape3=256):
        super().__init__()

        self.clip = clip_encoder(
            checkpoint_path = checkpoint_path, 
            device=device
        )

        self.xlstm = xlstm_bottle()

        self.clip_projection = nn.Linear(512, shape2)
        self.xlstm_projection = nn.Linear(512, shape3)

        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * shape2, shape2),
            nn.ReLU(),
            nn.Linear(shape2, shape3)
        ) # shape 3 same size as first layer in the upsampling

        self.decoder = Decoder(out_channels=64, class_num=1) 
 
    def fuse_clip_xlstm(self, batch):

        with torch.no_grad():
            clip_output = self.clip(batch)
            clip_output = clip_output.unsqueeze(1).to(device)

            proj_clip_out = self.clip_projection(clip_output)

            xlstm_output = self.xlstm(clip_output)
            proj_xlstm_out = self.xlstm_projection(xlstm_output)

            combined_out = torch.cat(proj_clip_out, proj_xlstm_out, dim = 2)

            fused_out = self.fusion_layer(combined_out)

        return fused_out

    def forward(self, x):
        fused_out = self.fuse_clip_xlstm(batch)

image_path = './Data/Data/masks/mask_0.png' 
text_path = './Data/Data/texts/text_0.txt' 
checkpoint_path = './runs/checkpoint/best_clip_pretrain_checkpoint.pt/best_checkpoint.chkpt'  
device = torch.device('cuda:2')

batch = {'image': image_path, 'text': text_path}

model = clip_xlstm(checkpoint_path, device).to(device)

model.eval()
with torch.no_grad():
    fused_output = model.fuse_clip_xlstm(batch)

print(fused_output.shape)
print(fused_output)
