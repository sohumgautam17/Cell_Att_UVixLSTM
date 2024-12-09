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
import numpy as np

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
        self.device = device

    def forward(self, x):
        model = self.clip_model
        freeze_clip_weights(model)

        # Handle image input
        if isinstance(x['image'], str):  # If it's a file path
            signal_image = Image.open(x['image']).convert("RGB")
        else:  # If it's a tensor
            # Convert tensor to PIL Image
            # Assuming input tensor is in shape [B, C, H, W] and normalized
            image_tensor = x['image'].squeeze(0)  # Remove batch dimension
            # Denormalize if your tensor is normalized
            image_tensor = (image_tensor * 255).clamp(0, 255).byte()
            image_tensor = image_tensor.permute(1, 2, 0).cpu().numpy()  # Change to [H, W, C]
            signal_image = Image.fromarray(image_tensor)

        # Handle text input
        if isinstance(x['text'], str):
            if os.path.isfile(x['text']):  # If it's a file path
                with open(x['text'], 'r') as f:
                    text = f.read().strip()
            else:  # If it's a direct text string
                text = x['text']
        else:
            raise ValueError("Text input must be a string or file path")

        # Process inputs
        processor = self.processor
        inputs_text = processor(text=text, return_tensors="pt", padding='max_length', 
                            truncation=True, max_length=77)
        inputs_image = processor(images=signal_image, return_tensors="pt")

        inputs = {
            'input_ids': inputs_text.input_ids.to(self.device),
            'pixel_values': inputs_image.pixel_values.to(self.device)
        }

        with torch.no_grad():
            outputs = model(**inputs)
            return outputs['image_embeds']  # (1, 512)
# class clip_encoder(nn.Module):
#     def __init__(self, checkpoint_path, device):
#         super().__init__()
#         self.clip_model = clip_load(checkpoint_path, device)
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   

#     def forward(self, x):
#         model = self.clip_model
#         freeze_clip_weights(model)

#         signal_image = Image.open(x['image']).convert("RGB")

#         with open(x['text'], 'r') as f:
#             text = f.read().strip()

#         processor = self.processor
#         inputs_text = processor(text=text, return_tensors="pt", padding='max_length', 
#                             truncation=True, max_length=77)
#         inputs_image = processor(images=signal_image, return_tensors="pt")

#         inputs = {
#             'input_ids': inputs_text.input_ids.to(device),
#             'pixel_values': inputs_image.pixel_values.to(device)
#         }

#         # print_model_summary(model)
        
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # print("Output keys:", outputs.keys())
#             print(outputs['image_embeds'].shape)
#             return outputs['image_embeds'] # (1, 512)

class encoder_block(nn.Module):
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

        self.encoder1 = encoder_block(out_channels, out_channels * 2, stride=2)
        self.encoder2 = encoder_block(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = encoder_block(out_channels * 4, out_channels * 8, stride=2)
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

        # directions
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
        x = self.conv1(x) # after conv1 torch.Size([1, 64, 48, 48, 48])
        # print(x.shape)
        x = self.norm1(x)
        x1 = self.relu(x)
        x2 = self.encoder1(x1) # after encoder1 torch.Size([1, 128, 24, 24, 24])
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = self.patch_embed(x)
        x = einops.rearrange(x, "b ... d -> b (...) d")

        print(f'Shape before xLSTM: {x.shape}')
        for block in self.blocks:
            x = block(x)
        x = self.legacy_norm(x)
        x = self.norm(x) # torch.Size([1, 9, 256])
        x = rearrange(x, "b (x y) c -> b c x y", x=self.output_shape[0], y=self.output_shape[0])
        print(f'Output of the xLSTM (x) shape is {x.shape}')
        return x, x1, x2, x3

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


    def forward(self, x, x1, x2, x3):

        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x

class clip_xlstm(nn.Module):
    def __init__(self, checkpoint_path, device, shape2=512, shape3=256, 
                class_num=1, img_dim=256, in_channels=3,
                out_channels=64, depth=12, dim=256):
        super().__init__()
        self.device = device
        
        self.clip = clip_encoder(
            checkpoint_path=checkpoint_path, 
            device=device
        ).to(device)

        self.encoder = Encoder(img_dim, in_channels, out_channels, depth, dim).to(device)
        self.decoder = Decoder(out_channels, class_num).to(device)
        
        self.clip_projection = nn.Linear(512, shape2).to(device)
        self.xlstm_projection = nn.Linear(256, shape3).to(device)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(shape2 + shape3, shape2),
            nn.ReLU(),
            nn.Linear(shape2, shape3)
        ).to(device)
        
        self.decoder = Decoder(out_channels=64, class_num=1).to(device)

    def fuse_clip_xlstm(self, batch):
        with torch.no_grad():
            if torch.is_tensor(batch['image']):
                batch['image'] = batch['image'].to(self.device)
            
            clip_output = self.clip(batch)
            proj_clip_out = self.clip_projection(clip_output)

            xlstm_output, x1, x2, x3 = self.encoder(batch['image'])
            
            B, C, H, W = xlstm_output.shape
            xlstm_flat = xlstm_output.view(B, C, -1).mean(-1)
            proj_xlstm_out = self.xlstm_projection(xlstm_flat)

            combined_out = torch.cat([proj_clip_out, proj_xlstm_out], dim=1)
            fused_out = self.fusion_layer(combined_out)

            x1, x2, x3 = x1.to(self.device), x2.to(self.device), x3.to(self.device)
            x = self.decoder(fused_out.unsqueeze(-1).unsqueeze(-1), x1, x2, x3)

        return x

    def forward(self, batch):
        fused_out = self.fuse_clip_xlstm(batch)
        return fused_out


batch = {
    'image': torch.randn(1, 3, 256, 256),  # or path to image
    'text': "sample text" # or path to text file
}

model = clip_xlstm(
    checkpoint_path='../runs/checkpoint/best_clip_pretrain_checkpoint.pt/best_checkpoint.chkpt',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    shape2=512,
    shape3=256,
    class_num=1,
    img_dim=256,
    in_channels=3,
    out_channels=64,
    depth=12,
    dim=256
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = model.to(device)

output = model(batch)
print(output.shape)

output_image = Image.fromarray((output.squeeze().cpu().numpy() * 255).astype(np.uint8))
output_image.save("test_output.png")
