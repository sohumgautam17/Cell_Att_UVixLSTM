# Load model directly
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, CLIPVisionModelWithProjection, CLIPVisionModel, CLIPModel, CLIPTokenizer
from huggingface_hub import login
from PIL import Image
import requests
import torch
import json
import glob
import os
from dataloader import ECGCLIPPretrain
from optim import early_stopping
from tqdm import tqdm
from main import ensure_directory_exists


def main():
    device = torch.device('cuda:2')    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './.huggingface').to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir = './.huggingface')
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
    
    all_signals_path = './Data/Data/images/*'
    all_texts_path = './Data/Data/texts/*'

    dataset = ECGCLIPPretrain(all_signals_path, all_texts_path, tokenizer, processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 150

    print(f'Training for {150} epochs')
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
            optimizer.zero_grad()
            inputs = batch
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        
        average_loss = sum(epoch_loss) / len(epoch_loss)
        losses.append(average_loss)

        
        print(f'Epoch {epoch} Loss: {average_loss}')
        
        early_stop = early_stopping(losses, patience=10, delta=0.01)
        if early_stop:
            print('Loss has stopped decreasing. Early stopping...')
            break
        
        model_state_dict = model.state_dict()
        checkpoint = {
            'model_state_dict': model_state_dict,
            'epoch': epoch,
        }

        directory_path = f'./runs/checkpoint/best_clip_pretrain_checkpoint.pt'
        ensure_directory_exists(directory_path)

        if average_loss <= min(losses):
            torch.save(checkpoint, f'./{directory_path}/best_checkpoint.chkpt')
            print('fBest model saved at epoch {epoch}')

if __name__ == '__main__':
    main()
