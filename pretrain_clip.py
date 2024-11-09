# Load model directly
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, CLIPVisionModelWithProjection, CLIPVisionModel, CLIPModel
from huggingface_hub import login
from PIL import Image
import requests
import torch
import json
import glob
from data_loader import ECGCLIPPretrain
from utils import early_stopping, plot_train_val_loss
from tqdm import tqdm

def normalize_all(signal, V, percentiles):
    normalized = (signal - (percentiles['percentile_1'] - 0.5)) / ((percentiles['percentile_99']+0.5) - (percentiles['percentile_1']-0.5) + 1e-6) 
    clipped_normalized = np.clip(normalized, 0, 1)
    scaled_signal = clipped_normalized * V
    int_signal = np.round(scaled_signal).astype(np.uint8)
    return int_signal

def main():
    device = torch.device('cuda:2')    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './.huggingface').to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir = './.huggingface')
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
    
    all_signals = glob.glob('./new_data/ecg/train/*.npy')
    all_text = glob.glob('./new_data/text/train/*.json')
    print(all_signals[0])
    print(len(all_signals))
    print(all_text[0])
    print(len(all_text))
    
    dataset = ECGCLIPPretrain(all_signals, all_text, processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    epochs = 150
    
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
        if average_loss <= min(losses):
            torch.save(checkpoint, f'./runs/0/clip/best_clip_pretrain.pt')
            print('fBest model saved at epoch {epoch}')
            
    plot_train_val_loss(losses, dir_path = './runs/0/clip')
            

if __name__ == '__main__':
    main()
