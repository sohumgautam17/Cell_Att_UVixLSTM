import torch
torch.set_num_threads(2)
import argparse
from torch.utils.data import DataLoader
import gc
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from segmentation_models_pytorch.losses import DiceLoss
import wandb

from models.models import UNet
from optim import ScheduledOptim, early_stopping
from runners import trainer, validater, tester
from dataloader import CellDataset


def get_args():
    parser = argparse.ArgumentParser(description='Initialize Hyperparameters')
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--patience', type = int, default = 5, help = 'Please choose the patience of the early stopper')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Please choose the type of device' )
    parser.add_argument('--warmup', type = int, default = 2000, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--epochs', type = int, default = 100, help = 'Please choose the number of epochs' )
    parser.add_argument('--batch', type = int, default = 8, help = 'Please choose the batch size')
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--model', type = str, default = 'unet', help = 'Please choose which model to use')

    parser.add_argument('--loss', type = str, default = 'dice', help = 'Please choose which loss to use')
    # Can change this to "parser.add_argument('--loss', type = str, choices = ['dice', 'bce'], help = 'Please choose which loss to use')"

    parser.add_argument('--checkpoint', type = str, help = 'Please choose the checkpoint to use')
    parser.add_argument('--inference', action='store_true', help = 'Please choose whether it is inference or not')
    parser.add_argument('--log', action='store_true', help = 'Please choose whether to log or not')
    parser.add_argument('--dev', action='store_true', help = 'Please choose whether to be in dev mode or not')
    return parser.parse_args()

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")



def main(args):
    # Development mode for rapid testing 
    if args.dev:
        args.epochs = 1
        args.inference = False
        args.log = False

    directory_path = f'./runs/checkpoint/saved_best_{args.lr}_{args.batch}_{args.patience}_{args.weight_decay}_{args.model}'
    ensure_directory_exists(directory_path)

    # Free memory and empty cache, and set device to use GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(2) # Change to 42 
    device = torch.device(args.device)
    print(device)
    
    # Load data compiled in preprocess.py and extract train, val
    print('Loading Data...')
    all_data = np.load('./Data/all_data.npy', allow_pickle=True).item()
    train_data_imgs = all_data['train_patched_images']
    train_data_masks = all_data['train_patched_masks']
    val_data_imgs = all_data['val_patched_images']
    val_data_masks = all_data['val_patched_masks']

    # Instantiate custom PyTorch dataset and create DataLoaders
    train_dataset = CellDataset(train_data_imgs, train_data_masks, args)
    val_dataset = CellDataset(val_data_imgs, val_data_masks, args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle = True)   
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle = True)

    # Instantiate model unet
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        model_hidden_size = 1024 
        
    model = model.to(device)
    
    # If in Dev Mode Inference Mode OFF, else inference on test dataset
    if args.inference:
        test_data_imgs = all_data['test_patched_images']
        test_data_masks = all_data['test_patched_masks']
        test_dataset = CellDataset(test_data_imgs, test_data_masks, args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle = False)

        # Load saved weights from trained model and inference
        checkpoint = torch.load(f'./runs/checkpoint/{args.checkpoint}/best_checkpoint.chkpt', map_location = args.device)
        model.load_state_dict(checkpoint['model'])
        tester(model, test_loader, device, args)
    else:
        # Continue in Dev Mode
        optimizer = ScheduledOptim(
        Adam(filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-4, lr = args.lr, weight_decay=args.weight_decay), model_hidden_size, args.warmup)
        
        if args.loss == 'dice':
            loss = DiceLoss(mode='binary')
        elif args.loss == 'bce':
            loss = torch.nn.BCEWithLogitsLoss()

        # Train, Val loss tracking for visualization
        train_losses = []
        val_losses = []
        all_epochs = []

        # Training loop 
        for epoch in range(args.epochs):
            
            all_epochs.append(epoch)
            train_loss = trainer(model, train_loader, optimizer, device, args, loss)
            print(f"Training - Epoch: {epoch+1},Train Loss: {train_loss}")
            train_losses.append(train_loss)
            
            val_loss = validater(model, val_loader, device, args, loss)
            print(f"Evaluation - Epoch: {epoch+1}, Val Loss: {val_loss}")
            val_losses.append(val_loss)
            
            model_state_dict = model.state_dict()
                
            checkpoint = {
                'model' : model_state_dict,
                'config_file' : 'config',
                'epoch' : epoch
            }
            
            # Save models best performance
            if val_loss <= min(val_losses):
                torch.save(checkpoint, f'./{directory_path}/best_checkpoint.chkpt')
                print('    - [Info] The checkpoint file has been updated.')
            
            early_stop = early_stopping(val_losses, patience = args.patience, delta = 0.01)
        
            if early_stop:
                print('Validation loss has stopped decreasing. Early stopping...')
                break   
        
        fig1 = plt.figure('Figure 1')
        plt.plot(train_losses, label = 'train')
        plt.plot(val_losses, label= 'valid')
        plt.xlabel('epoch')
        plt.ylim([0.0, max(train_losses)])
        plt.ylabel('loss')
        plt.legend(loc ="upper right")
        plt.title('loss change curve')
        plt.savefig(f'./{directory_path}/loss_plot.png')
        plt.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
