import torch
torch.set_num_threads(2)
import argparse

from torch.utils.data import DataLoader
import gc
from utils import get_args
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from segmentation_models_pytorch.losses import DiceLoss
import wandb

from models.models import UNet
from models.UVixLSTM_GateAtt import UVixLSTM_Att
from models.UVixLSTM_SelfAtt import UVixLSTM_noAtt
from models.HoVerNet.HoVerNet import HoVerNet
from add_losses import FocalLoss, JaccardLoss
# from models.AttTwoDUVixLSTM import AttUVixLSTM
# from models.AttTwoDUVixLSTM2 import AttUVixLSTM2

# from postprocess.watershed import inference_watershed

from optim import ScheduledOptim, early_stopping
from runners import trainer, validater, tester
from dataloader import CellDataset


def get_args():
    parser = argparse.ArgumentParser(description='Initialize Hyperparameters')
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--patience', type = int, default = 5, help = 'Please choose the patience of the early stopper')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Please choose the type of device' )
    parser.add_argument('--dataset', type = str, default = './Data/pannuke_6c.npy', help = 'Please choose a dataset' )
    parser.add_argument('--warmup', type = int, default = 2000, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--epochs', type = int, default = 100, help = 'Please choose the number of epochs' )
    parser.add_argument('--batch', type = int, default = 8, help = 'Please choose the batch size')
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--model', type = str, default = 'unet', help = 'Please choose which model to use')
    parser.add_argument('--patch_size', type=int, default=256, help='please enter patch size')
    parser.add_argument('--loss', type = str, default = 'dice', help = 'Please choose which loss to use')
    parser.add_argument('--checkpoint', type = str, help = 'Please choose the checkpoint to use')
    parser.add_argument('--inference', action='store_true', help = 'Please choose whether it is inference or not')
    parser.add_argument('--log', action='store_true', help = 'Please choose whether to log or not')
    parser.add_argument('--dev', action='store_true', help = 'Please choose whether to be in dev mode or not')
    parser.add_argument('--augfly', action='store_true', help = 'Please choose whether to do augmentations of the fly, or at the start in preprocess.py')

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
        args.log = False


    # Wandb

    if args.log:
        wandb.init(
            project = 'cell',
            name = f'{args.lr}_{args.batch}_{args.patience}_{args.weight_decay}_{args.model}',
            config = {
                'lr': args.lr,
                'batch': args.batch,
                'patience': args.patience,
                'weight_decay': args.weight_decay,
                'model': args.model,
                'patch_size': args.patch_size,
                'epochs': args.epochs,
                ## anything else to config
            }
        )
    


    # Free memory and empty cache, and set device to use GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(2) # Change to 42 
    device = torch.device(args.device)
    print(device)
    
    # Load data compiled in preprocess.py and extract train, val
    print('Loading Data...')
    print(args.dataset)
    all_data = np.load(args.dataset, allow_pickle=True).item()
    print(all_data.keys())
    # input()
    train_data_imgs = all_data['train_patched_images']
    print(len(train_data_imgs))
    train_data_masks = all_data['train_patched_masks']
    print(len(train_data_masks))

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
    elif args.model == 'attxlstm':
        model = UVixLSTM_Att(class_num = 6, img_dim = 256, in_channels=3)
        model_hidden_size = 256
    elif args.model == 'xlstm':
        model = UVixLSTM_noAtt(class_num = 6, img_dim = 256, in_channels=3)
        model_hidden_size = 256
    elif args.model == "hovernet":
        model = HoVerNet()
        model_hidden_size = 256

    ### ABOVE CHANGE CLASS_NUM TO 6 for 6 classes
  

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
        # inference_watershed(model, test_loader, device, args)


        # This is where Watershed is run

    else:
        
        directory_path = f'./runs/checkpoint/saved_best_{args.lr}_{args.batch}_{args.patience}_{args.weight_decay}_{args.model}_{args.augfly}_{args.loss}'
        ensure_directory_exists(directory_path)
        
        # Continue in Dev Mode
        optimizer = ScheduledOptim(
        Adam(filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-4, lr = args.lr, weight_decay=args.weight_decay), model_hidden_size, args.warmup)
        '''
        optimizer = Adam(
                        filter(lambda x: x.requires_grad, model.parameters()), 
                        lr=args.lr, 
                        betas=(0.9, 0.98), 
                        eps=1e-4, 
                        weight_decay=args.weight_decay
                    )
        '''
        
        '''ScheduledOptim(
        Adam(filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-4, lr = args.lr, weight_decay=args.weight_decay), model_hidden_size, args.warmup)'''
        
        if args.loss == 'dice':
            dc_loss = DiceLoss(mode='mutliclass')
            bce_loss = None
            jaccard_loss = None
            focal_loss = None
        elif args.loss == 'bce':
            bce_loss = torch.nn.BCEWithLogitsLoss()
            dc_loss = None
            jaccard_loss = None
            focal_loss = None
        elif args.loss == 'all':
            # bce_loss = torch.nn.BCEWithLogitsLoss()
            bce_loss = torch.nn.CrossEntropyLoss()
            dc_loss = DiceLoss(mode='multiclass')
            jaccard_loss = JaccardLoss()
            focal_loss = FocalLoss()

        # Train, Val loss tracking for visualization
        train_losses = []
        val_losses = []
        all_epochs = []

        # Training loop 
        for epoch in range(args.epochs):
            
            all_epochs.append(epoch)
            train_loss = trainer(model, train_loader, optimizer, device, args, dc_loss, bce_loss, jaccard_loss, focal_loss)
            print(f"Training - Epoch: {epoch+1},Train Loss: {train_loss}")
            train_losses.append(train_loss)
            
            val_loss = validater(model, val_loader, device, args, dc_loss, bce_loss, jaccard_loss, focal_loss)
            print(f"Evaluation - Epoch: {epoch+1}, Val Loss: {val_loss}")
            val_losses.append(val_loss)
            
            model_state_dict = model.state_dict()
                
            checkpoint = {
                'model' : model_state_dict,
                'config_file' : 'config',
                'epoch' : epoch
            }
            
            if args.log:
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

            # Save models best performance
            if val_loss <= min(val_losses):
                torch.save(checkpoint, f'./{directory_path}/best_checkpoint.chkpt')
                print('    - [Info] The checkpoint file has been updated.')
            
            early_stop = early_stopping(val_losses, patience = args.patience, delta = 0.01)
        
            if early_stop:
                print('Validation loss has stopped decreasing. Early stopping...')
                break   
        
        if args.log:
            wandb.finish()

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
    print("Preprocessing data")
    print('#'*50)
    main(args)