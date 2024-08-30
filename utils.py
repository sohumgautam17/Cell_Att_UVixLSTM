import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Initialize Hyperparameters')
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--patience', type = int, default = 5, help = 'Please choose the patience of the early stopper')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Please choose the type of device' )
    parser.add_argument('--dataset', type=str, default='./Data/pannuke_fold1+2.npy', help='choose a dataset to train with')
    parser.add_argument('--warmup', type = int, default = 2000, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--epochs', type = int, default = 25, help = 'Please choose the number of epochs' )
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