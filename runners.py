from tqdm import tqdm
import torch


def trainer(model, train_loader, optimizer, device, args, loss):
    model.train()
    losses = 0
    len_of_batch = 0
    
    for batch in tqdm(train_loader, desc = 'Training'):
        optimizer.zero_grad()
        img, mask = batch
        img, mask = img.to(device), mask.to(device)

        output = model(img)
        loss_value = loss(output, mask)
        loss_value.backward()
        optimizer.step_and_update_lr()
        losses += loss_value.item()
        len_of_batch += 1
        
        if args.dev:
            if len_of_batch == 10:
                break
    
    average_loss = losses/len_of_batch
    
    return average_loss

def validater(model, val_loader, device, args, loss):
    model.eval()
    losses = 0
    len_of_batch = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc = 'Vaidating'):
            img, mask = batch
            img, mask = img.to(device), mask.to(device)

            output = model(img)
            loss_value = loss(output, mask)
            losses += loss_value.item()
            len_of_batch += 1
            
            if args.dev:
                if len_of_batch == 10:
                    break
    average_loss = losses/len_of_batch
    return average_loss

def tester(model, test_loader, device, args):
    model.eval()
    losses = 0
    len_of_batch = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc = 'Vaidating'):
            img, mask = batch
            img, mask = img.to(device), mask.to(device)

            output = model(img)
            
            if args.dev:
                if len_of_batch == 10:
                    break
    
