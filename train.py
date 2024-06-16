import torch
import wandb
from tqdm.auto import tqdm


def train(model, X_train, EPOCHS, learning_rate, num_params, BATCH_SIZE, train_loader, DICE_loss, BCE_loss, optimizer):
    model.train()
    total_train_loss = 0
    train_losses = []
    #eval_losses = []
    print(f"Number of Images Trained on: {len(X_train)} | # Epochs: {EPOCHS} | LR: {learning_rate} | Number of Params: {num_params}")

    wandb.init(
        project='Cellseg_research',
        config={
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': learning_rate,
                        
        }
    )
    
    for epoch in tqdm(range(EPOCHS)):
        epoch_train_loss = 0
        for batch_idx, (image, annote) in enumerate(train_loader):
        
            image = image.permute(0, 3, 1, 2) # This is because the image was originally in size (32, 250, 250, 3), Torch doesnt like this
            
            logits = model(image)
            annote = annote.unsqueeze(1)

            dice_loss = DICE_loss(logits, annote)
            bce_loss = BCE_loss(logits, annote)
            loss = dice_loss + bce_loss

            if batch_idx % 30 == 0:
                print(f"Epoch #{epoch}_______Batch #{batch_idx}...")
                print(f"Dice Loss: {dice_loss.item()}  BCE Loss: {bce_loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        wandb.log({
                "batch_dice_loss": dice_loss,
                "batch_bce_loss": bce_loss,
                "total_loss":loss,
            })

        epoch_train_loss /= len(train_loader)
        total_train_loss += epoch_train_loss
        print(f"Epoch {epoch} Train Loss: {epoch_train_loss}")

    
    total_train_loss /= EPOCHS
    train_losses.append(total_train_loss)
    print(f"Total Training Loss: {total_train_loss}")
    return total_train_loss




def test(model, test_loader, DICE_loss, BCE_loss):
    model.eval()
    total_dice_loss = 0
    total_bce_loss = 0

    with torch.no_grad():
        for image, annote in test_loader:
            test_logits = model(image)
            annote = annote.unsqueeze(1)  # Ensure the annotation shape matches the logits shape if needed

            dice_loss = DICE_loss(test_logits, annote)
            bce_loss = BCE_loss(test_logits, annote)


            total_dice_loss += dice_loss.item()
            total_bce_loss += bce_loss.item()

    avg_dice_loss = total_dice_loss / len(test_loader)
    avg_bce_loss = total_bce_loss / len(test_loader)

    return print(f'DICEL: {avg_dice_loss}, BCEL: {avg_bce_loss}')