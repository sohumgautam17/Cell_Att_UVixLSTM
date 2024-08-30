import numpy as np
import matplotlib.pyplot as plt


class ScheduledOptim():

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        self.lr_track = []

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.lr_track.append(lr)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
# Stop training if validtion loss has not improved after 5 epochs (Prevent Overfit)
def early_stopping(validation_losses, patience=5, delta=0):
    # Do not stop if not enough val_losses have passes  
    if len(validation_losses) < patience + 1:
        return False
    
    best_loss = min(validation_losses[:-patience])
    current_loss = validation_losses[-1]
    
    if current_loss > best_loss + delta:
        return True
    
    return False


