import torch
import numpy as np

class SmartSave:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            pass
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def get_timeF(x):
    timeF = torch.zeros(x.shape[0], 4)
    timeF[:, 0] = x
    x = (x + 0.5) * 86399
    timeF[:, 1] = (x // 3600) / 23 - 0.5
    timeF[:, 2] = (x % 3600) // 60 / 59 - 0.5
    timeF[:, 3] = (x % 60) / 59 - 0.5

    return timeF