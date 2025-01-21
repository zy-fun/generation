import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predictions, targets, mask):
        squared_error = (predictions - targets) ** 2
        masked_squared_error = squared_error * mask
        num_valid = mask.sum()
        loss = masked_squared_error.sum() / num_valid
        return loss