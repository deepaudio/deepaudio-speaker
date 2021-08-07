import torch
from torch import nn

EPSILON = 1e-6


class CMVN(nn.Module):
    def __init__(self, var_norm=False):
        super(CMVN, self).__init__()
        self.var_norm = var_norm

    def forward(self, x):
        mean = x.mean(dim=1, keepdims=True)
        if self.var_norm:
            std = torch.sqrt(x.var(dim=1, keepdims=True) + EPSILON)
        x = x - mean
        if self.var_norm:
            x /= std
        return x
