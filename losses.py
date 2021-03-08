import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6, dims=3):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.dims = dims
    
    def forward(self, pred, target):
        if self.dims == 3:
            dim = (-3, -2, -1)
        elif self.dims == 2:
            dim = (-2, -1)
    
        intersection = torch.sum(target * pred, dim=dim)
        dn = torch.sum(target * target + pred * pred, dim=dim) + self.eps
        return - torch.mean(2 * intersection / dn, dim=[0, 1])
    
class DiceLossLoss(nn.Module):
    def __init__(self, eps=1e-6, dims=3):
        super(DiceLossLoss, self).__init__()
        self.eps = eps
        self.dims = dims
    
    def forward(self, pred, target):
        if self.dims == 3:
            dim = (-3, -2, -1)
        elif self.dims == 2:
            dim = (-2, -1)
    
        intersection = torch.sum(target * pred, dim=dim)
        dn = torch.sum(target + pred, dim=dim) + self.eps
        return - torch.mean(2 * intersection / dn, dim=[0, 1])

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, logvar):
        # c, D, H, W = pred.shape()
        # n = c * D * H * W
        n = 10000
        # return (1 / n) * torch.sum(torch.exp(logvar) + mu * mu - 1. - logvar)
        return (1 / n) * torch.mean(- 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, pred, target):
        return nn.MSELoss()(pred, target)