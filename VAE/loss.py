import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    
    def __init__(self, kld_weight: float=1., recon_weight: float=1.):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight
        self.recon_weight = recon_weight
        
    def forward(self, x, recon_x, mu, log_var):
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return self.kld_weight * KLD + self.recon_weight * BCE