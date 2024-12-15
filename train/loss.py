import torch
import torch.nn as nn

class VAELoss(nn.Module):
    def __init__(self, kl_weight=0.001):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.kl_weight = kl_weight
        
    def forward(self, recon, target, mu, log_var):
        """
        Args:
            recon: reconstructed value from decoder
            target: original input value
            mu: mean from encoder
            log_var: log variance from encoder
            
        Returns:
            dict: containing individual losses and total loss
        """
        # Reconstruction loss (MSE)
        recon_loss = self.mse_loss(recon, target)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss
        } 
    

class VQVAELoss(nn.Module):
    def __init__(self, vq_weight=1.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.vq_weight = vq_weight
        
    def forward(self, recon, target, vq_loss):
        """
        Args:
            recon: reconstructed value from decoder
            target: original input value
            vq_loss: loss from vector quantizer
            
        Returns:
            dict: containing individual losses and total loss
        """
        # Reconstruction loss
        recon_loss = self.mse_loss(recon, target)
        
        # Total loss
        total_loss = recon_loss + self.vq_weight * vq_loss
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'vq': vq_loss
        }