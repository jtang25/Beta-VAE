import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, latent_dim: int, beta: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta = beta

    def encode(self, X):
        h = self.encoder(X)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, X):
        mu, logvar = self.encode(X)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss(self, recon, X, mu, logvar):
        recon_loss = F.mse_loss(recon, X, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_div

    def generate(self, num_samples: int):
        z = torch.randn(num_samples, self.latent_dim, device=next(self.parameters()).device)
        return self.decode(z)

