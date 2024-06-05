
import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.encoder = self.build_network(input_dim, hidden_dims, latent_dim * 2)
        self.decoder = self.build_network(latent_dim, hidden_dims[::-1], input_dim)
        
    def build_network(self, input_dim, hidden_dims, output_dim):
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        
        # Reparameterization
        z = self.reparameterize(mu, log_var)
        
        # Decoding
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

    def loss_function(self, x, x_reconstructed, mu, log_var):
        recon_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss
