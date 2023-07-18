import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, in_dim, hidden_dim1, hidden_dim2, z_dim):
        super(Encoder, self).__init__()

        self.e1 = nn.Linear(in_dim, hidden_dim1)
        self.e2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.mu = nn.Linear(hidden_dim2, z_dim)
        self.std = nn.Linear(hidden_dim2, z_dim)

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(x)
        mu, std = self.mu(x), self.std(x)
        return x, mu, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim1, hidden_dim2, z_dim, out_dim):
        super(Decoder, self).__init__()

        self.d1 = nn.Linear(z_dim, hidden_dim2)
        self.d2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.out = nn.Linear(hidden_dim1, out_dim)

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return F.sigmoid(x)


class VAE(nn.Module):

    def __init__(self, in_dim, hidden_dim1, hidden_dim2, z_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_dim, hidden_dim1, hidden_dim2, z_dim)
        self.decoder = Decoder(hidden_dim1, hidden_dim2, z_dim, in_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # reparameterization trick

    def forward(self, src):
        x, mu, std = self.encoder(src)
        z = self.sampling(mu, std)
        out = self.decoder(z)
        return out, mu, std