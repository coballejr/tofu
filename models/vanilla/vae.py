import torch
from torch.nn import Linear, Sequential

class Encoder(torch.nn.Module):

    def __init__(self,  inp_dim: int = 784,
                        h_dim: int = 128,
                        out_dim: int = 2):

        super(Encoder, self).__init__()
        self.mlp = Sequential(Linear(inp_dim, h_dim), torch.nn.ReLU(),
                              Linear(h_dim, h_dim), torch.nn.ReLU(),
                              Linear(h_dim, h_dim), torch.nn.ReLU())
        self.mu = Linear(h_dim, out_dim)
        self.logvar = Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        x = x.flatten(start_dim = 1)
        h = self.mlp(x)
        mu, logvar = self.mu(h), self.logvar(h)

        # reparameterization
        sigma = torch.exp(0.5*logvar)
        w = torch.randn_like(mu)
        z = sigma*w + mu
        return z, mu, logvar

class Decoder(torch.nn.Module):

    def __init__(self, inp_dim: int = 2,
                       h_dim: int = 128,
                       out_dim: int = 784):

        super(Decoder, self).__init__()
        self.mlp = Sequential(Linear(inp_dim, h_dim), torch.nn.ReLU(),
                              Linear(h_dim, h_dim), torch.nn.ReLU(),
                              Linear(h_dim, out_dim), torch.nn.Sigmoid())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x

class VAE(torch.nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return z, mu, logvar, x_recon
