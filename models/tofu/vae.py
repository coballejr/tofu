import torch
from torch.nn import Linear, Sequential
from .tofu import ToFULayer, CCDiagramLayer

class ToFUEncoder(torch.nn.Module):

    def __init__(self,  inp_dim: int = 784,
                        h_dim: int = 128,
                        out_dim: int = 1,
                        n_dgms: int = 2,
                        n_features: int = 1):



        super(ToFUEncoder, self).__init__()

        self.tofu = Sequential(CCDiagramLayer((28, 28), sub = False),
                               ToFULayer(n_dgms, n_features, birth_lims = [-1,0]),
                               Linear(n_dgms, inp_dim), torch.nn.Sigmoid())
        self.mlp = Sequential(Linear(inp_dim, h_dim), torch.nn.ReLU(),
                              Linear(h_dim, h_dim), torch.nn.ReLU(),
                              Linear(h_dim, h_dim), torch.nn.ReLU())

        self.mu_x = Linear(h_dim, out_dim)
        self.logvar_x = Linear(h_dim, out_dim)

        self.mu_t = Linear(h_dim, out_dim)
        self.logvar_t = Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self.tofu(x)
        h_t = self.mlp(x_t)
        mu_t, logvar_t = self.mu_t(h_t), self.logvar_t(h_t)

        x = x.squeeze()
        x = x.flatten(start_dim = 1)
        h = self.mlp(x)
        mu_x, logvar_x = self.mu_x(h), self.logvar_x(h)

        mu = torch.cat([mu_t, mu_x], dim = -1)
        logvar = torch.cat([logvar_t, logvar_x], dim = -1)

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

class ToFUVAE(torch.nn.Module):


    def __init__(self):
        super(ToFUVAE, self).__init__()
        self.encoder = ToFUEncoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return z, mu, logvar, x_recon
