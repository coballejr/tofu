import torch
from torch.nn import BCELoss

class VAELoss:

    def __init__(self):
        self.bce = BCELoss(reduction = 'none')
        return

    def recon(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        loss_by_sample = self.bce(x_recon, x)
        loss_by_sample = loss_by_sample.sum(axis = 1)
        batch_loss = loss_by_sample.mean()
        return batch_loss

    def kld(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = torch.exp(logvar)
        loss_by_sample = -0.5*(1 + logvar - mu.pow(2) - var).sum(axis = 1)
        batch_loss = loss_by_sample.mean()
        return batch_loss

    def __call__(self, mu: torch.Tensor,
                       logvar: torch.Tensor,
                       x: torch.Tensor,
                       x_recon: torch.Tensor) -> torch.Tensor:
        r = self.recon(x, x_recon)
        k = self.kld(mu, logvar)
        return r + k
