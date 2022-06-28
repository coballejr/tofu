import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import Tuple
import gudhi as gd
import numpy as np

class ToFU(torch.nn.Module):

    def __init__(self,n_features: int,
                 birth_lims: Tuple = (0,1),
                 birth_death: bool = True):

        super(ToFU,self).__init__()

        # initialize learnable diagram
        lwr_lim, upper_lim = birth_lims
        rng = upper_lim - lwr_lim
        b = torch.maximum(rng*torch.rand(n_features)+lwr_lim,torch.tensor(lwr_lim))
        d = b+(rng*torch.rand(n_features)+lwr_lim) if birth_death else rng*torch.rand(n_features)+lwr_lim
        dgm = torch.hstack((b.view(n_features,1),d.view(n_features,1)))

        self.dgm = torch.nn.Parameter(dgm)
        self.dgm_np = dgm.detach().cpu().numpy()

    def forward(self,dgm_x: torch.Tensor):

        # compute minimal cost matchings
        dgm_x_np = dgm_x.detach().cpu().numpy()
        cost = cdist(self.dgm_np, dgm_x_np)
        tofu_idx, target_idx = linear_sum_assignment(cost)

        # compute forward pass
        dgm_target = self.dgm.clone().detach()
        dgm_target[tofu_idx,:] = dgm_x[target_idx,:]

        return 0.5*((self.dgm-dgm_target)**2).sum()

class BatchToFU(torch.nn.Module):

    def __init__(self,n_features,birth_lims = [0,1], birth_death = True):
        super(BatchToFU,self).__init__()
        self.tofu = ToFU(n_features,birth_lims,birth_death)

    def forward(self,dgm_batch):
        h_batch = torch.zeros((dgm_batch.shape[0],1))

        for b_idx,dgm_x in enumerate(dgm_batch):
            h = self.tofu(dgm_x)
            h_batch[b_idx,:] = h

        return h_batch

class ToFULayer(torch.nn.Module):

    def __init__(self,n_dgms,n_features,birth_lims = [0,1], birth_death = True):
        super(ToFULayer,self).__init__()
        self.n_dgms = n_dgms
        self.units = torch.nn.ModuleList([BatchToFU(n_features,birth_lims,birth_death) for i in range(n_dgms)]) # (n_dgms,batch_size,1)

    def forward(self,dgm_batch):
        h = torch.zeros((self.n_dgms,dgm_batch.shape[0],1))

        for i,unit in enumerate(self.units):
            h[i,:,:] = unit(dgm_batch)

        h = torch.einsum('cbf -> bcf',h)
        return h.view(dgm_batch.shape[0],self.n_dgms)

class CCDiagramLayer(torch.nn.Module):

    def __init__(self, img_dims: Tuple,
                       sub = True):
        self.sub = sub
        self.img_dims = img_dims
        super(CCDiagramLayer, self).__init__()

    def _pad(self, dgm: np.array, max_feats: int = 10) -> np.array:
        if not dgm.any():
            dgm = np.array([[0,0]])

        # sort by persistence
        p = dgm[:,1] - dgm[:,0]
        sidx = np.argsort(-p) # neg because default is ascending
        dgm = dgm[sidx,:]
        n_feats = dgm.shape[0]

        if n_feats <= max_feats:
            dgm =np.pad(dgm,((0,max_feats-n_feats),(0,0)),constant_values=((4,4),(0,0)))
        else:
            dgm = dgm[0:max_feats,:]
        return dgm

    def forward(self, x: torch.Tensor,
                      hom_dim: int = 1,
                      max_feats: int = 4) -> torch.Tensor:
        x = x.squeeze()
        dgms = torch.zeros((x.shape[0], max_feats, 2))

        x = x.cpu().numpy() if self.sub else -x.cpu().numpy()
        nx, ny = self.img_dims

        for i,img in enumerate(x):
            cc = gd.CubicalComplex(dimensions = [nx, ny],
                                   top_dimensional_cells = img.flatten())
            cc.compute_persistence()
            dgm = cc.persistence_intervals_in_dimension(hom_dim)
            dgm = self._pad(dgm, max_feats)
            dgms[i,:,:] = torch.Tensor(dgm)

        return dgms
