import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class ToFU(torch.nn.Module):
    
    def __init__(self,n_features,birth_lims = [0,1],birth_death = True):
        super(ToFU,self).__init__()
        
        # initialize learnable diagram
        lwr_lim,upper_lim = birth_lims
        rng = upper_lim - lwr_lim
        b = torch.maximum(rng*torch.rand(n_features)+lwr_lim,torch.tensor(lwr_lim))
        d = b+(rng*torch.rand(n_features)+lwr_lim) if birth_death else rng*torch.rand(n_features)+lwr_lim
        dgm = torch.hstack((b.view(n_features,1),d.view(n_features,1)))
        
        self.dgm = torch.nn.Parameter(dgm)
        self.dgm_np = dgm.detach().cpu().numpy()
        
    def forward(self,dgm_x):
             
        # compute minimal cost matchings
        dgm_x_np = dgm_x.detach().cpu().numpy()
        cost = cdist(self.dgm_np, dgm_x_np)
        tofu_idx, target_idx = linear_sum_assignment(cost)
        
        # compute forward pass
        dgm_target = self.dgm.clone().detach()
        dgm_target[tofu_idx,:] = dgm_x[target_idx,:]

        return 0.5*((self.dgm-dgm_target)**2).sum()

class BatchToFU(torch.nn.Module):
    
    def __init__(self,n_features,device,birth_lims = [0,1], birth_death = True):
        super(BatchToFU,self).__init__()
        self.tofu = ToFU(n_features,birth_lims,birth_death)
        self.device = device
        
    def forward(self,dgm_batch):
        h_batch = torch.zeros((dgm_batch.shape[0],1),requires_grad = True).to(self.device)
        
        for b_idx,dgm_x in enumerate(dgm_batch):
            h = self.tofu(dgm_x)
            h_batch[b_idx,:] = h
            
        return h_batch
    
class ToFULayer(torch.nn.Module):
    
    def __init__(self,n_dgms,n_features,device,birth_lims = [0,1], birth_death = True):
        super(ToFULayer,self).__init__()
        self.n_dgms = n_dgms
        self.device = device
        self.units = torch.nn.ModuleList([BatchToFU(n_features,device,birth_lims,birth_death) for i in range(n_dgms)]) # (n_dgms,batch_size,1)
        
    def forward(self,dgm_batch):
        h = torch.zeros((self.n_dgms,dgm_batch.shape[0],1),requires_grad = True).to(self.device)
        
        for i,unit in enumerate(self.units):
            h[i,:,:] = unit(dgm_batch)
        
        h = torch.einsum('cbf -> bcf',h)
        return h.view(dgm_batch.shape[0],self.n_dgms)
    