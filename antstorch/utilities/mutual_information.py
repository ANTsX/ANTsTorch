
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.neighbors import KernelDensity


class MINE(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, hidden_size=128):
        super(MINE, self).__init__()
        self.fc1_x = nn.Linear(input_dim_x, hidden_size)
        self.fc1_y = nn.Linear(input_dim_y, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x, y):
        h = F.relu(self.fc1_x(x) + self.fc1_y(y))
        return self.fc2(h)

def mutual_information_mine(x, y, mine_net, ma_et=None, ma_rate=0.01):
    # x, y shape: [batch_size, dim]
    joint = mine_net(x, y).mean()

    # Shuffle y to create independent pairs
    y_shuffle = y[torch.randperm(y.size(0))]
    marginal = torch.exp(mine_net(x, y_shuffle)).mean()

    # Moving average for stability
    if ma_et is None:
        ma_et = marginal.detach()
    else:
        ma_et = (1 - ma_rate) * ma_et + ma_rate * marginal.detach()

    mi_est = joint - torch.log(ma_et + 1e-8)
    return mi_est, ma_et


def mutual_information_kde(x, y, bandwidth=0.1):
    """
    Calculate mutual information between two tensors using kernel 
    density estimation.  

    Arguments
    ---------
    x : PyTorch tensor
        PyTorch tensor.

    y : PyTorch tensor
        PyTorch tensor.

    bandwidth : scalar
        Parameter for kernel density estimation.
        
    Returns
    -------

    Singleton PyTorch tensor

    Data frame with probability values for each disease category.

    Example
    -------
    >>> x = torch.rand((5, 2))
    >>> y = torch.rand((5, 2))
    >>> mi = antstorch.mutual_information_kde(x, y)
    """    

    x_np = x.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    xy_np = np.hstack([x_np, y_np])
    
    kde_xy = KernelDensity(bandwidth=bandwidth).fit(xy_np)
    kde_x = KernelDensity(bandwidth=bandwidth).fit(x_np)
    kde_y = KernelDensity(bandwidth=bandwidth).fit(y_np)
    
    joint_log_density = kde_xy.score_samples(xy_np)
    x_log_density = kde_x.score_samples(x_np)
    y_log_density = kde_y.score_samples(y_np)
    
    mi_np = np.mean(joint_log_density - x_log_density - y_log_density)

    return torch.tensor(mi_np, requires_grad=True, device=x.device)
