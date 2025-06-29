
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.neighbors import KernelDensity


class MINE(nn.Module):
    """
    MINE network for estimating mutual information between two variables.

    References
    ----------
    Belghazi, Mohamed Ishmael, et al. "Mutual Information Neural Estimation."
    Proceedings of the 35th International Conference on Machine Learning, 2018.
    https://arxiv.org/abs/1801.04062

    Arguments
    ---------
    input_dim_x : int
        Number of features (dimensions) for the first input variable x.

    input_dim_y : int
        Number of features (dimensions) for the second input variable y.

    hidden_size : int, optional
        Number of hidden units in the intermediate layer. Default is 128.

    Example
    -------
    >>> net = MINE(32, 32)
    >>> x = torch.rand((10, 32))
    >>> y = torch.rand((10, 32))
    >>> output = net(x, y)
    """
    def __init__(self, input_dim_x, input_dim_y, hidden_size=128):
        super(MINE, self).__init__()
        self.fc1_x = nn.Linear(input_dim_x, hidden_size)
        self.fc1_y = nn.Linear(input_dim_y, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x, y):
        h = F.relu(self.fc1_x(x) + self.fc1_y(y))
        return torch.tanh(self.fc2(h))

def mutual_information_mine(x, y, mine_net, ma_et=None, ma_rate=0.01):
    """
    Estimate mutual information between two PyTorch tensors using
    the MINE (Mutual Information Neural Estimation) approach.

    Arguments
    ---------
    x : torch.Tensor
        First input tensor of shape [batch_size, dim].

    y : torch.Tensor
        Second input tensor of shape [batch_size, dim].

    mine_net : MINE
        Neural network (critic) used to estimate mutual information.

    ma_et : torch.Tensor or None, optional
        Moving average of the marginal term for numerical stability.
        If None, initialized on first call.

    ma_rate : float, optional
        Update rate for the moving average (default: 0.01).

    Returns
    -------
    mi_est : torch.Tensor
        Scalar tensor representing the estimated mutual information.

    ma_et : torch.Tensor
        Updated moving average term.

    Example
    -------
    >>> mine_net = MINE(32, 32)
    >>> x = torch.rand((10, 32))
    >>> y = torch.rand((10, 32))
    >>> mi_est, ma_et = mutual_information_mine(x, y, mine_net)
    """

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
