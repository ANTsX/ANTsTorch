
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from sklearn.neighbors import KernelDensity


class MINE(nn.Module):
    """
    MINE network for estimating mutual information between two variables.

    References
    ----------
    Belghazi, Mohamed Ishmael, et al. "Mutual Information Neural Estimation."
    Proceedings of the 35th International Conference on Machine Learning, 2018.
    https://arxiv.org/abs/1801.04062

    Also https://github.com/gtegner/mine-pytorch.

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
        return self.fc2(h)
    
def mutual_information_mine(x, y, mine_net, running_mean=None, alpha=0.01, loss_type='mine'):
    """
    Estimate mutual information using different loss options and EMA stabilization.  Variable
    loss type available:

    | Loss type       | Gradient stability     | Bias risk                 | Practical use                             |
    | --------------- | -----------------------| ------------------------- | ----------------------------------------- |
    | `'mine'`        | Needs EMA tuning       | Unbiased if done properly | Most commonly used if EMA well-tuned      |
    | `'mine_biased'` | More stable early on   | Biased downward           | Good for small batches or simpler code    |
    | `'fdiv'`        | Generally stable       | Lower bound may be loose  | Alternative if other two fail to converge |

    Parameters
    ----------
    x : torch.Tensor
        Input X.

    y : torch.Tensor
        Input Y.

    mine_net : nn.Module
        Critic network.

    running_mean : torch.Tensor or None
        Exponential moving average of the marginal term.

    alpha : float
        Smoothing parameter for EMA.

    loss_type : str
        'mine', 'mine_biased', or 'fdiv'.

    Returns
    -------
    mi_est : torch.Tensor
        MI estimate.

    running_mean : torch.Tensor
        Updated running mean.

    Example
    -------
    >>> mine_net = MINE(32, 32)
    >>> x = torch.rand((100, 32))
    >>> y = torch.rand((100, 32))
    >>> 
    >>> running_mean = None
    >>> for i in range(1000):
    >>>     mi_est, running_mean = mutual_information_mine(
    >>>         x, y, mine_net, running_mean, alpha=0.01, loss_type='mine'
    >>>     )
    >>>     # Backprop using -mi_est (if maximizing MI)
    """

    def ema(value, running_mean, alpha):
        if running_mean is None:
            return value.detach()
        else:
            return alpha * value.detach() + (1. - alpha) * running_mean

    t_joint = mine_net(x, y)
    joint_mean = torch.mean(t_joint)

    # Shuffle y for marginal
    y_shuffle = y[torch.randperm(y.size(0))]
    t_marg = mine_net(x, y_shuffle)

    if loss_type == 'mine':
        # Exponential moving average on exp term
        marginal_exp_mean = torch.exp(t_marg).mean()
        running_mean = ema(marginal_exp_mean, running_mean, alpha)
        mi_est = joint_mean - torch.log(running_mean + torch.finfo(torch.float32).eps)
    elif loss_type == 'mine_biased':
        logsumexp_marg = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0])
        mi_est = joint_mean - logsumexp_marg
    elif loss_type == 'fdiv':
        f_marg = torch.exp(t_marg - 1).mean()
        mi_est = joint_mean - f_marg
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return mi_est, running_mean


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
