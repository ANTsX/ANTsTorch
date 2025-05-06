
import torch
import numpy as np

from sklearn.neighbors import KernelDensity

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

    bandwith : scalar
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
