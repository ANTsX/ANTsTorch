import torch

def absolute_pearson_correlation(x, y):
    """
    Compute absolute Pearson correlation between two vectors (batch-wise).

    Args:
        x: Tensor of shape [batch_size, dim]
        y: Tensor of shape [batch_size, dim]

    Returns:
        Scalar tensor representing average absolute correlation across dimensions.
    """
    x_centered = x - x.mean(0, keepdim=True)
    y_centered = y - y.mean(0, keepdim=True)

    cov = (x_centered * y_centered).mean(0)
    x_std = x_centered.std(0) + torch.finfo(torch.float32).eps
    y_std = y_centered.std(0) + torch.finfo(torch.float32).eps

    corr = cov / (x_std * y_std)

    # Average absolute correlation across dimensions
    return torch.mean(torch.abs(corr))