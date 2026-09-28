"""Targeted compatibility for PyTorch accelerator operations."""

import warnings
from torch.nn import functional as F


def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    """Use CPU interpolation when the MPS 3-D forward kernel is unavailable."""
    try:
        return F.grid_sample(
            input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )
    except NotImplementedError as error:
        if input.device.type != "mps" or input.ndim != 5 or "grid_sampler_3d" not in str(error):
            raise
        warnings.warn(
            "MPS grid_sampler_3d is unavailable; evaluating 3-D interpolation "
            "on CPU and returning the result to MPS. Timings include CPU transfers.",
            RuntimeWarning,
        )
        return F.grid_sample(
            input.cpu(), grid.cpu(), mode=mode, padding_mode=padding_mode,
            align_corners=align_corners,
        ).to(input.device)
