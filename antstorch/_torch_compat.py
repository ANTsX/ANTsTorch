"""Targeted compatibility for PyTorch accelerator operations."""

import warnings
import os
from functools import lru_cache
import torch
from ._grid_sample_3d import grid_sample_3d
from torch.nn import functional as F

# PyTorch's silent CPU fallback (PYTORCH_ENABLE_MPS_FALLBACK=1) turns an
# unimplemented-on-MPS op into a UserWarning ("... is not currently
# supported on the MPS backend and will fall back to run on the CPU")
# instead of a raised exception. A capability probe that only catches
# exceptions would then misreport the op as natively available. _available()
# promotes any UserWarning raised during its narrow probe body to an error
# so a silent CPU fallback is never mistaken for native MPS support.


@lru_cache(maxsize=128)
def _available(backend, dtype, mode, padding_mode, align_corners, backward):
    """Exercise both coordinate and image derivatives when required.

    Forces PYTORCH_ENABLE_MPS_FALLBACK off and promotes PyTorch's MPS
    CPU-fallback warning to an error for the duration of the probe, so a
    user who has that variable set globally still gets an honest answer:
    a silent CPU fallback must never be reported as native support.
    """
    if backend == "metal" and (dtype != torch.float32 or not hasattr(torch.mps, "compile_shader")):
        return False
    previous_fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        sampler = F.grid_sample
        if backend == "metal":
            from ._grid_sample_3d_metal import grid_sample_3d_metal
            sampler = grid_sample_3d_metal
        with torch.enable_grad(), warnings.catch_warnings():
            # warnings.filterwarnings' `message` filter only matches a regex
            # anchored at the *start* of the text, so it cannot key off a
            # substring like "not currently supported"; promote every
            # UserWarning instead -- the probe body is two ops, so any
            # UserWarning raised here is the MPS CPU-fallback warning (or
            # something equally worth treating as "not really native").
            warnings.simplefilter("error", UserWarning)
            image = torch.ones(1, 1, 3, 3, 3, device="mps", dtype=dtype, requires_grad=backward)
            grid = torch.full((1, 1, 1, 1, 3), 0.17, device="mps", dtype=dtype, requires_grad=backward)
            output = sampler(image, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
            if backward:
                torch.autograd.grad(output.sum(), (image, grid))
            torch.mps.synchronize()
        return True
    except (RuntimeError, NotImplementedError, SyntaxError, UserWarning):
        return False
    finally:
        if previous_fallback_env is None:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = previous_fallback_env


def grid_sample_for_probe(input, grid, **kwargs):
    """Select native, then Metal; capability probes never fall back to CPU.

    ANTSTORCH_MPS_GRID_SAMPLE defaults to auto; native/metal/torch force a backend.
    Set the override before starting Python (registration probes are cached).
    """
    if input.device.type != "mps" or input.ndim != 5:
        return F.grid_sample(input, grid, **kwargs)
    backend = os.environ.get("ANTSTORCH_MPS_GRID_SAMPLE", "auto")
    if backend not in ("auto", "native", "metal", "torch"):
        raise ValueError("ANTSTORCH_MPS_GRID_SAMPLE must be auto, native, metal, or torch")
    if backend == "auto":
        backward = torch.is_grad_enabled() and (input.requires_grad or grid.requires_grad)
        options = (input.dtype, kwargs.get("mode", "bilinear"),
                   kwargs.get("padding_mode", "zeros"), bool(kwargs.get("align_corners", False)), backward)
        if _available("native", *options):
            backend = "native"
        elif _available("metal", *options):
            backend = "metal"
        else:
            raise NotImplementedError("grid_sampler_3d: neither native nor Metal supports this MPS call")
    if backend == "torch":
        return grid_sample_3d(input, grid, **kwargs)
    if backend == "metal":
        from ._grid_sample_3d_metal import grid_sample_3d_metal
        return grid_sample_3d_metal(input, grid, **kwargs)
    return F.grid_sample(input, grid, **kwargs)


def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    """Use the selected MPS sampler; default native failures fall back to CPU."""
    try:
        return grid_sample_for_probe(
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
