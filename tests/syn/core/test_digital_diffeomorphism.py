"""antstorch.syn.core.digital_diffeomorphism (port of syntx v6.39.0)."""
import numpy as np
import pytest
import torch

from antstorch.syn.core.digital_diffeomorphism import enforce_digital_diffeomorphism_nd
from antstorch.syn.core.inverse import update_inverse_field_nd
from antstorch.syn.core.jacobian import compute_jacobian_determinant_nd
from antstorch.syn.core.smoothing import get_boundary_mask

_META = dict(spacing=(1.0, 1.1, 0.9), origin=(2.0, -1.0, 0.5), direction=np.eye(3).flatten())


def _field(amplitude, seed=0, shape=(16, 18, 14)):
    gen = torch.Generator().manual_seed(seed)
    coarse = torch.randn((1, 3, 4, 4, 4), generator=gen, dtype=torch.float64)
    field = amplitude * torch.nn.functional.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    field = field.movedim(1, -1)
    return field * get_boundary_mask(shape, field.device, field.dtype)


def test_idempotent_on_a_regular_field():
    W = _field(0.3)
    W_inv = update_inverse_field_nd(W, None, steps=40, method="anderson", **_META)
    stats = {}
    f, i = enforce_digital_diffeomorphism_nd(W.clone(), W_inv.clone(), ice_tol=0.3, jac_floor=0.05, steps=50,
                                             stats=stats, **_META)
    assert stats["steps"] == 0 and torch.equal(f, W) and torch.equal(i, W_inv)


def test_unfolds_a_folding_field():
    W = _field(4.0)
    spacing_t = torch.tensor(_META["spacing"][::-1], dtype=torch.float64)
    assert float(compute_jacobian_determinant_nd(W, physical_spacing=spacing_t).min()) < 0
    W_inv = update_inverse_field_nd(W, None, steps=20, method="anderson", **_META)
    stats = {}
    enforce_digital_diffeomorphism_nd(W.clone(), W_inv.clone(), ice_tol=0.3, jac_floor=0.05, steps=50,
                                      stats=stats, **_META)
    assert stats["initial_min_det_fwd"] < 0 < stats["final_min_det_fwd"]
