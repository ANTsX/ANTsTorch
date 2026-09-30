import math

import numpy as np
import pytest
import torch

from antstorch.lamnr_flows.core.lamnr_glow_tool_base import (
    _pullback_path_length,
    _slerp_through_endpoints,
)

SHAPES = [(2, 2, 2), (1, 4, 4)]  # two levels, 8 + 16 = 24 dims


def _levels(rng):
    return [rng.standard_normal(int(np.prod(s))) for s in SHAPES]


def _linear_decoder(scale):
    """Toy 'decoder': concatenate the flattened levels and scale them."""
    def decode(z_list):
        return scale * torch.cat([z.reshape(z.shape[0], -1) for z in z_list], dim=1)
    return decode


def test_lerp_length_equals_scaled_latent_distance_for_linear_decoder():
    rng = np.random.default_rng(0)
    z0, z1 = _levels(rng), _levels(rng)
    length, chord = _pullback_path_length(_linear_decoder(2.0), z0, z1, SHAPES, steps=8)
    expected = 2.0 * np.linalg.norm(np.concatenate(z1) - np.concatenate(z0))
    assert length == pytest.approx(expected, rel=1e-5)
    assert chord == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("chunk", [1, 2, 4, 9, 50])
def test_result_does_not_depend_on_chunk_size(chunk):
    rng = np.random.default_rng(1)
    z0, z1 = _levels(rng), _levels(rng)
    decode = lambda zl: torch.tanh(_linear_decoder(1.0)(zl))  # nonlinear
    ref = _pullback_path_length(decode, z0, z1, SHAPES, steps=8, chunk=9)
    got = _pullback_path_length(decode, z0, z1, SHAPES, steps=8, chunk=chunk)
    assert got[0] == pytest.approx(ref[0], rel=1e-6)
    assert got[1] == pytest.approx(ref[1], rel=1e-6)


def test_path_length_is_at_least_the_chord_and_converges():
    rng = np.random.default_rng(2)
    z0, z1 = _levels(rng), _levels(rng)
    decode = lambda zl: torch.tanh(3.0 * _linear_decoder(1.0)(zl))
    l8, chord = _pullback_path_length(decode, z0, z1, SHAPES, steps=8)
    l64, _ = _pullback_path_length(decode, z0, z1, SHAPES, steps=64)
    l128, _ = _pullback_path_length(decode, z0, z1, SHAPES, steps=128)
    assert l8 >= chord - 1e-6
    assert l8 <= l64 + 1e-6  # refining a polyline never shortens it
    assert l128 == pytest.approx(l64, rel=1e-2)


def test_slerp_passes_through_endpoints_and_matches_arc_length():
    rng = np.random.default_rng(3)
    mu = [rng.standard_normal(int(np.prod(s))) for s in SHAPES]
    # Endpoints at the same radius about mu on every level -> the path is an arc.
    z0, z1 = [], []
    for m in mu:
        a, b = rng.standard_normal(m.size), rng.standard_normal(m.size)
        z0.append(m + 3.0 * a / np.linalg.norm(a))
        z1.append(m + 3.0 * b / np.linalg.norm(b))
    for l in range(len(SHAPES)):
        np.testing.assert_allclose(_slerp_through_endpoints(0.0, z0[l], z1[l], mu[l]), z0[l], atol=1e-12)
        np.testing.assert_allclose(_slerp_through_endpoints(1.0, z0[l], z1[l], mu[l]), z1[l], atol=1e-12)
    length, _ = _pullback_path_length(_linear_decoder(1.0), z0, z1, SHAPES,
                                      steps=256, path="slerp", mu_levels=mu)
    # All levels move together: the joint path length is sqrt(sum_l (r * theta_l)^2).
    arcs = []
    for l in range(len(SHAPES)):
        u0 = (z0[l] - mu[l]) / 3.0
        u1 = (z1[l] - mu[l]) / 3.0
        arcs.append(3.0 * math.acos(float(np.clip(u0 @ u1, -1, 1))))
    assert length == pytest.approx(math.sqrt(sum(a * a for a in arcs)), rel=1e-3)


def test_invalid_arguments():
    rng = np.random.default_rng(4)
    z0, z1 = _levels(rng), _levels(rng)
    dec = _linear_decoder(1.0)
    with pytest.raises(ValueError):
        _pullback_path_length(dec, z0, z1, SHAPES, steps=0)
    with pytest.raises(ValueError):
        _pullback_path_length(dec, z0, z1, SHAPES, path="geodesic")
    with pytest.raises(ValueError):
        _pullback_path_length(dec, z0, z1, SHAPES, path="slerp")  # no mu
