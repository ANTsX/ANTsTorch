from statistics import NormalDist

import numpy as np
import pytest
import torch

from antstorch.lamnr_flows.core.lamnr_glow_tool_base import (
    GlowToolBase,
    _marginal_variance,
    _parse_edit_levels,
    _per_level_values,
)


def _gaussian_blob(shapes, means, covariances):
    dims = [int(np.prod(shape)) for shape in shapes]
    return {
        "views": ["view"],
        "dims_per_level_per_view": [dims],
        "shapes_by_view": [shapes],
        "level_view_slices": [[(0, dimension)] for dimension in dims],
        "L": len(shapes),
        "mu": [np.asarray(mean, dtype=np.float64) for mean in means],
        "Sigma": covariances,
    }


def _edit(z_list, blob, levels, mode, **kwargs):
    return GlowToolBase.edit_latents_to_mean(
        None,
        z_list,
        blob,
        "view",
        levels,
        mode=mode,
        **kwargs,
    )


def test_parse_edit_levels_supports_spaces_commas_and_all():
    assert _parse_edit_levels(["0", "2"], 3) == [0, 2]
    assert _parse_edit_levels(["0,2"], 3) == [0, 2]
    assert _parse_edit_levels(["all"], 3) == [0, 1, 2]
    assert _parse_edit_levels(["none"], 3) == []

    with pytest.raises(ValueError, match="more than once"):
        _parse_edit_levels(["0", "0"], 3)
    with pytest.raises(ValueError, match="out of range"):
        _parse_edit_levels(["3"], 3)


def test_per_level_quantiles_are_positionally_matched():
    assert _per_level_values([2, 0], 0.99, None, "edit-quantiles") == {
        2: 0.99,
        0: 0.99,
    }
    assert _per_level_values(
        [2, 0], 0.99, [0.95, 0.975], "edit-quantiles"
    ) == {2: 0.95, 0: 0.975}

    with pytest.raises(ValueError, match="one value per"):
        _per_level_values([0, 1], 0.99, [0.95], "edit-quantiles")


def test_winsorize_uses_gaussian_marginals_at_multiple_levels():
    shapes = [(1, 1, 2), (1, 1, 1)]
    blob = _gaussian_blob(
        shapes,
        means=[[1.0, -1.0], [2.0]],
        covariances=[
            np.asarray([4.0, 1.0]),
            np.asarray([[9.0]]),
        ],
    )
    z_list = [
        torch.tensor([[[[100.0, -100.0]]]]),
        torch.tensor([[[[100.0]]]]),
    ]
    quantiles = {0: 0.95, 1: 0.75}

    edited = _edit(
        z_list, blob, [0, 1], "winsorize", quantiles=quantiles
    )

    k0 = NormalDist().inv_cdf(0.95)
    k1 = NormalDist().inv_cdf(0.75)
    expected0 = torch.tensor([1.0 + 2.0 * k0, -1.0 - k0])
    expected1 = torch.tensor([2.0 + 3.0 * k1])
    assert torch.allclose(edited[0].reshape(-1), expected0)
    assert torch.allclose(edited[1].reshape(-1), expected1)


@pytest.mark.parametrize("shape", [(1, 2, 2), (1, 2, 2, 2)])
def test_shrink_is_dimension_agnostic_and_leaves_other_levels_untouched(shape):
    dimension = int(np.prod(shape))
    mean = np.arange(dimension, dtype=np.float64)
    blob = _gaussian_blob(
        [shape, shape],
        means=[mean, mean],
        covariances=[np.ones(dimension), np.ones(dimension)],
    )
    z_list = [
        torch.full((2, *shape), 10.0),
        torch.full((2, *shape), -3.0),
    ]

    edited = _edit(
        z_list, blob, [0], "shrink", shrink_strength=0.25
    )

    mu = torch.as_tensor(mean, dtype=z_list[0].dtype).view(1, *shape)
    expected = mu + 0.25 * (z_list[0] - mu)
    assert torch.allclose(edited[0], expected)
    assert edited[1] is z_list[1]


def test_sample_is_reproducible_and_temperature_zero_returns_mean():
    shape = (1, 1, 4)
    blob = _gaussian_blob(
        [shape],
        means=[[1.0, 2.0, 3.0, 4.0]],
        covariances=[np.asarray([1.0, 4.0, 9.0, 16.0])],
    )
    z_list = [torch.zeros(3, *shape)]

    first = _edit(
        z_list, blob, [0], "sample",
        sample_temperature=0.8, sample_seed=17,
    )
    repeated = _edit(
        z_list, blob, [0], "sample",
        sample_temperature=0.8, sample_seed=17,
    )
    different = _edit(
        z_list, blob, [0], "sample",
        sample_temperature=0.8, sample_seed=18,
    )
    at_mean = _edit(
        z_list, blob, [0], "sample",
        sample_temperature=0.0, sample_seed=17,
    )

    assert torch.equal(first[0], repeated[0])
    assert not torch.equal(first[0], different[0])
    expected_mean = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, *shape)
    assert torch.equal(at_mean[0], expected_mean.expand_as(at_mean[0]))


def test_lowrank_sample_and_marginal_variance_do_not_require_dense_covariance():
    shape = (1, 1, 4)
    U = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, -0.5],
        ]
    )
    covariance = {
        "type": "lowrank",
        "U": U,
        "eig": np.asarray([4.0, 9.0]),
        "sigma2": 0.25,
    }
    blob = _gaussian_blob(
        [shape],
        means=[np.zeros(4)],
        covariances=[covariance],
    )

    variance = _marginal_variance(covariance, 0, 4)
    expected = (U * U * np.asarray([4.0, 9.0])).sum(axis=1) + 0.25
    np.testing.assert_allclose(variance, expected)

    z_list = [torch.zeros(2, *shape)]
    edited = _edit(
        z_list, blob, [0], "sample",
        sample_temperature=1.0, sample_seed=7,
    )
    assert edited[0].shape == z_list[0].shape
    assert torch.isfinite(edited[0]).all()


@pytest.mark.parametrize(
    "mode,kwargs,error",
    [
        ("winsorize", {"quantiles": {0: 0.5}}, "strictly between"),
        ("shrink", {"shrink_strength": -0.1}, "between 0 and 1"),
        ("sample", {"sample_temperature": -0.1}, "non-negative"),
    ],
)
def test_new_edit_parameter_validation(mode, kwargs, error):
    shape = (1, 1, 1)
    blob = _gaussian_blob([shape], means=[[0.0]], covariances=[np.ones(1)])
    with pytest.raises(ValueError, match=error):
        _edit([torch.zeros(1, *shape)], blob, [0], mode, **kwargs)