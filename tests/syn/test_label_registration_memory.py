"""Memory/time refactors of label_image_registration (issue #91): behaviour must be unchanged.

Covers the one-pass label statistics (centroids), the ``label_msq``
multivariate metric (masks built from shared label maps), and the per-term
backward of multivariate SyN objectives.
"""

import gc
import weakref

import ants
import numpy as np
import pytest
import torch

import antstorch.syn.syn as syn_module
from antstorch import label_image_registration
from antstorch.syn import syn_registration
from antstorch.syn.label_registration import _label_centroids, _label_index_statistics


def _label_map(d=3, size=20, n=5, shift=0, seed=0, ids_step=3):
    rng = np.random.default_rng(seed)
    array = np.zeros((size,) * d, dtype=np.float32)
    for k in range(1, n + 1):
        center = rng.integers(4, size - 4, size=d)
        region = tuple(slice(int(center[axis]) - 3 + (shift if axis == 0 else 0),
                             int(center[axis]) + 3 + (shift if axis == 0 else 0)) for axis in range(d))
        array[region] = k * ids_step
    direction = np.eye(d)
    direction[:2, :2] = [[0, -1], [1, 0]]
    return ants.from_numpy(array, spacing=tuple(1 + .2 * i for i in range(d)),
                           origin=tuple(5 + i for i in range(d)), direction=direction)


def _argwhere_centroid(image, label):
    """The pre-#91 per-label centroid."""
    index = np.argwhere(image.numpy() == label).mean(axis=0)
    return np.asarray(image.origin) + np.asarray(image.direction) @ (index * np.asarray(image.spacing))


@pytest.mark.parametrize("d", [2, 3])
def test_one_pass_centroids_equal_argwhere_mean_exactly(d):
    image = _label_map(d=d, size=23, n=7, seed=d)
    counts, sums = _label_index_statistics(image.numpy())
    present = np.flatnonzero(counts)
    expected_present = np.unique(image.numpy()).astype(np.int64)
    np.testing.assert_array_equal(present, expected_present)
    ids = present[present != 0]
    for label, centroid in zip(ids, _label_centroids(image, (counts, sums), ids)):
        assert np.array_equal(centroid, _argwhere_centroid(image, float(label)))
        assert counts[label] == np.count_nonzero(image.numpy() == label)


def _mask_extras(fixed, moving, ids, weight=1.0):
    return [("MSQ", fixed == k, moving == k, weight, 0) for k in ids]


def _label_extras(fixed, moving, ids, weight=1.0):
    return [("label_msq", fixed, moving, weight, int(k)) for k in ids]


@pytest.mark.parametrize("levels,iterations,sigmas", [((1,), (4,), None), ((4, 2, 1), (3, 3, 2), None),
                                                     ((2, 1), (3, 2), (1.0, 0.5))])
def test_label_msq_matches_explicit_binary_masks(levels, iterations, sigmas):
    fixed, moving = _label_map(), _label_map(shift=1)
    ids = [3, 6, 9, 12, 15]
    eye, zero = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    common = dict(type_of_transform="SyNOnly", initial_affine=(eye, zero), primary_metric_weight=0,
                  syn_metric="mse", device="cpu", dtype=torch.float64, levels=levels,
                  reg_iterations=iterations, grad_step=0.2, smoothing_sigmas=sigmas)
    explicit = syn_registration(fixed, moving, multivariate_extras=_mask_extras(fixed, moving, ids), **common)
    on_the_fly = syn_registration(fixed, moving, multivariate_extras=_label_extras(fixed, moving, ids), **common)
    np.testing.assert_allclose(on_the_fly["loss_history"], explicit["loss_history"], rtol=0, atol=1e-14)
    for key in ("fwdtransforms", "invtransforms"):
        a = ants.image_read(explicit[key][-1 if key == "invtransforms" else 0]).numpy()
        b = ants.image_read(on_the_fly[key][-1 if key == "invtransforms" else 0]).numpy()
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-6)


def test_label_msq_validation():
    fixed, moving = _label_map(), _label_map(shift=1)
    with pytest.raises(ValueError, match="integer"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", primary_metric_weight=0,
                         multivariate_extras=[("label_msq", fixed, moving, 1.0, 2.5)], device="cpu")
    with pytest.raises(ValueError, match="2\\^24"):
        syn_registration(fixed, moving, type_of_transform="SyNOnly", primary_metric_weight=0,
                         multivariate_extras=[("LabelMSQ", fixed, moving, 1.0, 2**25)], device="cpu")


def test_multivariate_terms_are_backpropagated_one_at_a_time(monkeypatch):
    fixed, moving = _label_map(), _label_map(shift=1)
    ids = [3, 6, 9, 12, 15]
    sampled_inputs = []
    alive_counts = []
    original = syn_module.grid_sample_nd

    def spy(input, grid, *args, **kwargs):
        if grid.requires_grad and input.shape[2:] == (20, 20, 20):
            gc.collect()
            alive_counts.append(sum(ref() is not None for ref in sampled_inputs))
            sampled_inputs.append(weakref.ref(input))
        return original(input, grid, *args, **kwargs)

    monkeypatch.setattr(syn_module, "grid_sample_nd", spy)
    eye, zero = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    syn_registration(fixed, moving, type_of_transform="SyNOnly", initial_affine=(eye, zero), primary_metric_weight=0,
                     syn_metric="mse", device="cpu", dtype=torch.float64, levels=(1,), reg_iterations=(2,),
                     multivariate_extras=_label_extras(fixed, moving, ids))
    # Two samplings (fixed, moving) per label and iteration. Masks are built
    # on the fly, and each term's graph is released by its own backward: at
    # most the current label's fixed mask is still alive when its moving mask
    # is sampled (all five labels' masks were alive before).
    assert len(alive_counts) == 2 * len(ids) * 2
    assert max(alive_counts) <= 1


def test_label_image_registration_reports_the_same_common_ids_and_affine():
    fixed = [_label_map(), _label_map(n=3, seed=5)]
    moving = [_label_map(shift=1), _label_map(n=3, seed=5, shift=2)]
    result = label_image_registration(fixed, moving, initial_transforms="affine", type_of_deformable_transform=None,
                                      device="cpu", dtype=torch.float64)
    expected_ids, fixed_centers, moving_centers = [], [], []
    for a, b in zip(fixed, moving):
        ids = np.intersect1d(np.unique(a.numpy()), np.unique(b.numpy()))
        ids = ids[ids != 0]
        expected_ids.append(ids.tolist())
        fixed_centers += [_argwhere_centroid(a, k) for k in ids]
        moving_centers += [_argwhere_centroid(b, k) for k in ids]
    assert result["common_label_ids"] == expected_ids
    from antstorch.syn.label_registration import _landmark_affine
    matrix, translation = _landmark_affine(np.asarray(fixed_centers), np.asarray(moving_centers), "affine")
    np.testing.assert_allclose(np.asarray(result["affine_matrix"]), matrix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(result["affine_translation"]), translation, rtol=0, atol=1e-12)
