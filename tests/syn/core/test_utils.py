import numpy as np
import pytest
import torch

from antstorch.syn.core import (
    normalize_tensor,
    auto_select_intensity_percentiles,
    normalize_image,
)


def test_normalize_tensor_minmax_maps_to_unit_interval():
    tensor = torch.tensor([-3.0, 0.0, 2.0, 7.0], dtype=torch.double)
    normalized = normalize_tensor(tensor, method='minmax')
    torch.testing.assert_close(normalized.min(), torch.tensor(0.0, dtype=torch.double), atol=1e-6, rtol=0)
    torch.testing.assert_close(normalized.max(), torch.tensor(1.0, dtype=torch.double), atol=1e-6, rtol=0)


def test_normalize_tensor_zscore_is_zero_mean_unit_variance():
    torch.manual_seed(0)
    tensor = torch.randn(1000, dtype=torch.double) * 5.0 + 3.0
    normalized = normalize_tensor(tensor, method='zscore')
    assert normalized.mean().abs().item() < 1e-6
    torch.testing.assert_close(normalized.std(unbiased=False), torch.tensor(1.0, dtype=torch.double), atol=1e-6, rtol=0)


def test_normalize_tensor_robust_clamps_to_unit_interval():
    tensor = torch.linspace(0.0, 100.0, 101, dtype=torch.double)
    normalized = normalize_tensor(tensor, method='robust', p_min=1.0, p_max=99.0)
    assert normalized.min().item() >= 0.0
    assert normalized.max().item() <= 1.0


def test_normalize_tensor_l2_unit_norm():
    tensor = torch.tensor([3.0, 4.0], dtype=torch.double)
    normalized = normalize_tensor(tensor, method='l2')
    torch.testing.assert_close(torch.linalg.vector_norm(normalized, ord=2), torch.tensor(1.0, dtype=torch.double))


def test_normalize_tensor_sigmoid_is_bounded():
    tensor = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.double)
    normalized = normalize_tensor(tensor, method='sigmoid')
    assert normalized.min().item() >= 0.0
    assert normalized.max().item() <= 1.0
    torch.testing.assert_close(normalized[1], torch.tensor(0.5, dtype=torch.double))


def test_normalize_tensor_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_tensor(torch.zeros(4), method='bogus')


def test_auto_select_intensity_percentiles_returns_ordered_pair_for_small_input():
    percentiles = auto_select_intensity_percentiles(np.zeros((2, 2)))
    assert percentiles == (2.0, 98.0)


def test_auto_select_intensity_percentiles_orders_low_below_high():
    rng = np.random.default_rng(0)
    image = rng.random((32, 32, 32)).astype(np.float32)
    p_low, p_high = auto_select_intensity_percentiles(image)
    assert p_low < p_high


def test_normalize_image_minmax_numpy_roundtrip():
    array = np.array([[-2.0, 0.0], [4.0, 10.0]], dtype=np.float32)
    normalized = normalize_image(array, method='minmax')
    assert isinstance(normalized, np.ndarray)
    assert normalized.min() == pytest.approx(0.0, abs=1e-6)
    assert normalized.max() == pytest.approx(1.0, abs=1e-6)


def test_normalize_image_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_image(np.zeros((2, 2)), method='bogus')


# --- Ported from syntx: idempotency guard ---


def test_normalize_image_is_idempotent_on_already_normalized_input():
    rng = np.random.default_rng(0)
    array = rng.random((16, 16)).astype(np.float32)
    once = normalize_image(array, method='auto')
    twice = normalize_image(once, method='auto')
    np.testing.assert_allclose(twice, np.clip(once, 0.0, 1.0), atol=1e-6)


def test_normalize_image_force_recomputes_even_when_already_normalized():
    # A pre-normalized array with an active [0, 1] range would otherwise be
    # returned as-is; force=True re-derives percentile statistics instead.
    array = np.array([[0.1, 0.2], [0.6, 0.9]], dtype=np.float32)
    forced = normalize_image(array, method='minmax', force=True)
    assert forced.min() == pytest.approx(0.0, abs=1e-6)
    assert forced.max() == pytest.approx(1.0, abs=1e-5)


def test_normalize_image_does_not_short_circuit_unnormalized_input():
    # An input outside [0, 1] (or with a degenerate active range) must still
    # go through full normalization, not the idempotency fast path.
    array = np.array([[-2.0, 0.0], [4.0, 10.0]], dtype=np.float32)
    normalized = normalize_image(array, method='minmax')
    assert normalized.min() == pytest.approx(0.0, abs=1e-6)
    assert normalized.max() == pytest.approx(1.0, abs=1e-6)
