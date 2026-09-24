import ants
import numpy as np
import pytest

import antstorch


def _options(dimension):
    return {
        "device": "cpu",
        "shrink_factor": 1,
        "convergence": {"iters": [1], "tol": 0.0},
        "spline_param": (1,) * dimension,
        "number_of_histogram_bins": 16,
    }


def test_n4_bias_field_correction_preserves_geometry_and_returns_bias():
    array = np.linspace(1.0, 2.0, 80, dtype=np.float32).reshape(10, 8)
    image = ants.from_numpy(
        array,
        origin=(2.0, -3.0),
        spacing=(1.2, 1.7),
        direction=np.asarray(((0.0, -1.0), (1.0, 0.0))),
    )
    mask = image * 0 + 1

    corrected = antstorch.n4_bias_field_correction(
        image, mask=mask, **_options(2)
    )
    bias = antstorch.n4_bias_field_correction(
        image, mask=mask, return_bias_field=True, **_options(2)
    )

    assert ants.image_physical_space_consistency(image, corrected)
    assert ants.image_physical_space_consistency(image, bias)
    assert np.isfinite(corrected.numpy()).all()
    assert np.all(bias.numpy() > 0)
    np.testing.assert_allclose(corrected.numpy(), image.numpy() / bias.numpy())


def test_n4_bias_field_correction_rejects_mismatched_mask():
    image = ants.from_numpy(np.ones((8, 7), dtype=np.float32))
    mask = ants.from_numpy(np.ones((9, 7), dtype=np.float32))

    with pytest.raises(ValueError, match="same physical space"):
        antstorch.n4_bias_field_correction(image, mask=mask, **_options(2))
