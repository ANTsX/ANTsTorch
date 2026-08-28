import pytest
import torch

from antstorch.bspline_flows import (
    ImageDomain,
    PhysicalGradientDescent,
    gaussian_svf_registration,
)


def test_identity_registration_contract_and_inverse():
    domain = ImageDomain((9, 7))
    image = torch.rand(1, 1, 7, 9)
    result = gaussian_svf_registration(image, image, domain, iterations=0, squaring_steps=2)

    assert result["velocity"].shape == (1, 2, 7, 9)
    assert torch.count_nonzero(result["velocity"]) == 0
    assert torch.allclose(result["warpedmovout"], image, atol=1e-6)
    assert torch.count_nonzero(result["fwdtransforms"]) == 0
    assert torch.count_nonzero(result["invtransforms"]) == 0
    assert result["loss_history"] == []


def test_dense_gaussian_svf_reduces_shifted_image_loss():
    domain = ImageDomain((16, 16))
    fixed = torch.zeros(1, 1, 16, 16)
    fixed[:, :, 5:11, 5:11] = 1
    moving = torch.roll(fixed, shifts=1, dims=-1)

    initial = gaussian_svf_registration(fixed, moving, domain, iterations=0, squaring_steps=2)["loss"]
    result = gaussian_svf_registration(
        fixed,
        moving,
        domain,
        iterations=8,
        optimizer=PhysicalGradientDescent(0.2, momentum=0.5),
        update_field_sigma=1.0,
        total_field_sigma=0.1,
        squaring_steps=2,
        padding_mode="border",
    )

    assert result["loss"] < initial
    assert len(result["loss_history"]) == 8
    assert torch.count_nonzero(result["velocity"][:, :, (0, -1), :]) == 0
    assert torch.count_nonzero(result["velocity"][:, :, :, (0, -1)]) == 0


def test_multiresolution_and_verbose_output(capsys):
    domain = ImageDomain((12, 10), spacing=(1.2, 1.5))
    image = torch.rand(1, 1, 10, 12)
    result = gaussian_svf_registration(
        image,
        image,
        domain,
        iterations=(0, 0),
        shrink_factors=(2, 1),
        smoothing_sigmas=(1, 0),
        verbose=True,
    )
    output = capsys.readouterr().out

    assert "ANTsTorch Gaussian SVF registration configuration:" in output
    assert "update_field_sigma: 3.0" in output
    assert "Resolution level 1/2" in output
    assert "velocity_parameters=" in output
    assert result["velocity"].shape == (1, 2, 10, 12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"update_field_sigma": -1}, "update_field_sigma"),
        ({"total_field_sigma": float("inf")}, "total_field_sigma"),
        ({"optimizer": "adam"}, "optimizer"),
        ({"similarity": "bad"}, "similarity"),
    ],
)
def test_validation(kwargs, message):
    domain = ImageDomain((6, 6))
    image = torch.zeros(1, 1, 6, 6)
    with pytest.raises((TypeError, ValueError), match=message):
        gaussian_svf_registration(image, image, domain, iterations=0, **kwargs)
