"""Offline tensor-native N4 correction on a synthetic biased image."""
import torch
from antstorch.bspline_flows import ImageDomain, n4_bias_field_correction_tensor


def main():
    y, x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    mask = ((x.square() + y.square()) < 0.7).float()[None, None]
    image = (100 * torch.exp(0.3 * x))[None, None] * mask
    domain = ImageDomain(size=(64, 64), spacing=(1., 1.), origin=(0., 0.),
                         direction=((1., 0.), (0., 1.)))
    corrected = n4_bias_field_correction_tensor(
        image, domain=domain, mask=mask, shrink_factor=2,
        convergence={"iters": [5, 5], "tol": 0.0}, spline_param=(4, 4),
    )
    assert corrected.shape == image.shape
    assert torch.isfinite(corrected).all()
    print("Corrected tensor:", tuple(corrected.shape))
    print("Short demonstration only; tune convergence for real data.")


if __name__ == "__main__":
    main()
