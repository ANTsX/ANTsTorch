"""Separable pyramid blur (``gaussian_blur_replicate``) and the run_syn_registration CLI sigma-units flag (#91)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import ants
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from antstorch.syn.core.smoothing import gaussian_blur_replicate
from antstorch.syn.syn import _gaussian_blur_image

TOOL = Path(__file__).resolve().parents[2] / "tools" / "registration" / "run_syn_registration.py"


def _dense_blur(image, sigmas):
    """The pre-#91 dense N-D kernel (outer product of 1-D Gaussians, one conv)."""
    axes, radii = [], []
    for sigma in sigmas:
        radius = max(1, int(3.0 * sigma + 0.5))
        radii.append(radius)
        coordinate = torch.arange(-radius, radius + 1, dtype=image.dtype)
        kernel_1d = torch.exp(-0.5 * (coordinate / sigma).square())
        axes.append(kernel_1d / kernel_1d.sum())
    kernel = axes[0]
    for axis_kernel in axes[1:]:
        kernel = kernel.unsqueeze(-1) * axis_kernel.reshape((1,) * kernel.ndim + (-1,))
    kernel = kernel.reshape((1, 1) + kernel.shape).expand((image.shape[1], 1) + kernel.shape)
    padding = tuple(value for radius in reversed(radii) for value in (radius, radius))
    conv = F.conv2d if image.ndim == 4 else F.conv3d
    return conv(F.pad(image, padding, mode="replicate"), kernel, groups=image.shape[1])


@pytest.mark.parametrize("dim,sigmas", [(2, (1.0, 2.3)), (3, (0.6, 1.0, 1.7)), (3, (1.0, 1.0, 1.0))])
def test_gaussian_blur_replicate_equals_dense_kernel(dim, sigmas):
    gen = torch.Generator().manual_seed(dim)
    image = torch.randn((2, 3) + (13, 11, 9)[:dim], generator=gen, dtype=torch.float64)
    torch.testing.assert_close(gaussian_blur_replicate(image, sigmas), _dense_blur(image, sigmas), rtol=0, atol=1e-12)


def test_gaussian_blur_replicate_needs_one_sigma_per_axis():
    with pytest.raises(ValueError, match="one sigma per spatial axis"):
        gaussian_blur_replicate(torch.zeros(1, 1, 5, 5, 5), (1.0, 1.0))


@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0])
def test_pyramid_blur_equals_dense_kernel(sigma):
    gen = torch.Generator().manual_seed(1)
    image = torch.randn((1, 1, 12, 14, 10), generator=gen, dtype=torch.float64)
    blurred = _gaussian_blur_image(image, sigma)
    if sigma == 0:
        assert blurred is image
    else:
        torch.testing.assert_close(blurred, _dense_blur(image, (sigma,) * 3), rtol=0, atol=1e-12)


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_syn_registration", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Captured(Exception):
    pass


@pytest.mark.parametrize("extra,expected", [([], "vox"), (["--affine-smoothing-sigmas-units", "mm"], "mm")])
def test_cli_passes_affine_smoothing_sigmas_units(tmp_path, extra, expected):
    image = ants.from_numpy(np.random.default_rng(0).random((8, 8, 8)).astype(np.float32))
    ants.image_write(image, str(tmp_path / "f.nii.gz"))
    tool = _load_tool()
    captured = {}

    def fake_registration(**kwargs):
        captured.update(kwargs)
        raise _Captured

    argv = ["run_syn_registration.py", "--fixed", str(tmp_path / "f.nii.gz"), "--moving", str(tmp_path / "f.nii.gz"),
            "--output-dir", str(tmp_path / "out"), "--device", "cpu"] + extra
    with patch.object(sys, "argv", argv), patch.object(tool, "syn_registration", fake_registration):
        with pytest.raises(_Captured):
            tool.main()
    assert captured["affine_smoothing_sigmas_units"] == expected


def test_cli_rejects_unknown_affine_smoothing_sigmas_units(tmp_path):
    tool = _load_tool()
    with patch.object(sys, "argv", ["run_syn_registration.py", "--affine-smoothing-sigmas-units", "pixels"]):
        with pytest.raises(SystemExit):
            tool.parse_args()
