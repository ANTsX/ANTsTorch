"""
Shared helpers for the ANTsTorch architecture-only test suite.

These tests exercise antstorch's architecture builders directly (no
pretrained weights, no ANTs image I/O) to guard against import- and
shape-level regressions in the modules ported from ANTsPyNet during the
2026-08 lung/mouse/WMH porting effort:

    * antstorch.utilities.lung_extraction
    * antstorch.utilities.lung_segmentation
    * antstorch.utilities.mouse
    * antstorch.utilities.white_matter_hyperintensity_segmentation

Each test rebuilds the exact same architecture (same class, same kwargs)
that the corresponding application function builds internally, feeds it
a small random tensor with the right number of channels, and checks that
the forward pass produces a finite tensor of the expected shape with a
value range consistent with the model's output activation (sigmoid ->
[0, 1]; softmax -> [0, 1] and sums to 1 along the channel axis).

No pretrained weights are loaded, so these tests do NOT validate
numerical correctness against the original Keras models -- that is
covered separately by tools/convert_antspynet_weights_to_antstorch.py
and tools/convert_wmh_bespoke.py, which each reload a converted .pt file
and compare against a bit-exact round trip.

A few branches (protonLobes/maskLobes, ct, mouse_brain_parcellation)
size their input channels from ANTsXNet prior-image counts that are
only known once the real template/prior data is loaded at runtime.
Where that's the case, the test uses a small placeholder count
(documented in a comment next to it) -- enough to exercise the real
architecture wiring (attentionGating, create_multihead_unet_model_3d,
etc.) without requiring the real ANTsXNet data files.
"""

import torch

# Tests always run on CPU, regardless of get_default_device(): a couple
# of the generic U-Net's deconvolution paths are known to be unstable on
# MPS (see the existing @pytest.mark.skipif(torch.backends.mps.is_available(), ...)
# pattern in tests/test_brain_extraction.py), so pinning to CPU keeps
# this architecture-only suite deterministic and portable.
DEVICE = torch.device("cpu")


def assert_finite(t):
    assert torch.isfinite(t).all(), "output contains NaN/Inf"


def assert_sigmoid_range(t, atol=1e-6):
    assert_finite(t)
    assert t.min().item() >= -atol, f"sigmoid output below 0: min={t.min().item()}"
    assert t.max().item() <= 1.0 + atol, f"sigmoid output above 1: max={t.max().item()}"


def assert_softmax_output(t, channel_dim=1, atol=1e-4):
    """Values in [0, 1] and sum to ~1 along `channel_dim`."""
    assert_sigmoid_range(t)
    sums = t.sum(dim=channel_dim)
    assert torch.allclose(sums, torch.ones_like(sums), atol=atol), (
        f"softmax output does not sum to 1 along dim {channel_dim} "
        f"(min={sums.min().item()}, max={sums.max().item()})"
    )
