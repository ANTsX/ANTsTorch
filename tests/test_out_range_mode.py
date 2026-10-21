"""--out-range-mode ({rescale,clamp}) of the 2D/3D inference tools, the reference-header
mapping onto the working grid, and the options of the template sub-commands."""
import argparse

import ants
import numpy as np
import pytest
import torch

from antstorch.lamnr_flows.core.lamnr_glow_tool_base import (
    _reference_on_working_grid, copy_reference_header)
from antstorch.lamnr_flows.tools.lamnr_glow_tool_2d import (
    GlowTool2D, _coerce_nchw_4d, save_grid_2d)
from antstorch.lamnr_flows.tools.lamnr_glow_tool_3d import GlowTool3D, _coerce_nchwd_5d


def _out_of_range(shape, seed=0):
    """Random tensor spanning roughly [-0.5, 1.5]."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g) * 2.0 - 0.5


# --------------------------------------------------------------------------- #
# _coerce_nchwd_5d (3D) and _coerce_nchw_4d (2D)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("coerce,shape", [
    (_coerce_nchwd_5d, (1, 1, 8, 6, 8)),
    (_coerce_nchw_4d, (1, 1, 8, 8)),
])
class TestCoerce:
    def test_in_range_output_is_untouched_in_both_modes(self, coerce, shape):
        x = torch.rand(*shape)
        for mode in ("rescale", "clamp"):
            assert torch.allclose(coerce(x, range_mode=mode), x)

    def test_clamp_only_clips(self, coerce, shape):
        x = _out_of_range(shape)
        out = coerce(x, range_mode="clamp")
        assert torch.allclose(out, x.clamp(0.0, 1.0))
        inside = (x >= 0.0) & (x <= 1.0)
        assert torch.equal(out[inside], x[inside])  # in-range voxels keep their value

    def test_rescale_changes_the_intensity_scale(self, coerce, shape):
        x = _out_of_range(shape)
        out = coerce(x, range_mode="rescale")
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert not torch.allclose(out, x.clamp(0.0, 1.0))

    def test_default_is_rescale(self, coerce, shape):
        x = _out_of_range(shape)
        assert torch.allclose(coerce(x), coerce(x, range_mode="rescale"))

    def test_tolerance_ignores_tiny_excursions(self, coerce, shape):
        x = torch.rand(*shape).clamp(0.01, 0.99)
        x.view(-1)[0] = -5e-4          # inside the 1e-3 tolerance
        x.view(-1)[1] = 1.0 + 5e-4
        out = coerce(x, range_mode="rescale")
        assert torch.allclose(out, x.clamp(0.0, 1.0))  # clamped, not rescaled

    def test_range_line_is_logged_to_stderr(self, coerce, shape, capsys):
        coerce(_out_of_range(shape), range_mode="clamp")
        err = capsys.readouterr().err
        assert "[to01]" in err and "out-of-[0,1]=" in err
        assert "clamped only" in err or "mode=clamp" in err


def test_3d_log_says_whether_rescale_was_applied(capsys):
    x = _out_of_range((1, 1, 8, 6, 8))
    _coerce_nchwd_5d(x, range_mode="rescale")
    assert "APPLIED" in capsys.readouterr().err
    _coerce_nchwd_5d(torch.rand(1, 1, 8, 6, 8), range_mode="rescale")
    assert "not applied" in capsys.readouterr().err


def test_multichannel_ranges_are_logged_per_channel(capsys):
    x = _out_of_range((1, 2, 8, 6, 8))
    out = _coerce_nchwd_5d(x, keep_channels=True, range_mode="clamp")
    assert out.shape[1] == 2 and torch.allclose(out, x.clamp(0.0, 1.0))
    err = capsys.readouterr().err
    assert "c0=" in err and "c1=" in err


# --------------------------------------------------------------------------- #
# save_grid_2d
# --------------------------------------------------------------------------- #

def _saved_array(x, path, **kwargs):
    save_grid_2d(x, path, nrow=1, **kwargs)
    return ants.image_read(str(path)).numpy()


def test_save_grid_2d_clamp_keeps_the_scale(tmp_path):
    x = _out_of_range((1, 1, 12, 12))
    arr = _saved_array(x, tmp_path / "a.nii.gz", range_mode="clamp")
    assert np.allclose(arr, x.clamp(0.0, 1.0).numpy(), atol=1e-6)


def test_save_grid_2d_rescale_is_the_default_and_stretches(tmp_path):
    x = _out_of_range((1, 1, 12, 12))
    default = _saved_array(x, tmp_path / "d.nii.gz")
    rescaled = _saved_array(x, tmp_path / "r.nii.gz", range_mode="rescale")
    assert np.allclose(default, rescaled)
    assert not np.allclose(rescaled, x.clamp(0.0, 1.0).numpy())


def test_save_grid_2d_rescale_stretches_even_in_range_images(tmp_path):
    x = torch.rand(1, 1, 12, 12) * 0.5 + 0.1       # in [0.1, 0.6]
    rescaled = _saved_array(x, tmp_path / "r.nii.gz", range_mode="rescale")
    clamped = _saved_array(x, tmp_path / "c.nii.gz", range_mode="clamp")
    assert rescaled.max() > 0.9                     # stretched
    assert np.allclose(clamped, x.numpy(), atol=1e-6)


# --------------------------------------------------------------------------- #
# GlowToolBase._finalize_range (templates)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tool_cls,shape", [
    (GlowTool3D, (1, 2, 8, 6, 8)),
    (GlowTool2D, (1, 2, 8, 8)),
])
class TestFinalizeRange:
    def test_clamp(self, tool_cls, shape, capsys):
        tool = tool_cls()
        tool._out_range_mode = "clamp"
        x = _out_of_range(shape)
        assert torch.allclose(tool._finalize_range(x, "unit"), x.clamp(0.0, 1.0))
        assert "[to01] unit" in capsys.readouterr().err

    def test_rescale_is_the_default_and_matches_to01(self, tool_cls, shape):
        from antstorch.lamnr_flows.core.lamnr_glow_tool_base import to01
        tool = tool_cls()
        x = _out_of_range(shape)
        assert torch.allclose(tool._finalize_range(x), to01(x, winsorize=True))

    def test_non_finite_values_do_not_propagate(self, tool_cls, shape):
        tool = tool_cls()
        tool._out_range_mode = "clamp"
        x = torch.rand(*shape)
        x.view(-1)[0] = float("nan")
        x.view(-1)[1] = float("inf")
        assert torch.isfinite(tool._finalize_range(x)).all()


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tool_cls,extra", [
    (GlowTool3D, ["--spatial-dims", "8", "8", "8"]),
    (GlowTool2D, []),
])
def test_cli_option(tool_cls, extra):
    tool = tool_cls()

    def parse(argv):
        parser = argparse.ArgumentParser()
        tool._add_size_arg(parser)
        return parser.parse_args(argv + extra)

    assert parse([]).out_range_mode == "rescale"
    assert parse(["--out-range-mode", "clamp"]).out_range_mode == "clamp"
    with pytest.raises(SystemExit):
        parse(["--out-range-mode", "bogus"])


@pytest.mark.parametrize("tool_cls", [GlowTool3D, GlowTool2D])
def test_get_target_size_records_the_mode(tool_cls):
    tool = tool_cls()
    parser = argparse.ArgumentParser()
    tool._add_size_arg(parser)
    argv = ["--out-range-mode", "clamp"]
    argv += ["--spatial-dims", "8", "8", "8"] if tool_cls is GlowTool3D else []
    args = parser.parse_args(argv)
    cfg = {} if tool_cls is GlowTool3D else {"H": 8, "W": 8}
    tool._get_target_size(args, cfg)
    assert tool._out_range_mode == "clamp"


class _Stop(Exception):
    pass


def _declared_options(monkeypatch, tool, command):
    """Option strings a sub-command declares, without running it."""
    seen = {}

    def fake_parse_args(self, *a, **k):
        seen["options"] = set(self._option_string_actions)
        raise _Stop

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    with pytest.raises(_Stop):
        getattr(tool, command)([])
    return seen["options"]


@pytest.mark.parametrize("tool_cls", [GlowTool3D, GlowTool2D])
def test_template_commands_options(monkeypatch, tool_cls):
    tool = tool_cls()
    template = _declared_options(monkeypatch, tool, "cmd_recon_template")
    cohort = _declared_options(monkeypatch, tool, "cmd_recon_cohort_template")
    assert "--out-range-mode" in template and "--out-range-mode" in cohort
    assert "--sharpen-image" in template            # applied by recon-template
    assert "--sharpen-image" not in cohort          # was never used, removed


@pytest.mark.parametrize("tool_cls", [GlowTool3D, GlowTool2D])
def test_recon_declares_reference_image_once(monkeypatch, tool_cls):
    # A duplicate add_argument would raise argparse.ArgumentError before parsing.
    options = _declared_options(monkeypatch, tool_cls(), "cmd_recon")
    assert "--reference-image" in options and "--out-range-mode" in options


# --------------------------------------------------------------------------- #
# Reference header mapped onto the working grid
# --------------------------------------------------------------------------- #

def _image(shape, spacing, origin=(0.0, 0.0, 0.0)):
    return ants.from_numpy(np.zeros(shape, dtype="float32"), spacing=spacing, origin=origin)


def test_reference_header_is_mapped_when_shapes_differ():
    ref = _image((32, 18, 32), (4.0, 4.0, 4.0))
    out = _image((16, 10, 16), (1.0, 1.0, 1.0))
    copy_reference_header(out, ref)
    assert out.spacing == pytest.approx((8.0, 8.0, 8.0))   # factor 0.5: true spacing, not 4.0
    assert out.shape == (16, 10, 16)


def test_reference_header_is_copied_unchanged_when_shapes_match():
    ref = _image((8, 6, 8), (2.0, 3.0, 4.0), origin=(1.0, 2.0, 3.0))
    out = _image((8, 6, 8), (1.0, 1.0, 1.0))
    copy_reference_header(out, ref)
    assert out.spacing == pytest.approx(ref.spacing)
    assert out.origin == pytest.approx(ref.origin)


def test_reference_header_none_leaves_the_image_alone():
    out = _image((8, 6, 8), (1.5, 1.5, 1.5))
    assert copy_reference_header(out, None).spacing == pytest.approx((1.5, 1.5, 1.5))


def test_reference_header_dimension_mismatch_raises():
    ref = ants.from_numpy(np.zeros((8, 8), dtype="float32"))
    with pytest.raises(ValueError):
        copy_reference_header(_image((8, 6, 8), (1.0, 1.0, 1.0)), ref)


def test_reference_on_working_grid_has_the_requested_shape():
    ref = _image((32, 18, 32), (4.0, 4.0, 4.0))
    grid = _reference_on_working_grid(ref, (16, 10, 16))
    assert grid.shape == (16, 10, 16)
    assert grid.spacing == pytest.approx((8.0, 8.0, 8.0))
