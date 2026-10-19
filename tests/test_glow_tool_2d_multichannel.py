"""Multichannel support in the LAM-Flow 2D inference tool (lamnr_glow_tool_2d)."""
import argparse

import ants
import numpy as np
import pytest
import torch

from antstorch.lamnr_flows.tools.lamnr_glow_tool_2d import GlowTool2D, _coerce_nchw_4d

H, W, D = 16, 16, 8
CPU = torch.device("cpu")
CFG = {"H": H, "W": W, "K": 2, "L": 2, "hidden": 8}


@pytest.fixture()
def subject(tmp_path):
    rng = np.random.default_rng(0)
    for name, offset in (("vent", 1.0), ("anatvent", 0.0)):
        img = ants.from_numpy((rng.random((H, W, D)) + offset).astype("float32"))
        ants.image_write(img, str(tmp_path / f"{name}.nii.gz"))
    return tmp_path / "vent.nii.gz"


def _tool(argv, cfg=None):
    cfg = {**CFG, **(cfg or {})}
    tool = GlowTool2D()
    parser = argparse.ArgumentParser()
    tool._add_size_arg(parser)
    args = parser.parse_args(argv + ["--slice-axis", "2", "--slice-index", "4"])
    size = tool._get_target_size(args, cfg)
    model = tool.build_model(cfg, CPU, size)
    tool.prime_if_needed(model, size, CPU)
    return tool, model, size


def test_two_channels_are_read_decoded_and_saved_separately(subject, tmp_path):
    tool, model, size = _tool(["--channel-files", "vent.nii.gz", "anatvent.nii.gz"])
    x = tool.read_image(subject, size).unsqueeze(0)
    assert tool._n_channels() == 2 and x.shape == (1, 2, H, W)
    assert not torch.allclose(x[0, 0], x[0, 1])

    with torch.no_grad():
        z, _ = model.inverse_and_log_det(x)
        decoded = tool.decode_latents(model, z if isinstance(z, list) else [z], size)
    assert decoded.shape == (1, 2, H, W)  # channels are not averaged

    out = tmp_path / "out" / "template.nii.gz"
    out.parent.mkdir()
    tool.save_volume(decoded, out)
    assert sorted(p.name for p in out.parent.iterdir()) == [
        "template_anatvent.nii.gz", "template_vent.nii.gz"]


def test_png_grids_are_written_per_channel(subject, tmp_path):
    tool, _, _ = _tool(["--channel-files", "vent.nii.gz", "anatvent.nii.gz"])
    tool.save_volume(torch.rand(3, 2, H, W), tmp_path / "grid.png", nrow=3)
    assert (tmp_path / "grid_vent.png").exists() and (tmp_path / "grid_anatvent.png").exists()


def test_single_channel_behaviour_is_unchanged(subject, tmp_path):
    tool, _, size = _tool([])
    x = tool.read_image(subject, size).unsqueeze(0)
    assert tool._n_channels() == 1 and x.shape == (1, 1, H, W)
    assert _coerce_nchw_4d(torch.rand(1, 2, H, W)).shape[1] == 1  # default still averages
    assert _coerce_nchw_4d(torch.rand(1, 3, H, W)).shape[1] == 3  # RGB still kept
    tool.save_volume(x, tmp_path / "one.png")
    assert (tmp_path / "one.png").exists()  # no channel suffix


def test_inconsistent_channel_arguments_are_rejected():
    with pytest.raises(ValueError):
        _tool(["--channels", "3", "--channel-files", "a.nii.gz", "b.nii.gz"])


def test_reference_image_header_is_copied(tmp_path):
    ref = ants.from_numpy(np.zeros((H, W), dtype="float32"), origin=(1.0, 2.0), spacing=(2.0, 3.0))
    ref_path = tmp_path / "reference.nii.gz"
    ants.image_write(ref, str(ref_path))
    tool, _, _ = _tool(["--channel-files", "vent.nii.gz", "anatvent.nii.gz",
                        "--reference-image", str(ref_path)])
    tool.save_volume(torch.rand(1, 2, H, W), tmp_path / "out.nii.gz")
    for name in ("vent", "anatvent"):
        out = ants.image_read(str(tmp_path / f"out_{name}.nii.gz"))
        assert out.spacing == pytest.approx(ref.spacing)
        assert out.origin == pytest.approx(ref.origin)


def test_reference_image_dimension_must_match_the_output(tmp_path):
    ref = ants.from_numpy(np.zeros((H, W, D), dtype="float32"))
    ref_path = tmp_path / "reference3d.nii.gz"
    ants.image_write(ref, str(ref_path))
    tool, _, _ = _tool(["--reference-image", str(ref_path)])
    with pytest.raises(ValueError):
        tool.save_volume(torch.rand(1, 1, H, W), tmp_path / "out.nii.gz")


def test_missing_reference_image_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        _tool(["--reference-image", str(tmp_path / "nope.nii.gz")])
