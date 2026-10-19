"""Multichannel support in the LAM-Flow 3D inference tool (lamnr_glow_tool_3d)."""
import argparse

import ants
import numpy as np
import pytest
import torch

from antstorch.lamnr_flows.tools.lamnr_glow_tool_3d import GlowTool3D, _coerce_nchwd_5d

H, W, D = 16, 8, 16
CPU = torch.device("cpu")
CFG = {"K": 2, "L": 2, "hidden": 8}


@pytest.fixture()
def subject(tmp_path):
    rng = np.random.default_rng(0)
    for name, offset in (("vent", 1.0), ("anatvent", 0.0)):
        img = ants.from_numpy((rng.random((H, W, D)) + offset).astype("float32"))
        ants.image_write(img, str(tmp_path / f"{name}.nii.gz"))
    return tmp_path / "vent.nii.gz"


def _tool(argv, cfg=None):
    tool = GlowTool3D()
    parser = argparse.ArgumentParser()
    tool._add_size_arg(parser)
    args = parser.parse_args(argv + ["--spatial-dims", str(H), str(W), str(D)])
    size = tool._get_target_size(args, cfg or {})
    model = tool.build_model({**CFG, **(cfg or {})}, CPU, size)
    tool.prime_if_needed(model, size, CPU)
    return tool, model, size


def test_two_channels_are_read_decoded_and_saved_separately(subject, tmp_path):
    tool, model, size = _tool(["--channel-files", "vent.nii.gz", "anatvent.nii.gz"])
    x = tool.read_image(subject, size).unsqueeze(0)
    assert tool._n_channels() == 2 and x.shape == (1, 2, H, W, D)
    assert not torch.allclose(x[0, 0], x[0, 1])

    with torch.no_grad():
        z, _ = model.inverse_and_log_det(x)
        decoded = tool.decode_latents(model, z if isinstance(z, list) else [z], size)
    assert decoded.shape == (1, 2, H, W, D)  # channels are not averaged

    out = tmp_path / "out" / "template.nii.gz"
    out.parent.mkdir()
    tool.save_volume(decoded, out)
    assert sorted(p.name for p in out.parent.iterdir()) == [
        "template_anatvent.nii.gz", "template_vent.nii.gz"]
    assert ants.image_read(str(out.parent / "template_vent.nii.gz")).shape == (H, W, D)


def test_channels_from_checkpoint_config_when_no_channel_files():
    tool, _, _ = _tool([], cfg={"C": 2})
    assert tool._n_channels() == 2
    assert tool._channel_labels() == ["ch0", "ch1"]


def test_single_channel_behaviour_is_unchanged(subject, tmp_path):
    tool, _, size = _tool([])
    x = tool.read_image(subject, size).unsqueeze(0)
    assert tool._n_channels() == 1 and x.shape == (1, 1, H, W, D)
    assert _coerce_nchwd_5d(torch.rand(1, 2, H, W, D)).shape[1] == 1  # still averages
    tool.save_volume(x, tmp_path / "one.nii.gz")
    assert (tmp_path / "one.nii.gz").exists()  # no channel suffix


def test_inconsistent_channel_arguments_are_rejected():
    with pytest.raises(ValueError):
        _tool(["--channels", "3", "--channel-files", "a.nii.gz", "b.nii.gz"])


def _reference(tmp_path):
    direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ref = ants.from_numpy(np.zeros((H, W, D), dtype="float32"),
                          origin=(1.0, 2.0, 3.0), spacing=(2.0, 3.0, 4.0), direction=direction)
    path = tmp_path / "reference.nii.gz"
    ants.image_write(ref, str(path))
    return ref, path


def test_reference_image_header_is_copied_to_every_channel(tmp_path):
    ref, ref_path = _reference(tmp_path)
    tool, _, _ = _tool(["--channel-files", "vent.nii.gz", "anatvent.nii.gz",
                        "--reference-image", str(ref_path)])
    tool.save_volume(torch.rand(1, 2, H, W, D), tmp_path / "out.nii.gz")
    for name in ("vent", "anatvent"):
        out = ants.image_read(str(tmp_path / f"out_{name}.nii.gz"))
        assert out.spacing == pytest.approx(ref.spacing)
        assert out.origin == pytest.approx(ref.origin)
        assert np.allclose(out.direction, ref.direction)


def test_header_is_untouched_without_reference_image(tmp_path):
    tool, _, _ = _tool([])
    tool.save_volume(torch.rand(1, 1, H, W, D), tmp_path / "out.nii.gz")
    out = ants.image_read(str(tmp_path / "out.nii.gz"))
    assert out.spacing == pytest.approx((1.0, 1.0, 1.0))


def test_missing_reference_image_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        _tool(["--reference-image", str(tmp_path / "nope.nii.gz")])
