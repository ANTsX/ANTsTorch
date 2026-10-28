"""2D tool: recon-cohort-template --mean-method image, and the reference header on a 2D grid.

Subjects are 3D NIfTI volumes from which one slice is read (the normal 2D input)."""
import ants
import numpy as np
import pytest
import torch

from antstorch.lamnr_flows.core.lamnr_glow_tool_base import (
    _reference_on_working_grid, copy_reference_header)
from antstorch.lamnr_flows.tools.lamnr_glow_tool_2d import GlowTool2D

H, W, NZ = 8, 8, 6
SLICE = 3
CHANNELS = ("vent.nii.gz", "anatvent.nii.gz")


def _expected_input(volume):
    """Slice SLICE along axis 2, then the per-image min-max of GlowTool2D.read_image."""
    a = volume[:, :, SLICE]
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


@pytest.fixture
def cohort(tmp_path):
    rng = np.random.default_rng(1)
    volumes, firsts = [], []
    for k in range(3):
        d = tmp_path / f"sub{k}"
        d.mkdir()
        subject = []
        for c, name in enumerate(CHANNELS):
            v = (rng.random((H, W, NZ)) * (1.0 + 3.0 * k) + 0.5 * c + k).astype(np.float32)
            ants.image_write(ants.from_numpy(v, spacing=(2.0, 2.0, 2.0)), str(d / name))
            subject.append(v)
        volumes.append(subject)
        firsts.append(d / CHANNELS[0])
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("vent\n" + "\n".join(str(p) for p in firsts) + "\n")
    ckpt = tmp_path / "training_state.pt"
    torch.save({"config": {"intensity": "to01", "H": H, "W": W}}, ckpt)
    return dict(volumes=volumes, manifest=manifest, ckpt=ckpt, root=tmp_path)


def _run(cohort, out, extra=()):
    argv = ["--ckpt", str(cohort["ckpt"]), "--manifest", str(cohort["manifest"]),
            "--views", "vent", "--view-index", "0", "--devices", "cpu",
            "--slice-axis", "2", "--slice-index", str(SLICE),
            "--mean-method", "image", "--out", str(out), *extra]
    GlowTool2D().cmd_recon_cohort_template(argv)


def _read(path):
    return ants.image_read(str(path))


def test_two_channels_equal_the_mean_of_the_normalised_slices(cohort):
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out, ["--channel-files", *CHANNELS])
    for c, name in enumerate(CHANNELS):
        expected = np.mean([_expected_input(v[c]) for v in cohort["volumes"]], axis=0)
        got = _read(out.with_name(f"template_{name[:-len('.nii.gz')]}.nii.gz")).numpy()
        assert got.shape == (H, W)
        assert np.allclose(got, expected, atol=1e-5)


def test_single_channel_writes_one_file(cohort):
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out)
    expected = np.mean([_expected_input(v[0]) for v in cohort["volumes"]], axis=0)
    assert np.allclose(_read(out).numpy(), expected, atol=1e-5)


@pytest.mark.parametrize("mode", ["rescale", "clamp"])
def test_mean_is_saved_unchanged_in_both_out_range_modes(cohort, mode):
    """The 2D saver rescales in its default mode; the image mean must bypass that."""
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out, ["--out-range-mode", mode])
    expected = np.mean([_expected_input(v[0]) for v in cohort["volumes"]], axis=0)
    assert np.allclose(_read(out).numpy(), expected, atol=1e-5)


def test_no_model_is_built_or_loaded(cohort, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the image mean must not build a model")
    monkeypatch.setattr(GlowTool2D, "build_model", boom)
    monkeypatch.setattr(GlowTool2D, "prime_if_needed", boom)
    _run(cohort, cohort["root"] / "tpl" / "template.nii.gz")


def test_2d_reference_header_is_copied(cohort):
    ref = ants.from_numpy(np.zeros((H, W), dtype=np.float32), spacing=(3.0, 3.0), origin=(5.0, 6.0))
    ref_path = cohort["root"] / "ref2d.nii.gz"
    ants.image_write(ref, str(ref_path))
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out, ["--reference-image", str(ref_path)])
    got = _read(out)
    assert got.dimension == 2
    assert np.allclose(got.spacing, (3.0, 3.0))
    assert np.allclose(got.origin, (5.0, 6.0))


def test_3d_reference_is_refused_for_a_2d_output(cohort):
    ref = ants.from_numpy(np.zeros((H, W, NZ), dtype=np.float32), spacing=(3.0, 3.0, 3.0))
    ref_path = cohort["root"] / "ref3d.nii.gz"
    ants.image_write(ref, str(ref_path))
    with pytest.raises(ValueError, match="2D|3D"):
        _run(cohort, cohort["root"] / "tpl" / "template.nii.gz", ["--reference-image", str(ref_path)])


def test_invalid_method_is_rejected(cohort):
    with pytest.raises(SystemExit):
        GlowTool2D().cmd_recon_cohort_template(
            ["--ckpt", str(cohort["ckpt"]), "--manifest", str(cohort["manifest"]),
             "--views", "vent", "--out", str(cohort["root"] / "o.nii.gz"),
             "--mean-method", "median"])


# --------------------------------------------------------------------------- #
# Reference header mapped onto a 2D working grid
# --------------------------------------------------------------------------- #

def test_reference_mapped_onto_a_2d_working_grid():
    ref = ants.from_numpy(np.zeros((32, 18), dtype=np.float32), spacing=(4.0, 4.0))
    out = ants.from_numpy(np.zeros((16, 10), dtype=np.float32))
    copy_reference_header(out, ref)
    assert np.allclose(out.spacing, (8.0, 8.0))
    work = _reference_on_working_grid(ref, (16, 10))
    assert work.shape == (16, 10) and np.allclose(work.spacing, (8.0, 8.0))


def test_same_shape_2d_reference_is_copied_unchanged():
    ref = ants.from_numpy(np.zeros((6, 5), dtype=np.float32), spacing=(2.5, 2.5), origin=(1.0, 2.0))
    out = ants.from_numpy(np.zeros((6, 5), dtype=np.float32))
    copy_reference_header(out, ref)
    assert np.allclose(out.spacing, (2.5, 2.5)) and np.allclose(out.origin, (1.0, 2.0))
