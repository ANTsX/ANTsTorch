"""recon-cohort-template --mean-method image: voxel-wise mean of the real inputs.

The method needs no network: the checkpoint is only read for its configuration."""
import ants
import numpy as np
import pytest
import torch

from antstorch.lamnr_flows.tools.lamnr_glow_tool_3d import GlowTool3D

SHAPE = (8, 6, 8)
CHANNELS = ("vent.nii.gz", "anatvent.nii.gz")


def _expected_input(arr):
    """Per-subject, per-channel min-max normalisation used by GlowTool3D.read_image."""
    a = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.clip(a, 1e-5, 1.0 - 1e-5)


@pytest.fixture
def cohort(tmp_path):
    """Three subjects with two channels each, a manifest, and a stub checkpoint."""
    rng = np.random.default_rng(0)
    arrays, firsts = [], []
    for k in range(3):
        d = tmp_path / f"sub{k}"
        d.mkdir()
        subject = []
        for c, name in enumerate(CHANNELS):
            # Different scale and offset per subject so that the normalisation matters.
            a = (rng.random(SHAPE) * (1.0 + 4.0 * k) + 0.5 * c + k).astype(np.float32)
            img = ants.from_numpy(a, spacing=(2.0, 2.0, 2.0), origin=(1.0, 2.0, 3.0))
            ants.image_write(img, str(d / name))
            subject.append(a)
        arrays.append(subject)
        firsts.append(d / CHANNELS[0])
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("vent\n" + "\n".join(str(p) for p in firsts) + "\n")
    ckpt = tmp_path / "training_state.pt"
    torch.save({"config": {"intensity": "to01"}}, ckpt)
    return dict(arrays=arrays, manifest=manifest, ckpt=ckpt, root=tmp_path)


def _run(cohort, out, extra=()):
    argv = ["--ckpt", str(cohort["ckpt"]), "--manifest", str(cohort["manifest"]),
            "--views", "vent", "--view-index", "0", "--devices", "cpu",
            "--spatial-dims", *map(str, SHAPE), "--mean-method", "image",
            "--out", str(out), *extra]
    GlowTool3D().cmd_recon_cohort_template(argv)


def _read(path):
    return ants.image_read(str(path))


def test_two_channels_equal_the_mean_of_the_normalised_inputs(cohort):
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out, ["--channel-files", *CHANNELS])
    for c, name in enumerate(CHANNELS):
        expected = np.mean([_expected_input(s[c]) for s in cohort["arrays"]], axis=0)
        got = _read(out.with_name(f"template_{name[:-len('.nii.gz')]}.nii.gz")).numpy()
        assert got.shape == SHAPE
        assert np.allclose(got, expected, atol=1e-5)


def test_single_channel_writes_one_file(cohort):
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out)
    expected = np.mean([_expected_input(s[0]) for s in cohort["arrays"]], axis=0)
    assert np.allclose(_read(out).numpy(), expected, atol=1e-5)


def test_output_stays_in_unit_interval(cohort):
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out, ["--channel-files", *CHANNELS])
    for name in CHANNELS:
        a = _read(out.with_name(f"template_{name[:-len('.nii.gz')]}.nii.gz")).numpy()
        assert a.min() >= 0.0 and a.max() <= 1.0


def test_no_model_is_built_or_loaded(cohort, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the image mean must not build a model")
    monkeypatch.setattr(GlowTool3D, "build_model", boom)
    monkeypatch.setattr(GlowTool3D, "prime_if_needed", boom)
    _run(cohort, cohort["root"] / "tpl" / "template.nii.gz")


def test_reference_image_header_is_copied(cohort):
    ref = ants.from_numpy(np.zeros(SHAPE, dtype=np.float32),
                          spacing=(3.0, 3.0, 3.0), origin=(5.0, 6.0, 7.0))
    ref_path = cohort["root"] / "ref.nii.gz"
    ants.image_write(ref, str(ref_path))
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out, ["--reference-image", str(ref_path)])
    got = _read(out)
    assert np.allclose(got.spacing, (3.0, 3.0, 3.0))
    assert np.allclose(got.origin, (5.0, 6.0, 7.0))


def test_invalid_method_is_rejected(cohort):
    with pytest.raises(SystemExit):
        GlowTool3D().cmd_recon_cohort_template(
            ["--ckpt", str(cohort["ckpt"]), "--manifest", str(cohort["manifest"]),
             "--views", "vent", "--out", str(cohort["root"] / "o.nii.gz"),
             "--mean-method", "median"])


def test_unsupported_intensity_checkpoint_is_still_refused(cohort):
    torch.save({"config": {"intensity": "0mean"}}, cohort["ckpt"])
    with pytest.raises(NotImplementedError):
        _run(cohort, cohort["root"] / "tpl" / "template.nii.gz")


def test_mean_is_not_rescaled_by_the_default_out_range_mode(cohort):
    """The mean of the normalised inputs never reaches 1 everywhere; the default
    'rescale' mode must not stretch it."""
    out = cohort["root"] / "tpl" / "template.nii.gz"
    _run(cohort, out, ["--out-range-mode", "rescale"])
    expected = np.mean([_expected_input(s[0]) for s in cohort["arrays"]], axis=0)
    got = _read(out).numpy()
    assert expected.max() < 1.0 - 1e-3 or expected.min() > 1e-3
    assert np.allclose(got, expected, atol=1e-5)
