"""--intensity for the LAMNr Glow 2D/3D trainers (and the shared ChannelNormalizer)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

import ants
import antstorch
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import _require_to01_intensity
from antstorch.lamnr_flows.core.train_lamnr_glow_base import (
    _extract_views_from_batch,
    _prime_if_needed,
)
from antstorch.lamnr_flows.misc import ChannelNormalizer
from antstorch.lamnr_flows.scripts import train_lamnr_glow_2d as glow2d
from antstorch.lamnr_flows.scripts.train_lamnr_glow_3d import LAMNrGlow3DTrainer


def _signed_fields(tmp_path, n_subjects=6, size=16):
    """Two views of signed 2-channel fields (u, v) saved as .npy, subject-prefixed."""
    rng = np.random.default_rng(0)
    for view, offset in (("woj", 0.0), ("wj", 0.5)):
        (tmp_path / view).mkdir()
        for s in range(n_subjects):
            field = np.stack([
                offset + rng.normal(size=(size, size)),
                -1.5 + 0.5 * rng.normal(size=(size, size)),
            ]).astype("float32")
            np.save(tmp_path / view / f"s{s:02d}_flow.npy", field)
    return [[str(tmp_path / "woj" / "*.npy")], [str(tmp_path / "wj" / "*.npy")]]


# ---------------------------------------------------------------------------
# ChannelNormalizer
# ---------------------------------------------------------------------------

def test_channel_normalizer_batch_and_sample_forms_agree():
    rng = np.random.default_rng(1)
    batches = [torch.from_numpy(rng.normal([[[[3.0]], [[-2.0]]]], 2.0, size=(4, 2, 5, 5)).astype("float32"))
               for _ in range(3)]
    by_batch = ChannelNormalizer("0mean").fit(batches, channel_dim=1)
    by_sample = ChannelNormalizer("0mean").fit((x for b in batches for x in b), channel_dim=0)
    np.testing.assert_allclose(by_batch._shift, by_sample._shift, rtol=1e-6)
    np.testing.assert_allclose(by_batch._scale, by_sample._scale, rtol=1e-6)
    z = by_batch.transform_batch(batches[0])
    torch.testing.assert_close(z[0], by_batch.transform(batches[0][0]))
    torch.testing.assert_close(by_batch.inverse_transform_batch(z), batches[0], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="channel count"):
        ChannelNormalizer().fit([torch.zeros(2, 3), torch.zeros(3, 3)])


def test_hybrid_trainer_uses_shared_normalizer():
    from antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid import (
        ChannelNormalizer as HybridChannelNormalizer,
        SignalNormalizer,
    )
    assert HybridChannelNormalizer is ChannelNormalizer
    assert SignalNormalizer is ChannelNormalizer


# ---------------------------------------------------------------------------
# ImageDataset(normalize_intensity=False)
# ---------------------------------------------------------------------------

def test_image_dataset_can_keep_absolute_intensities_2d():
    rng = np.random.default_rng(2)
    images = [[ants.from_numpy((300 + 50 * rng.normal(size=(16, 16))).astype("float32"))]
              for _ in range(3)]
    template = images[0][0]
    raw = antstorch.ImageDataset(images=images, template=template, do_data_augmentation=False,
                                 normalize_intensity=False, number_of_samples=3)
    normalized = antstorch.ImageDataset(images=images, template=template, do_data_augmentation=False,
                                        number_of_samples=3)
    x_raw = raw[0][0] if isinstance(raw[0], (tuple, list)) else raw[0]
    x_norm = normalized[0][0] if isinstance(normalized[0], (tuple, list)) else normalized[0]
    assert float(x_raw.mean()) > 100          # HU-like values survive
    assert 0.0 <= float(x_norm.min()) and float(x_norm.max()) <= 1.0  # default unchanged


# ---------------------------------------------------------------------------
# 2D loaders and trainer
# ---------------------------------------------------------------------------

def test_2d_loaders_read_npy_fields_and_skip_augmentation(tmp_path):
    views = _signed_fields(tmp_path)
    train_loader, val_loader, _ = glow2d.build_loaders_from_globs(
        view_specs=views, H=16, W=16, train_samples=4, val_samples=4, batch=4,
        num_workers=0, slice_idx=0, val_frac=0.2, subject_limit=None,
        do_aug=True, intensity="0mean",
    )
    assert train_loader.dataset.do_aug is False  # no flip / noise / clamp
    xs = _extract_views_from_batch(next(iter(train_loader)), num_views=2)
    assert xs[0].shape[1:] == (2, 16, 16)
    assert float(xs[0][:, 1].mean()) < -1.0      # signed values survive (no clamp to [0, 1])
    assert float(xs[1][:, 0].mean()) > 0.2


def _tiny_2d_args(views, out_dir, intensity, extra=()):
    argv = []
    for view in views:
        argv += ["--view", *view]
    argv += [
        "--H", "16", "--W", "16", "--L", "1", "--K", "1", "--hidden", "8",
        "--batch", "4", "--train-samples", "8", "--val-samples", "4",
        "--max-iter", "2", "--eval-interval", "2", "--devices", "cpu",
        "--out-dir", str(out_dir), "--align", "vicreg", "--val-frac", "0.2",
        "--num-workers", "0", "--sample-mode", "off", "--intensity", intensity,
        *extra,
    ]
    return glow2d._build_args(argv)


def test_2d_trainer_zero_mean_end_to_end_and_resume_guard(tmp_path):
    views = _signed_fields(tmp_path)
    out = tmp_path / "run"
    trainer = glow2d.LAMNrGlow2DTrainer()
    trainer.setup(_tiny_2d_args(views, out, "0mean"))
    normalizers = trainer.intensity_normalizers
    assert len(normalizers) == 2
    assert normalizers[0]._shift[1] == pytest.approx(-1.5, abs=0.1)
    assert normalizers[1]._shift[0] == pytest.approx(0.5, abs=0.15)
    x = trainer.extract_view(next(iter(trainer.train_loader)), 0, torch.device("cpu"))
    assert abs(float(x.mean())) < 0.5 and 0.5 < float(x.std()) < 1.5
    trainer.train()
    blob = torch.load(out / "training_state.pt", map_location="cpu", weights_only=False)
    assert blob["config"]["intensity"] == "0mean"
    assert [n["kind"] for n in blob["intensity_normalizers"]] == ["glow2d", "glow2d"]

    resumed = glow2d.LAMNrGlow2DTrainer()
    resumed.setup(_tiny_2d_args(views, out, "0mean", ["--auto-resume", "--max-iter", "3"]))
    np.testing.assert_allclose(resumed.intensity_normalizers[0]._shift, normalizers[0]._shift)

    with pytest.raises(ValueError, match="--intensity 0mean"):
        glow2d.LAMNrGlow2DTrainer().setup(
            _tiny_2d_args(views, out, "none", ["--auto-resume", "--max-iter", "4"])
        )


def test_2d_trainer_default_to01_is_unchanged(tmp_path):
    class Args:
        num_views = 1
        intensity = "to01"

    trainer = glow2d.LAMNrGlow2DTrainer()
    trainer.args = Args()
    batch = [torch.randn(2, 3, 8, 8) * 5]
    x = trainer.extract_view(batch, 0, torch.device("cpu"))
    torch.testing.assert_close(x, batch[0])  # 2D historically returns the batch as loaded


# ---------------------------------------------------------------------------
# 3D extract_view
# ---------------------------------------------------------------------------

def test_3d_extract_view_intensity_modes():
    class Args:
        num_views = 1
        intensity = "to01"

    trainer = LAMNrGlow3DTrainer()
    trainer.args = Args()
    trainer.model_dtype = torch.float32
    trainer.input_shape = (1, 4, 4, 4)
    batch = 300.0 + 50.0 * torch.randn(2, 1, 4, 4, 4)
    x = trainer.extract_view(batch, 0, torch.device("cpu"))
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0  # historical to01

    trainer.args.intensity = "none"
    torch.testing.assert_close(trainer.extract_view(batch, 0, torch.device("cpu")), batch)

    trainer.args.intensity = "0mean"
    trainer.fit_intensity_normalizers([batch], num_views=1)
    z = trainer.extract_view(batch, 0, torch.device("cpu"))
    assert abs(float(z.mean())) < 1e-4 and float(z.std()) == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------------------
# Helpers and tool guard
# ---------------------------------------------------------------------------

def test_prime_with_explicit_spatial_dims_keeps_two_channel_2d_batches():
    seen = []

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def inverse_and_log_det(self, x):
            seen.append(tuple(x.shape))
            return x, torch.zeros(x.shape[0])

    _prime_if_needed(Recorder(), torch.randn(3, 2, 8, 8), spatial_dims=2)
    assert seen == [(1, 2, 8, 8)]  # not mistaken for a channel-less 3D volume
    _prime_if_needed(Recorder(), torch.randn(3, 8, 8, 8), spatial_dims=3)
    assert seen[-1] == (1, 1, 8, 8, 8)


def test_tools_refuse_non_to01_checkpoints():
    _require_to01_intensity({}, "legacy.pt")
    _require_to01_intensity({"intensity": "to01"}, "ok.pt")
    with pytest.raises(NotImplementedError, match="'0mean'"):
        _require_to01_intensity({"intensity": "0mean"}, "flow.pt")


# ---------------------------------------------------------------------------
# --dequantize (Glow 2D/3D)
# ---------------------------------------------------------------------------

def _grid_fields(tmp_path, levels=49, n_subjects=6, size=16, channels=3):
    """Two views of soft one-hot-like maps whose values lie on {0, 1/K, ..., 1}."""
    rng = np.random.default_rng(4)
    for view in ("a", "b"):
        (tmp_path / view).mkdir()
        for s in range(n_subjects):
            counts = rng.integers(0, levels + 1, size=(channels, size, size))
            counts[:, :4] = 0  # plenty of exact zeros, like background
            np.save(tmp_path / view / f"s{s:02d}_seg.npy", (counts / levels).astype("float32"))
    return [[str(tmp_path / "a" / "*.npy")], [str(tmp_path / "b" / "*.npy")]]


def test_dequantize_deterministic_mapping_and_training_jitter():
    class Args:
        num_views = 1
        intensity = "none"
        dequantize = 49

    trainer = glow2d.LAMNrGlow2DTrainer()
    trainer.args = Args()
    n = torch.arange(50, dtype=torch.float32)
    batch = [(n / 49).view(1, 1, 1, 50).repeat(64, 1, 1, 1)]
    x = trainer.extract_view(batch, 0, torch.device("cpu"))
    torch.testing.assert_close(x[0, 0, 0], (n + 0.5) / 50)
    torch.manual_seed(0)
    jittered = trainer.jitter_dequantization(x)
    assert torch.all(jittered >= n / 50) and torch.all(jittered < (n + 1) / 50)
    assert jittered.std(dim=0).min() > 0

    trainer.args.dequantize = 0
    assert torch.equal(trainer.jitter_dequantization(x), x)  # disabled -> no-op

    trainer.args.dequantize = 49
    trainer.args.intensity = "to01"
    with pytest.raises(ValueError, match="requires --intensity none"):
        trainer.extract_view(batch, 0, torch.device("cpu"))


def test_dequantize_rejects_values_outside_unit_interval():
    class Args:
        num_views = 1
        intensity = "none"
        dequantize = 255

    trainer = glow2d.LAMNrGlow2DTrainer()
    trainer.args = Args()
    with pytest.raises(ValueError, match=r"expects view values on \[0, 1\]"):
        trainer._check_dequantization_range([[torch.full((2, 1, 4, 4), 200.0)]], num_views=1)
    trainer._check_dequantization_range([[torch.rand(2, 1, 4, 4)]], num_views=1)


def test_2d_trainer_dequantize_end_to_end_and_resume_guard(tmp_path):
    views = _grid_fields(tmp_path)
    out = tmp_path / "run"
    trainer = glow2d.LAMNrGlow2DTrainer()
    trainer.setup(_tiny_2d_args(views, out, "none", ["--dequantize", "49"]))
    x = trainer.extract_view(next(iter(trainer.train_loader)), 0, torch.device("cpu"))
    assert float(x.min()) >= 0.5 / 50 - 1e-6 and float(x.max()) <= 49.5 / 50 + 1e-6
    trainer.train()
    blob = torch.load(out / "training_state.pt", map_location="cpu", weights_only=False)
    assert blob["config"]["dequantize"] == 49
    summary = (out / "run_config.txt").read_text()
    assert "dequantize (levels)" in summary
    with pytest.raises(ValueError, match="--dequantize 49"):
        glow2d.LAMNrGlow2DTrainer().setup(
            _tiny_2d_args(views, out, "none", ["--auto-resume", "--max-iter", "3"])
        )


def test_3d_extract_view_dequantize():
    class Args:
        num_views = 1
        intensity = "none"
        dequantize = 255

    trainer = LAMNrGlow3DTrainer()
    trainer.args = Args()
    trainer.model_dtype = torch.float32
    trainer.input_shape = (1, 4, 4, 4)
    batch = torch.randint(0, 256, (2, 1, 4, 4, 4)).float() / 255
    x = trainer.extract_view(batch, 0, torch.device("cpu"))
    torch.testing.assert_close(x, (torch.round(batch * 255) + 0.5) / 256)
