import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from antstorch.lamnr_flows.misc import LatentAlignmentLossManager
from antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid import (
    HybridLAMNrTrainer,
    HybridManifestDataset,
    ChannelNormalizer,
    HybridViewSpec,
    SignalNormalizer,
    _augment_image_group,
    _augment_signal_group,
    _dequantize,
    _load_signal,
    _signal_augmentation_config,
    _build_args,
    _scheduled_augmentation_config,
    build_manifest_from_config,
    hybrid_collate,
)
from antstorch.lamnr_flows.scripts.train_lamnr_flows_tabular import TabularNormalizer


def test_hybrid_view_spec_validation():
    tab = HybridViewSpec.from_dict(
        {"name": "tau", "type": "tabular", "columns": ["b1", "b2", "b3"]}
    )
    vol = HybridViewSpec.from_dict(
        {"name": "t1", "type": "image3d", "path_column": "t1", "shape": [8, 8, 8]}
    )
    assert tab.kind == "tabular"
    assert vol.shape == (8, 8, 8)
    with pytest.raises(ValueError, match="at least two"):
        HybridViewSpec.from_dict(
            {"name": "amyloid", "type": "tabular", "columns": ["centiloid"]}
        )


def test_hybrid_cli_rejects_deepest_alignment(tmp_path):
    with pytest.raises(SystemExit):
        _build_args([
            "--manifest", str(tmp_path / "manifest.csv"),
            "--config", str(tmp_path / "views.json"),
            "--alignment-latents", "deepest",
        ])


def test_hybrid_dataset_and_missing_masks(tmp_path):
    image2d = tmp_path / "image2d.npy"
    image3d = tmp_path / "image3d.npy"
    np.save(image2d, np.random.default_rng(1).normal(size=(8, 8)).astype("float32"))
    np.save(image3d, np.random.default_rng(2).normal(size=(8, 8, 8)).astype("float32"))
    frame = pd.DataFrame({
        "im2": [str(image2d), ""],
        "im3": [str(image3d), str(image3d)],
        "b1": [1.0, np.nan], "b2": [2.0, np.nan], "b3": [3.0, np.nan],
    })
    views = [
        HybridViewSpec.from_dict(
            {"name": "im2", "type": "image2d", "path_column": "im2", "shape": [8, 8]}
        ),
        HybridViewSpec.from_dict(
            {"name": "im3", "type": "image3d", "path_column": "im3", "shape": [8, 8, 8]}
        ),
        HybridViewSpec.from_dict(
            {"name": "tau", "type": "tabular", "columns": ["b1", "b2", "b3"]}
        ),
    ]
    normalizer = TabularNormalizer("0mean").fit(np.array([[1.0, 2.0, 3.0]]))
    dataset = HybridManifestDataset(frame, views, {"tau": normalizer})
    batch_values, batch_masks = hybrid_collate([dataset[0], dataset[1]])
    assert [tuple(v.shape) for v in batch_values] == [
        (2, 1, 8, 8), (2, 1, 8, 8, 8), (2, 3)
    ]
    assert [m.tolist() for m in batch_masks] == [
        [True, False], [True, True], [True, False]
    ]
    virtual = HybridManifestDataset(
        frame, views, {"tau": normalizer}, number_of_samples=5
    )
    assert len(virtual) == 5
    assert torch.equal(virtual[2][1], virtual[0][1])


def test_manifest_can_be_outer_joined_from_globs_and_csv(tmp_path):
    for key in ("s1", "s2"):
        np.save(tmp_path / f"{key}_T1.npy", np.ones((4, 4), dtype="float32"))
    table = tmp_path / "tau.csv"
    pd.DataFrame({
        "id": ["s2", "s3"], "b1": [1.0, 2.0], "b2": [3.0, 4.0]
    }).to_csv(table, index=False)
    views = [
        HybridViewSpec.from_dict({
            "name": "T1", "type": "image2d", "path_column": "T1File",
            "shape": [4, 4], "glob": str(tmp_path / "*_T1.npy"),
            "key_regex": r"/(s\d)_T1",
        }),
        HybridViewSpec.from_dict({
            "name": "tau", "type": "tabular", "columns": ["b1", "b2"],
            "csv": str(table), "key_column": "id",
        }),
    ]
    manifest = build_manifest_from_config(views, {"join_key": "sample"})
    assert manifest["sample"].tolist() == ["s1", "s2", "s3"]
    assert pd.isna(manifest.loc[0, "b1"])
    assert pd.isna(manifest.loc[2, "T1File"])


def test_grouped_flip_is_shared_across_modalities():
    image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    augmented = _augment_image_group(
        [image, image * 2],
        {
            "enabled": True,
            "horizontal_flip_probability": 1.0,
            "sd_affine": 0.0,
            "sd_deformation": 0.0,
            "sd_simulated_bias_field": 0.0,
            "sd_histogram_warping": 0.0,
            "noise_parameters": [0.0, 0.0],
        },
    )
    assert torch.equal(augmented[0], torch.flip(image, dims=(-1,)))
    assert torch.equal(augmented[1], 2 * augmented[0])


def test_ants_spatial_augmentation_is_shared_across_modalities():
    torch.manual_seed(3)
    image = torch.zeros(1, 16, 16)
    image[:, 4:12, 6:10] = 1.0
    augmented = _augment_image_group(
        [image, image.clone()],
        {
            "enabled": True,
            "transform_type": "affine",
            "horizontal_flip_probability": 0.0,
            "sd_affine": 0.02,
            "sd_deformation": 0.0,
            "sd_simulated_bias_field": 0.0,
            "sd_histogram_warping": 0.0,
            "noise_model": "additivegaussian",
            "noise_parameters": [0.0, 0.0],
        },
    )
    assert torch.allclose(augmented[0], augmented[1], atol=1e-6)


def test_hybrid_augmentation_schedule_updates_all_specialized_parameters():
    config = {
        "noise_parameters": [0.0, 1.0],
        "schedules": (
            "noise_std:linear:1->0@10,sd_affine:linear:2->0@10,"
            "sd_deformation:linear:3->0@10,"
            "sd_simulated_bias_field:linear:4->0@10,"
            "sd_histogram_warping:linear:5->0@10,"
            "tabular_noise_std:linear:0.5->0@10"
        ),
    }
    midpoint = _scheduled_augmentation_config(config, 5)
    assert midpoint["noise_parameters"][1] == pytest.approx(0.5)
    assert midpoint["sd_affine"] == pytest.approx(1.0)
    assert midpoint["sd_deformation"] == pytest.approx(1.5)
    assert midpoint["sd_simulated_bias_field"] == pytest.approx(2.0)
    assert midpoint["sd_histogram_warping"] == pytest.approx(2.5)
    assert midpoint["tabular_noise_std"] == pytest.approx(0.25)


def test_masked_alignment_uses_pairwise_intersections(tmp_path):
    args = _build_args([
        "--manifest", str(tmp_path / "unused.csv"),
        "--config", str(tmp_path / "unused.json"),
        "--align", "mse", "--align-warmup", "0",
    ])
    projectors = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
    manager = LatentAlignmentLossManager(args, projectors, torch.device("cpu"))
    latents = [torch.randn(5, 4, requires_grad=True) for _ in range(3)]
    masks = [
        torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
        torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool),
        torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool),
    ]
    total, alignment, _, _ = manager.compute(
        latents, torch.tensor(1.0, requires_grad=True), 1, None, None, masks=masks
    )
    assert torch.isfinite(total)
    assert torch.isfinite(alignment)
    assert alignment.item() > 0
    total.backward()
    assert all(z.grad is not None for z in latents[:2])


def test_hybrid_trainer_real_flow_smoke(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for index in range(4):
        path2 = tmp_path / f"im2_{index}.npy"
        path3 = tmp_path / f"im3_{index}.npy"
        np.save(path2, rng.normal(size=(8, 8)).astype("float32"))
        np.save(path3, rng.normal(size=(8, 8, 8)).astype("float32"))
        rows.append({
            "subject": f"s{index}", "im2": str(path2), "im3": str(path3),
            "b1": 1.0 + index, "b2": 2.0 + index, "b3": 3.0 + index,
        })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = tmp_path / "views.json"
    config.write_text(json.dumps({
        "subject_column": "subject",
        "views": [
            {"name": "im2", "type": "image2d", "path_column": "im2",
             "shape": [8, 8], "model": {"L": 1, "K": [1], "hidden": [4],
              "glowbase_logscale_factor": 1.0, "glowbase_min_log": -5.0,
              "glowbase_max_log": 5.0}},
            {"name": "im3", "type": "image3d", "path_column": "im3",
             "shape": [8, 8, 8], "model": {"L": 1, "K": [1], "hidden": [4]}},
            {"name": "tau", "type": "tabular", "columns": ["b1", "b2", "b3"],
             "model": {"K": 1, "hidden": 4}},
        ],
    }))
    args = _build_args([
        "--manifest", str(manifest), "--config", str(config),
        "--out-dir", str(tmp_path / "run"), "--batch-size", "2",
        "--max-iter", "1", "--eval-interval", "1", "--align-warmup", "0",
        "--proj-dim", "4", "--proj-hidden", "8", "--devices", "cpu",
        "--augmentation-transform-type", "affine",
        "--augmentation-sd-deformation", "0",
        "--augmentation-sd-bias-field", "0",
        "--augmentation-sd-histogram-warping", "0",
        "--disable-aug-anneal",
        "--preview-interval", "1", "--preview-samples", "1",
        "--save-z", "--save-whitened", "--save-recon",
        "--export-max-samples", "1",
    ])
    trainer = HybridLAMNrTrainer()
    trainer.setup(args)
    summary = (tmp_path / "run" / "run_config.txt").read_text()
    assert "hybrid heterogeneous-view trainer" in summary
    assert "batch local per GPU" in summary
    assert "effective global batch" in summary
    assert "preview samples / columns" in summary
    assert "start / target iteration" in summary
    assert "planned steps this run" in summary
    assert "resolved checkpoint" in summary
    assert "validation mode" in summary
    assert "view[0] name / type" in summary
    assert "view[2] columns" in summary
    loss, align, bpds = trainer._batch_loss(next(iter(trainer.train_loader)), 1)
    assert torch.isfinite(loss)
    assert torch.isfinite(align)
    assert len(bpds) == 3
    loss.backward()
    trainer.opt.zero_grad(set_to_none=True)
    trainer.train()
    assert (tmp_path / "run" / "training_state.pt").exists()
    assert (tmp_path / "run" / "previews" / "im2_recon_it000001.png").exists()
    assert (tmp_path / "run" / "export" / "tau_latents.csv").exists()
    assert (tmp_path / "run" / "export" / "tau_whitened.csv").exists()
    assert (tmp_path / "run" / "export" / "tau_reconstructions.csv").exists()
    assert (tmp_path / "run" / "export" / "im3_reconstructions.csv").exists()
    assert trainer._load_checkpoint(
        tmp_path / "run" / "training_state.pt"
    ) == 2
    assert trainer.ema_models is not None


# ---------------------------------------------------------------------------
# signal1d views
# ---------------------------------------------------------------------------

def _signal_spec(**overrides):
    raw = {"name": "kp", "type": "signal1d", "path_column": "kp", "shape": [3, 16]}
    raw.update(overrides)
    return HybridViewSpec.from_dict(raw)


def test_signal_view_spec_validation():
    spec = _signal_spec()
    assert (spec.kind, spec.channels, spec.shape) == ("signal1d", 3, (16,))
    assert (spec.layout, spec.resample) == ("auto", "linear")
    assert _signal_spec(layout="lc").layout == "LC"
    with pytest.raises(ValueError, match="layout"):
        _signal_spec(layout="CHW")
    with pytest.raises(ValueError, match="resample"):
        _signal_spec(resample="nearest")
    with pytest.raises(ValueError, match="normalization"):
        _signal_spec(normalization="robust")
    with pytest.raises(ValueError, match="path_column"):
        HybridViewSpec.from_dict({"name": "kp", "type": "signal1d", "shape": [3, 16]})


def test_load_signal_layouts_formats_and_resampling(tmp_path):
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(3, 16)).astype("float32")  # (C, L)

    np.save(tmp_path / "cl.npy", signal)
    np.save(tmp_path / "lc.npy", signal.T)
    torch.save(torch.from_numpy(signal.T.copy()), tmp_path / "lc.pt")
    pd.DataFrame(signal.T, columns=["a", "b", "c"]).to_csv(tmp_path / "header.csv", index=False)
    pd.DataFrame(signal.T).to_csv(tmp_path / "noheader.csv", index=False, header=False)
    spec = _signal_spec()
    for name in ["cl.npy", "lc.npy", "lc.pt", "header.csv", "noheader.csv"]:
        loaded = _load_signal(str(tmp_path / name), spec)
        assert loaded.shape == (3, 16), name
        np.testing.assert_allclose(loaded.numpy(), signal, rtol=1e-5, atol=1e-5, err_msg=name)

    # Square arrays are ambiguous unless the layout is given.
    square = rng.normal(size=(3, 3)).astype("float32")
    np.save(tmp_path / "square.npy", square)
    square_spec = _signal_spec(shape=[3, 3])
    with pytest.raises(ValueError, match="ambiguous"):
        _load_signal(str(tmp_path / "square.npy"), square_spec)
    loaded = _load_signal(str(tmp_path / "square.npy"), _signal_spec(shape=[3, 3], layout="LC"))
    np.testing.assert_allclose(loaded.numpy(), square.T)

    # Resampling one period of a sinusoid from 40 to 16 samples.
    t40 = np.arange(40) / 40
    wave = np.stack([np.sin(2 * np.pi * (t40 + k / 3)) for k in range(3)]).astype("float32")
    np.save(tmp_path / "wave.npy", wave)
    t16 = np.arange(16) / 16
    expected = np.stack([np.sin(2 * np.pi * (t16 + k / 3)) for k in range(3)])
    periodic = _load_signal(str(tmp_path / "wave.npy"), _signal_spec(resample="periodic"))
    np.testing.assert_allclose(periodic.numpy(), expected, atol=1e-3)
    for mode in ["linear", "cubic"]:
        out = _load_signal(str(tmp_path / "wave.npy"), _signal_spec(resample=mode))
        assert out.shape == (3, 16) and torch.isfinite(out).all()


def test_signal_normalizer_roundtrip_and_state():
    rng = np.random.default_rng(1)
    signals = [
        torch.from_numpy((300 + 80 * rng.normal(size=(3, 16))).astype("float32"))
        for _ in range(20)
    ]
    for mode in ["0mean", "01", "none"]:
        normalizer = SignalNormalizer(mode).fit(signals)
        stacked = torch.stack([normalizer.transform(s) for s in signals])
        if mode == "0mean":
            assert stacked.mean().abs() < 1e-4
            assert (stacked.transpose(0, 1).reshape(3, -1).std(dim=1, unbiased=False) - 1).abs().max() < 1e-4
        elif mode == "01":
            assert stacked.min() >= -1e-6 and stacked.max() <= 1 + 1e-6
        restored = normalizer.inverse_transform(normalizer.transform(signals[0]))
        torch.testing.assert_close(restored, signals[0], rtol=1e-5, atol=1e-3)
        clone = SignalNormalizer()
        clone.load_state_dict(json.loads(json.dumps(normalizer.state_dict())))
        torch.testing.assert_close(clone.transform(signals[1]), normalizer.transform(signals[1]))


def test_hybrid_trainer_signal1d_smoke(tmp_path):
    rng = np.random.default_rng(5)
    rows = []
    t = np.arange(20) / 20
    for index in range(6):
        path = tmp_path / f"kp_{index}.npy"
        phase = rng.uniform(0, 1, size=(4, 1))
        # Pixel-scale keypoints stored as (L, C) with a different length.
        np.save(path, (300 + 50 * np.sin(2 * np.pi * (t + phase))).T.astype("float32"))
        rows.append({"subject": f"s{index}", "kp": str(path),
                     "b1": 1.0 + index, "b2": 0.5 * index ** 2})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = tmp_path / "views.json"
    config.write_text(json.dumps({
        "subject_column": "subject",
        "views": [
            {"name": "kp", "type": "signal1d", "path_column": "kp",
             "shape": [4, 16], "resample": "periodic",
             "augmentation": {"max_roll": 1, "noise_std": 0.005},
             "model": {"K": 2, "hidden": 8, "use_glow_blocks": True,
                       "padding_mode": "circular", "alignment_pool_size": 8}},
            {"name": "tau", "type": "tabular", "columns": ["b1", "b2"],
             "model": {"K": 1, "hidden": 4}},
        ],
    }))
    args = _build_args([
        "--manifest", str(manifest), "--config", str(config),
        "--out-dir", str(tmp_path / "run"), "--batch-size", "2",
        "--max-iter", "1", "--eval-interval", "1", "--align-warmup", "0",
        "--proj-dim", "4", "--proj-hidden", "8", "--devices", "cpu",
        "--preview-interval", "1", "--preview-samples", "2",
        "--save-recon", "--export-max-samples", "2",
    ])
    trainer = HybridLAMNrTrainer()
    trainer.setup(args)
    assert isinstance(trainer.normalizers["kp"], SignalNormalizer)
    summary = (tmp_path / "run" / "run_config.txt").read_text()
    assert "view[0] signal augmentation" in summary and "max_roll" in summary
    # all-pooled keeps alignment_pool_size temporal bins per channel.
    assert trainer._alignment_pool_size(trainer.views[0]) == 8
    values, masks = next(iter(trainer.train_loader))
    assert tuple(values[0].shape[1:]) == (4, 16)
    assert values[0].abs().max() < 10  # standardized, not pixels
    loss, align, bpds = trainer._batch_loss((values, masks), 1)
    assert torch.isfinite(loss) and torch.isfinite(align) and len(bpds) == 2
    trainer.opt.zero_grad(set_to_none=True)
    trainer.train()  # includes previews with tempered model sampling
    run = tmp_path / "run"
    assert (run / "previews" / "kp_recon_it000001.png").exists()
    assert (run / "previews" / "kp_samples_it000001.png").exists()
    exported = sorted((run / "export" / "kp" / "reconstructions").glob("*.npy"))
    assert exported
    recon = np.load(exported[0])
    assert recon.shape == (4, 16) and 200 < recon.mean() < 400  # original units
    blob = torch.load(run / "training_state.pt", map_location="cpu", weights_only=False)
    assert blob["normalizers"]["kp"]["kind"] == "signal1d"
    assert trainer._load_checkpoint(run / "training_state.pt") == 2


def test_signal_augmentation_validation_and_schedule():
    with pytest.raises(ValueError, match="unknown augmentation keys"):
        _signal_spec(augmentation={"flip": 0.5})
    with pytest.raises(ValueError, match="max_roll"):
        _signal_spec(augmentation={"max_roll": 16})
    with pytest.raises(ValueError, match="roll_prob"):
        _signal_spec(augmentation={"roll_prob": 1.5})
    with pytest.raises(ValueError, match="noise_std"):
        _signal_spec(augmentation={"noise_std": -0.1})
    assert _signal_augmentation_config(_signal_spec(), 0) == {
        "max_roll": 0, "roll_prob": 1.0, "noise_std": 0.0,
    }
    spec = _signal_spec(augmentation={
        "max_roll": 2, "noise_std": 0.005,
        "schedules": "noise_std:linear:0.005->0.0@100,max_roll:linear:4->0@100",
    })
    start = _signal_augmentation_config(spec, 0)
    end = _signal_augmentation_config(spec, 100)
    assert start["noise_std"] == pytest.approx(0.005) and start["max_roll"] == 4
    assert end["noise_std"] == pytest.approx(0.0) and end["max_roll"] == 0


def test_signal_augmentation_roll_range_sharing_and_noise():
    torch.manual_seed(0)
    base = torch.arange(16, dtype=torch.float32).repeat(3, 1)  # (3, 16) ramp
    roll_only = {"max_roll": 2, "roll_prob": 1.0, "noise_std": 0.0}

    def shift_of(tensor):
        # The ramp's first sample (0) moves to index k (mod 16) under roll(k).
        k = int(torch.nonzero(tensor[0] == 0)[0])
        return k if k < 8 else k - 16

    seen = set()
    for _ in range(300):
        a, b = _augment_signal_group([base.clone(), base.clone()], [roll_only, roll_only])
        assert shift_of(a) == shift_of(b)  # shared within a group
        assert torch.equal(torch.roll(base, shift_of(a), dims=-1), a)  # pure circular roll
        seen.add(shift_of(a))
    assert seen == {-2, -1, 0, 1, 2}

    # Separate groups draw independent shifts.
    different = 0
    for _ in range(200):
        (a,) = _augment_signal_group([base.clone()], [roll_only])
        (b,) = _augment_signal_group([base.clone()], [roll_only])
        different += shift_of(a) != shift_of(b)
    assert different > 100

    never = {"max_roll": 2, "roll_prob": 0.0, "noise_std": 0.0}
    (out,) = _augment_signal_group([base.clone()], [never])
    assert torch.equal(out, base)

    noisy = {"max_roll": 0, "roll_prob": 1.0, "noise_std": 0.005}
    zeros = torch.zeros(34, 64)
    (out,) = _augment_signal_group([zeros], [noisy])
    assert out.std().item() == pytest.approx(0.005, rel=0.1)


def test_signal_augmentation_in_dataset_train_only(tmp_path):
    ramp = np.tile(np.arange(16, dtype="float32"), (3, 1))
    rows = []
    for index in range(4):
        path = tmp_path / f"s{index}.npy"
        np.save(path, ramp)
        rows.append({"a": str(path), "b": str(path)})
    frame = pd.DataFrame(rows)
    aug = {"max_roll": 3, "roll_prob": 1.0}
    views = [
        _signal_spec(name="a", path_column="a", augmentation=aug, augmentation_group="clip"),
        _signal_spec(name="b", path_column="b", augmentation=aug, augmentation_group="clip"),
    ]
    none = {v.name: SignalNormalizer("none").fit([torch.from_numpy(ramp)]) for v in views}
    train = HybridManifestDataset(frame, views, none, do_augmentation=True)
    val = HybridManifestDataset(frame, views, none, do_augmentation=False)
    torch.manual_seed(1)
    shifted = 0
    for index in range(len(frame)):
        (a, b), _ = train[index]
        assert torch.equal(a, b)  # same augmentation_group -> same phase shift
        shifted += not torch.equal(a, torch.from_numpy(ramp))
        (va, vb), _ = val[index]
        assert torch.equal(va, torch.from_numpy(ramp))  # no augmentation in validation
    assert shifted > 0


# ---------------------------------------------------------------------------
# image intensity modes (e.g. decoded optical flow)
# ---------------------------------------------------------------------------

def test_image_intensity_spec_validation():
    spec = HybridViewSpec.from_dict(
        {"name": "im", "type": "image2d", "path_column": "im", "shape": [8, 8]}
    )
    assert spec.intensity == "to01"
    flow = HybridViewSpec.from_dict({
        "name": "flow", "type": "image2d", "path_column": "flow",
        "channels": 2, "shape": [8, 8], "intensity": "0MEAN",
    })
    assert flow.intensity == "0mean"
    with pytest.raises(ValueError, match="intensity must be"):
        HybridViewSpec.from_dict({
            "name": "im", "type": "image2d", "path_column": "im",
            "shape": [8, 8], "intensity": "robust",
        })
    with pytest.raises(ValueError, match="only applies to image views"):
        HybridViewSpec.from_dict({
            "name": "tau", "type": "tabular", "columns": ["a", "b"], "intensity": "none",
        })


def test_channel_normalizer_streaming_matches_stacked_statistics():
    rng = np.random.default_rng(3)
    images = [
        torch.from_numpy(rng.normal([[[2.0]], [[-5.0]]], [[[0.5]], [[3.0]]], size=(2, 8, 8)).astype("float32"))
        for _ in range(12)
    ]
    normalizer = ChannelNormalizer("0mean", kind="image2d").fit(iter(images))
    stacked = torch.stack(images).transpose(0, 1).reshape(2, -1).double()
    np.testing.assert_allclose(normalizer._shift, stacked.mean(dim=1).numpy(), rtol=1e-5)
    np.testing.assert_allclose(normalizer._scale, stacked.std(dim=1, unbiased=False).numpy(), rtol=1e-5)
    out = normalizer.transform(images[0])
    assert out.shape == (2, 8, 8)
    torch.testing.assert_close(normalizer.inverse_transform(out), images[0], rtol=1e-5, atol=1e-5)
    assert normalizer.state_dict()["kind"] == "image2d"
    assert SignalNormalizer is ChannelNormalizer


def test_hybrid_trainer_zero_mean_flow_view(tmp_path):
    rng = np.random.default_rng(6)
    rows = []
    for index in range(6):
        path = tmp_path / f"flow_{index}.npy"
        # Signed two-channel field (u, v), channels-first.
        flow = np.stack([
            3.0 + rng.normal(size=(8, 8)),
            -2.0 + 0.5 * rng.normal(size=(8, 8)),
        ]).astype("float32")
        np.save(path, flow)
        rows.append({"subject": f"s{index}", "flow": str(path),
                     "b1": 1.0 + index, "b2": 0.5 * index ** 2})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = tmp_path / "views.json"
    config.write_text(json.dumps({
        "subject_column": "subject",
        "views": [
            {"name": "flow", "type": "image2d", "path_column": "flow",
             "channels": 2, "shape": [8, 8], "intensity": "0mean",
             "augmentation": {"enabled": False},
             "model": {"L": 1, "K": [1], "hidden": [4]}},
            {"name": "tau", "type": "tabular", "columns": ["b1", "b2"],
             "model": {"K": 1, "hidden": 4}},
        ],
    }))
    args = _build_args([
        "--manifest", str(manifest), "--config", str(config),
        "--out-dir", str(tmp_path / "run"), "--batch-size", "2",
        "--max-iter", "1", "--eval-interval", "1", "--align-warmup", "0",
        "--proj-dim", "4", "--proj-hidden", "8", "--devices", "cpu",
        "--preview-interval", "1", "--preview-samples", "1",
        "--save-recon", "--export-max-samples", "2",
    ])
    trainer = HybridLAMNrTrainer()
    trainer.setup(args)
    assert isinstance(trainer.normalizers["flow"], ChannelNormalizer)
    summary = (tmp_path / "run" / "run_config.txt").read_text()
    assert "view[0] intensity" in summary and "0mean" in summary
    values, masks = next(iter(trainer.train_loader))
    x = trainer._prepare(values[0], trainer.views[0])
    assert x.min() < 0  # signed values survive (no per-sample min-max)
    assert abs(float(x.mean())) < 1.0
    loss, align, bpds = trainer._batch_loss((values, masks), 1)
    assert torch.isfinite(loss) and torch.isfinite(align)
    trainer.opt.zero_grad(set_to_none=True)
    trainer.train()
    exported = sorted((tmp_path / "run" / "export" / "flow" / "reconstructions").glob("*.npy"))
    assert exported
    recon = np.load(exported[0])
    assert recon.shape == (2, 8, 8)
    assert 1.5 < recon[0].mean() < 4.5 and -3.5 < recon[1].mean() < -0.5  # original units
    blob = torch.load(tmp_path / "run" / "training_state.pt", map_location="cpu", weights_only=False)
    assert blob["normalizers"]["flow"]["kind"] == "image2d"


# ---------------------------------------------------------------------------
# dequantization of grid-valued image views (e.g. soft one-hot segmentations)
# ---------------------------------------------------------------------------

def test_dequantize_spec_validation():
    spec = HybridViewSpec.from_dict({
        "name": "seg", "type": "image2d", "path_column": "seg", "channels": 6,
        "shape": [8, 8], "intensity": "none", "dequantize": 49,
    })
    assert spec.dequantize == 49
    with pytest.raises(ValueError, match="requires an image view"):
        HybridViewSpec.from_dict({
            "name": "seg", "type": "image2d", "path_column": "seg",
            "shape": [8, 8], "dequantize": 49,  # intensity defaults to to01
        })
    with pytest.raises(ValueError, match="requires an image view"):
        HybridViewSpec.from_dict({
            "name": "tau", "type": "tabular", "columns": ["a", "b"], "dequantize": 10,
        })


def test_dequantize_maps_grid_values_into_their_bins():
    levels = 49
    n = torch.arange(levels + 1, dtype=torch.float32)
    x = (n / levels).repeat(20, 1)
    torch.manual_seed(0)
    stochastic = _dequantize(x, levels, stochastic=True)
    assert torch.all(stochastic >= n / (levels + 1))
    assert torch.all(stochastic < (n + 1) / (levels + 1))
    assert stochastic.std(dim=0).min() > 0  # resampled noise
    fixed = _dequantize(x, levels, stochastic=False)
    torch.testing.assert_close(fixed[0], (n + 0.5) / (levels + 1))
    assert float(fixed.min()) > 0 and float(fixed.max()) < 1


def test_dequantize_is_stochastic_for_training_only(tmp_path):
    grid = np.zeros((6, 8, 8), dtype="float32")
    grid[1, 2:6, 3:5] = 1.0
    grid[2, 0, 0] = 10 / 49
    rows = []
    for index in range(2):
        path = tmp_path / f"seg_{index}.npy"
        np.save(path, grid)
        rows.append({"seg": str(path), "b1": 1.0 + index, "b2": 2.0})
    frame = pd.DataFrame(rows)
    views = [
        HybridViewSpec.from_dict({
            "name": "seg", "type": "image2d", "path_column": "seg", "channels": 6,
            "shape": [8, 8], "intensity": "none", "dequantize": 49,
        }),
        HybridViewSpec.from_dict({"name": "tab", "type": "tabular", "columns": ["b1", "b2"]}),
    ]
    normalizer = TabularNormalizer("0mean").fit(frame[["b1", "b2"]].to_numpy())
    train = HybridManifestDataset(frame, views, {"tab": normalizer},
                                  stochastic_dequantization=True)
    val = HybridManifestDataset(frame, views, {"tab": normalizer})
    a, b = train[0][0][0], train[0][0][0]
    assert not torch.equal(a, b)
    assert float(a.min()) > 0 and float(a.max()) < 1
    torch.testing.assert_close(val[0][0][0], val[0][0][0])
    assert float(val[0][0][0][1, 3, 3]) == pytest.approx(49.5 / 50)
    assert float(val[0][0][0][2, 0, 0]) == pytest.approx(10.5 / 50)

