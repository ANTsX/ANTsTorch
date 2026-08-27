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
    HybridViewSpec,
    _augment_image_group,
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
