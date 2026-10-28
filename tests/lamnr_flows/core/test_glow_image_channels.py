"""Native NIfTI channel grouping, leakage checks, and CPU training regressions."""
import csv
import numpy as np
import pytest
import torch
import ants
import antstorch
from antstorch.lamnr_flows.core.image_channel_loaders import assemble_channel_paths
from antstorch.lamnr_flows.core.train_lamnr_glow_base import _extract_views_from_batch
from antstorch.lamnr_flows.scripts import train_lamnr_glow_2d as g2
from antstorch.lamnr_flows.scripts import train_lamnr_glow_3d as g3


def cohort(tmp_path):
    rows = []
    fields = np.indices((16, 16, 16))
    vent = (fields[0] + 2 * fields[1] + 3 * fields[2]).astype('float32')
    anatomy = (fields[0] ** 2 + fields[1] + fields[2]).astype('float32')
    for i, split in enumerate(('training', 'validation', 'testing')):
        subject = f'CTC001-{i+1:03d}-01'
        rows.append(dict(subject_id=subject, participant_id=subject[:-3], split=split))
        if split == 'testing':
            continue  # Must not load test images or demand they exist.
        out = tmp_path / subject / 'transformed_to_template'
        out.mkdir(parents=True)
        for name, field in [('vent', vent), ('anatvent', anatomy)]:
            ants.image_write(ants.from_numpy(field), str(out / f'{name}.nii.gz'))
    manifest = tmp_path / 'split.csv'
    write_manifest(manifest, rows)
    views = [[str(tmp_path / '*' / 'transformed_to_template' / f'{name}.nii.gz')]
             for name in ('vent', 'anatvent')]
    return views, manifest, rows, (vent, anatomy)


def write_manifest(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize('dim,module,trainer_class', [(2, g2, g2.LAMNrGlow2DTrainer), (3, g3, g3.LAMNrGlow3DTrainer)])
def test_joint_channels_cpu_train_and_validation(tmp_path, dim, module, trainer_class):
    views, manifest, _, arrays = cohort(tmp_path)
    cli = ['--view', *views[0], '--view', *views[1], '--channels-per-view', '2',
           '--split-manifest', str(manifest), '--H', '16', '--W', '16',
           '--L', '1', '--K', '1', '--hidden', '8', '--batch', '1', '--num-workers', '0',
           '--train-samples', '2', '--max-iter', '1', '--eval-interval', '1',
           '--devices', 'cpu', '--precision', 'float', '--sample-mode', 'off',
           '--disable-aug-anneal', '--out-dir', str(tmp_path / 'run')]
    cli += ['--D', '16', '--spatial-dims', '3'] if dim == 3 else ['--slice-idx', '5']
    args = module._build_args(cli)
    assert args.num_views == 1
    trainer = trainer_class()
    trainer.setup(args)
    assert trainer.input_shape == (2, *(16 for _ in range(dim)))
    assert trainer.train_loader.dataset.used_subject_ids == ['CTC001-001-01']
    assert trainer.val_loader.dataset.used_subject_ids == ['CTC001-002-01']
    assert len(trainer.val_loader.dataset) == 1
    trainer.train_loader.dataset.do_data_augmentation = False
    batch = next(iter(trainer.val_loader))
    extracted = trainer.extract_view(batch, 0, torch.device('cpu'))
    assert extracted.shape == (1, 2, *(16 for _ in range(dim)))
    for channel, values in enumerate(arrays):
        if dim == 2:
            values = values[:, :, 5]
        expected = (values - values.min()) / (values.max() - values.min())
        if dim == 3:
            expected = np.clip(expected, 1e-5, 1 - 1e-5)
        np.testing.assert_allclose(extracted[0, channel].numpy(), expected, atol=1e-5)
    torch.testing.assert_close(batch['views'][0], next(iter(trainer.val_loader))['views'][0])
    trainer.train()
    checkpoint = torch.load(tmp_path / 'run/training_state.pt', map_location='cpu', weights_only=False)
    assert checkpoint['config']['channels'] == 2
    assert all(torch.isfinite(parameter).all() for model in trainer.models for parameter in model.parameters())


def test_missing_channels_and_group_leakage_are_rejected(tmp_path):
    views, manifest, rows, _ = cohort(tmp_path)
    missing = tmp_path / rows[0]['subject_id'] / 'transformed_to_template/anatvent.nii.gz'
    missing.unlink()
    with pytest.raises(ValueError, match='incomplete channel'):
        assemble_channel_paths(views, manifest)
    samples, _, excluded = assemble_channel_paths(views, manifest, allow_missing=True)
    assert excluded == [rows[0]['subject_id']]
    assert list(samples) == [rows[1]['subject_id']]
    rows[1]['participant_id'] = rows[0]['participant_id']
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match='leakage'):
        assemble_channel_paths(views, manifest, allow_missing=True)


def test_channel_pairing_uses_ids_not_positional_zip(tmp_path):
    views, manifest, rows, _ = cohort(tmp_path)
    p = tmp_path / rows[0]['subject_id'] / 'transformed_to_template/vent.nii.gz'
    duplicate = p.with_name('vent_duplicate.nii.gz')
    duplicate.write_bytes(p.read_bytes())
    with pytest.raises(ValueError, match='Ambiguous'):
        assemble_channel_paths([[str(p), str(duplicate)], views[1]], manifest)


@pytest.mark.parametrize('dim', [2, 3])
def test_image_dataset_shared_spatial_augmentation_and_legacy_output(dim):
    array = np.arange(16 ** dim, dtype='float32').reshape((16,) * dim)
    image = ants.from_numpy(array)
    dataset = antstorch.ImageDataset([[image, ants.image_clone(image)]], image,
        channels_per_view=2, number_of_samples=2, sampling='sequential',
        data_augmentation_noise_parameters=(0., 0.), data_augmentation_sd_simulated_bias_field=0.,
        data_augmentation_sd_histogram_warping=0., data_augmentation_sd_deformation=.1)
    batch = dataset[0]['views'][0]
    torch.testing.assert_close(batch[0], batch[1])
    legacy = antstorch.ImageDataset([[image]], image, do_data_augmentation=False)[0]
    assert isinstance(legacy, torch.Tensor)
    assert legacy.shape == (1, *(16 for _ in range(dim)))


def test_multiple_views_of_two_channels(tmp_path):
    views, manifest, _, _ = cohort(tmp_path)
    loader, _, _ = g3.build_loaders_from_globs_3d(
        view_specs=views + views, channels_per_view=2, split_manifest=manifest,
        H=16, W=16, D=16, train_samples=1, val_samples=1, batch=1,
        num_workers=0, val_frac=0, subject_limit=None, do_aug=False)
    parts = _extract_views_from_batch(next(iter(loader)), num_views=2)
    assert len(parts) == 2
    assert all(value.shape == (1, 2, 16, 16, 16) for value in parts)
    torch.testing.assert_close(parts[0], parts[1])


def test_geometry_mismatch_rejected_before_shared_augmentation(tmp_path):
    from antstorch.lamnr_flows.core.image_channel_loaders import read_channels
    views, _, rows, _ = cohort(tmp_path)
    directory = tmp_path / rows[0]['subject_id'] / 'transformed_to_template'
    path = directory / 'anatvent.nii.gz'
    image = ants.image_read(str(path))
    ants.set_origin(image, (4., 0., 0.))
    ants.image_write(image, str(path))
    with pytest.raises(ValueError, match='geometry mismatch'):
        read_channels([directory / 'vent.nii.gz', path], (16, 16, 16))


def test_two_channel_preview_preserves_both_channels():
    from antstorch.lamnr_flows.core.train_lamnr_glow_base import _coerce_nchw_4d
    x = torch.stack([torch.zeros(8, 8), torch.ones(8, 8)]).unsqueeze(0)
    preview = _coerce_nchw_4d(x)
    assert preview.shape == (2, 1, 8, 8)
    torch.testing.assert_close(preview[:, 0], x[0])


def test_auto_resume_rejects_changed_loaded_subject_pool(tmp_path):
    from types import SimpleNamespace
    from antstorch.lamnr_flows.core.image_channel_loaders import record_channel_selection
    args = SimpleNamespace(channels_per_view=2, view=[['vent'], ['anat']], input_shape=(2,16,16),
                           split_manifest=None, out_dir=str(tmp_path), auto_resume=True)
    train = SimpleNamespace(used_subject_ids=['one'], excluded_missing=[])
    val = SimpleNamespace(used_subject_ids=['two'])
    record_channel_selection(args, train, val, 0)
    record_channel_selection(args, train, val, 0)
    train.used_subject_ids.append('new')
    with pytest.raises(ValueError, match='selection changed'):
        record_channel_selection(args, train, val, 0)
