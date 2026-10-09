"""Shared scalar-NIfTI channel grouping and fixed-split loading for Glow."""
import csv
import glob
import os
import re
from pathlib import Path
from multiprocessing import Value

import ants
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from antstorch.utilities.image_dataset import ImageDataset
from antstorch.utilities import MultiParamScheduler, parse_schedules


class Schedule:
    def __init__(self, specification):
        self.scheduler = MultiParamScheduler(parse_schedules(specification))

    def __call__(self, step):
        return self.scheduler.step(step)


def subject_key(path):
    for part in reversed(path.parts):
        if re.fullmatch(r'CTC\d+-\d+-\d+', part) or re.fullmatch(r'sub-[A-Za-z0-9]+', part):
            return part
    match = re.search(r'sub-[A-Za-z0-9]+', path.name)
    return match.group() if match else path.name.split('_')[0]


def assemble_channel_paths(view_specs, split_manifest=None, allow_missing=False):
    """Join channels by acquisition ID, never by sorted positional zip."""
    pools = []
    for specifications in view_specs:
        mapping = {}
        for pattern in specifications:
            for value in sorted(set(glob.glob(os.path.expanduser(pattern)))):
                path = Path(value)
                key = subject_key(path)
                if key in mapping and mapping[key] != path:
                    raise ValueError(f'Ambiguous channel files for {key}: {mapping[key]}, {path}')
                mapping[key] = path
        pools.append(mapping)
    if not pools:
        raise ValueError('No channel specifications')
    all_keys = set().union(*(set(pool) for pool in pools))
    split = {}
    if split_manifest:
        with open(split_manifest, newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {'subject_id', 'split'}.issubset(rows[0]):
            raise ValueError('Split manifest needs subject_id and split columns')
        if len({row['subject_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate subject_id in split manifest')
        for row in rows:
            if row['split'] not in ('training', 'validation', 'testing'):
                raise ValueError('Invalid split name')
            split[row['subject_id']] = row['split']
        # Enforce available group identities in addition to the acquisition ID.
        group_columns = [('participant_id',), ('group_id',), ('site_id', 'local_subject_id')]
        for columns in group_columns:
            if not set(columns).issubset(rows[0]):
                continue
            assignments = {}
            for row in rows:
                key = tuple(row[column] for column in columns)
                if any(not value for value in key):
                    raise ValueError(f'Missing group identity: {columns}')
                assignments.setdefault(key, set()).add(row['split'])
            if any(len(values) > 1 for values in assignments.values()):
                raise ValueError(f'Split leakage for {columns}')
        # Also check repeat visits when only the minimal subject/split CSV is used.
        visits = {}
        for key, value in split.items():
            group = key.rsplit('-', 1)[0] if re.fullmatch(r'CTC\d+-\d+-\d+', key) else key
            visits.setdefault(group, set()).add(value)
        if any(len(values) > 1 for values in visits.values()):
            raise ValueError('Split leakage between visits')
        unknown = all_keys - set(split)
        if unknown:
            raise ValueError(f'Image IDs missing from manifest: {sorted(unknown)[:5]}')
        keys = {key for key, value in split.items() if value != 'testing'}
    else:
        keys = all_keys
    incomplete = sorted(key for key in keys if any(key not in pool for pool in pools))
    if incomplete and not allow_missing:
        raise ValueError(f'{len(incomplete)} incomplete channel sets; first IDs: {incomplete[:5]}')
    samples = {key: [pool[key] for pool in pools] for key in sorted(keys - set(incomplete))}
    if not samples:
        raise ValueError('No complete channel sets')
    return samples, split, incomplete


def read_channels(paths, size, slice_idx=0):
    images = [ants.image_read(str(path)) for path in paths]
    first = images[0]
    for image in images:
        if image.dimension not in (2, 3) or image.components != 1:
            raise ValueError('Channel grouping requires scalar 2-D or 3-D NIfTI images')
        if image.shape != first.shape or not ants.image_physical_space_consistency(first, image):
            raise ValueError(f'Channel geometry mismatch: {paths}')
        if not np.isfinite(image.numpy()).all():
            raise ValueError(f'Nonfinite channel: {paths}')
    if len(size) == 2 and first.dimension == 3:
        if not 0 <= slice_idx < first.shape[2]:
            raise ValueError('slice_idx outside volume')
        images = [ants.slice_image(image, axis=2, idx=slice_idx, collapse_strategy=1) for image in images]
    if images[0].dimension != len(size):
        raise ValueError('Image and requested spatial dimensions differ')
    first = images[0]
    factor = min(target / actual for target, actual in zip(size, first.shape))
    reference = ants.resample_image(first, tuple(s / factor for s in first.spacing), use_voxels=False, interp_type=1)
    reference = ants.pad_or_crop_image_to_size(reference, size)
    return [reference] + [ants.resample_image_to_target(image, reference, interp_type='linear') for image in images[1:]]


def build_channel_loaders(view_specs, size, train_samples, val_samples, batch, num_workers,
                          val_frac, subject_limit, channels_per_view=1, split_manifest=None,
                          allow_missing=False, slice_idx=0, do_aug=True, aug_schedules=None,
                          disable_aug_anneal=False, seed=0, is_ddp=False, rank=0,
                          world_size=1, intensity='to01'):
    if channels_per_view < 1 or len(view_specs) % channels_per_view:
        raise ValueError('channels_per_view must divide the number of --view specifications')
    samples, split, missing = assemble_channel_paths(view_specs, split_manifest, allow_missing)
    keys = list(samples)
    if subject_limit:
        if split_manifest:
            raise ValueError('Do not use subject_limit with a fixed split manifest')
        keys = keys[:subject_limit]
    if split_manifest:
        train_keys = [key for key in keys if split[key] == 'training']
        val_keys = [key for key in keys if split[key] == 'validation']
        if not train_keys or not val_keys:
            raise ValueError('Fixed split requires both training and held-out validation')
    else:
        # Repeated acquisitions stay together during a random split too.
        groups = sorted({key.rsplit('-', 1)[0] if re.fullmatch(r'CTC\d+-\d+-\d+', key) else key for key in keys})
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        count = min(max(0, round(val_frac * len(groups))), len(groups) - 1)
        held_out = set(groups[:count])
        val_keys = [key for key in keys if (key.rsplit('-', 1)[0] if re.fullmatch(r'CTC\d+-\d+-\d+', key) else key) in held_out]
        train_keys = [key for key in keys if key not in val_keys]
    images_train = [read_channels(samples[key], size, slice_idx) for key in train_keys]
    images_val = [read_channels(samples[key], size, slice_idx) for key in (val_keys or train_keys)]
    raw = intensity != 'to01'
    step = Value('i', 0)
    schedule = Schedule(aug_schedules) if aug_schedules and not disable_aug_anneal else None
    train = ImageDataset(images_train, images_train[0][0], do_data_augmentation=do_aug and not raw,
                         normalize_intensity=not raw, number_of_samples=train_samples,
                         channels_per_view=channels_per_view, aug_scheduler=schedule,
                         data_augmentation_sd_affine=.05, data_augmentation_sd_deformation=10.,
                         data_augmentation_noise_parameters=(0., .05),
                         data_augmentation_sd_simulated_bias_field=1e-8,
                         data_augmentation_sd_histogram_warping=.025)
    train.global_step_ref = step
    validation = ImageDataset(images_val, images_train[0][0], do_data_augmentation=False,
                              normalize_intensity=not raw, number_of_samples=len(images_val),
                              channels_per_view=channels_per_view, sampling='sequential')
    sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if is_ddp else None
    train_loader = DataLoader(train, batch_size=batch, shuffle=sampler is None, sampler=sampler,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(validation, batch_size=min(16, batch), shuffle=False, num_workers=0)
    train.used_subject_ids = train_keys
    validation.used_subject_ids = val_keys or train_keys
    train.excluded_missing = validation.excluded_missing = missing
    if rank == 0:
        print(f'[data] {len(view_specs)//channels_per_view} views, {channels_per_view} channels/view; '
              f'training={len(images_train)}, validation={len(images_val)}, incomplete={len(missing)}')
    return train_loader, val_loader, step


def record_channel_selection(args, train, validation, rank):
    """Keep the actual training pool fixed when resuming an experiment."""
    if not hasattr(train, 'used_subject_ids') or getattr(args, 'check_data', False):
        return
    import hashlib
    import json
    report = dict(channels_per_view=args.channels_per_view, channel_specs=args.view,
                  input_shape=list(args.input_shape),
                  training_ids=train.used_subject_ids, validation_ids=validation.used_subject_ids,
                  excluded_missing=train.excluded_missing, testing_used=False)
    if args.split_manifest:
        report['manifest_sha256'] = hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest()
    output = Path(args.out_dir) / 'channel_data_selection.json'
    if output.exists() and args.auto_resume:
        previous = json.loads(output.read_text())
        if previous != report:
            raise ValueError('Channel data selection changed; use a new --out-dir instead of auto-resuming')
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
