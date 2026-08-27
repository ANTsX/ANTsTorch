"""Hybrid LAMNr trainer for tabular, 2D-image, and 3D-image views.

The trainer consumes one row-aligned CSV manifest plus a JSON view
configuration. Each view owns its normalizing-flow architecture and its
alignment projector. Missing views are supported: likelihood terms use only
observed rows and alignment losses use each view pair's observed intersection.

Example configuration::

    {
      "subject_column": "SubjectID",
      "views": [
        {"name": "T1", "type": "image3d", "path_column": "T1File",
         "shape": [40, 40, 64]},
        {"name": "tau", "type": "tabular",
         "columns": ["Braak1_2", "Braak3_4", "Braak5_6"]}
      ]
    }

Run with::

    python -m antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid \
        --manifest manifest.csv --config views.json --align vicreg \
        --alignment-latents all-pooled --out-dir runs_hybrid
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import json
import math
import os
import platform
import shutil
from dataclasses import dataclass, field
from multiprocessing import Value
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import antstorch

from antstorch.lamnr_flows.architectures.create_normalizing_flow_model import (
    create_glow_normalizing_flow_model_2d,
    create_glow_normalizing_flow_model_3d,
    create_real_nvp_normalizing_flow_model,
)
from antstorch.lamnr_flows.core.train_lamnr_glow_base import (
    make_warmup,
    n_params,
    set_deterministic,
    to01,
)
from antstorch.lamnr_flows.misc.latent_alignment import (
    LatentAlignmentLossManager,
    Projector,
    flatten_latents,
)
from antstorch.lamnr_flows.scripts.train_lamnr_flows_tabular import (
    TabularNormalizer,
    _build_base_distribution,
    _extract_whitened,
    _inverse_with_guard,
)


VIEW_TYPES = {"tabular", "image2d", "image3d"}


class HybridViewStep(nn.Module):
    """DDP-safe forward wrapper for one heterogeneous flow view."""

    def __init__(
        self,
        flow: nn.Module,
        kind: str,
        alignment_latents: str,
        alignment_pool_size: int,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.kind = kind
        self.alignment_latents = alignment_latents
        self.alignment_pool_size = alignment_pool_size

    def forward(self, x: torch.Tensor):
        logp = self.flow.log_prob(x)
        if self.kind == "tabular":
            z, _ = _inverse_with_guard(self.flow, x)
            flat = z.flatten(1)
        else:
            z, _ = self.flow.inverse_and_log_det(x)
            flat = flatten_latents(
                z,
                strategy=self.alignment_latents,
                target_pool_size=self.alignment_pool_size,
            )
        return logp, flat


@dataclass
class HybridViewSpec:
    """Validated description of one heterogeneous observation view."""

    name: str
    kind: str
    path_column: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    shape: Tuple[int, ...] = field(default_factory=tuple)
    channels: int = 1
    normalization: str = "0mean"
    slice_axis: int = 2
    slice_index: Optional[int] = None
    augmentation_group: Optional[str] = None
    augmentation: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    glob: List[str] = field(default_factory=list)
    key_regex: Optional[str] = None
    csv: Optional[str] = None
    key_column: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "HybridViewSpec":
        kind = str(raw.get("type", raw.get("kind", ""))).lower()
        if kind not in VIEW_TYPES:
            raise ValueError(f"View type must be one of {sorted(VIEW_TYPES)}; got {kind!r}.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("Every hybrid view requires a non-empty 'name'.")

        spec = cls(
            name=name,
            kind=kind,
            path_column=raw.get("path_column"),
            columns=list(raw.get("columns", [])),
            shape=tuple(int(x) for x in raw.get("shape", [])),
            channels=int(raw.get("channels", 1)),
            normalization=str(raw.get("normalization", "0mean")),
            slice_axis=int(raw.get("slice_axis", 2)),
            slice_index=(
                None if raw.get("slice_index") is None else int(raw["slice_index"])
            ),
            augmentation_group=raw.get("augmentation_group"),
            augmentation=dict(raw.get("augmentation", {})),
            model=dict(raw.get("model", {})),
            glob=(
                [str(raw["glob"])] if isinstance(raw.get("glob"), str)
                else [str(x) for x in raw.get("glob", [])]
            ),
            key_regex=raw.get("key_regex"),
            csv=raw.get("csv"),
            key_column=raw.get("key_column"),
        )
        if kind == "tabular":
            if not spec.columns:
                raise ValueError(f"Tabular view {name!r} requires a non-empty 'columns' list.")
            if len(spec.columns) < 2:
                raise ValueError(
                    f"Tabular RealNVP view {name!r} needs at least two columns; "
                    "a one-dimensional affine-coupling flow is degenerate."
                )
        else:
            expected = 2 if kind == "image2d" else 3
            if not spec.path_column:
                raise ValueError(f"Image view {name!r} requires 'path_column'.")
            if len(spec.shape) != expected or any(x <= 0 for x in spec.shape):
                raise ValueError(
                    f"{kind} view {name!r} requires a positive shape of length {expected}."
                )
            if spec.channels <= 0:
                raise ValueError("channels must be positive.")
        return spec


def load_hybrid_config(path: str | Path) -> Tuple[List[HybridViewSpec], Dict[str, Any]]:
    with open(path) as stream:
        config = json.load(stream)
    views = [HybridViewSpec.from_dict(v) for v in config.get("views", [])]
    if len(views) < 2:
        raise ValueError("Hybrid LAMNr training requires at least two configured views.")
    names = [v.name for v in views]
    if len(set(names)) != len(names):
        raise ValueError(f"View names must be unique; got {names}.")
    return views, config


def build_manifest_from_config(
    views: Sequence[HybridViewSpec], config: Dict[str, Any]
) -> pd.DataFrame:
    """Build an outer-joined manifest from configured image globs/CSVs.

    Image keys are either the named/first group of ``key_regex`` or the file
    stem. Tabular sources require ``key_column``. This deliberately keeps
    unmatched keys: the resulting empty cells become ordinary missing views.
    """
    import re

    join_key = str(config.get("join_key", config.get("subject_column", "sample_id")))
    tables: List[pd.DataFrame] = []
    for view in views:
        if view.kind == "tabular":
            if not view.csv or not view.key_column:
                raise ValueError(
                    f"Tabular view {view.name!r} needs 'csv' and 'key_column' "
                    "when --manifest is omitted."
                )
            table = pd.read_csv(Path(view.csv).expanduser())
            needed = [view.key_column, *view.columns]
            missing = sorted(set(needed) - set(table.columns))
            if missing:
                raise ValueError(f"Source for {view.name!r} lacks columns {missing}.")
            table = table[needed].rename(columns={view.key_column: join_key})
        else:
            if not view.glob:
                raise ValueError(
                    f"Image view {view.name!r} needs 'glob' when --manifest is omitted."
                )
            paths = sorted({p for pattern in view.glob for p in glob.glob(
                str(Path(pattern).expanduser()), recursive=True
            )})
            if not paths:
                raise ValueError(f"No files matched globs for view {view.name!r}.")
            keys = []
            for path in paths:
                if view.key_regex:
                    match = re.search(view.key_regex, path)
                    if match is None:
                        raise ValueError(f"{path!r} does not match key_regex for {view.name!r}.")
                    keys.append(match.groupdict().get("key") or match.group(1))
                else:
                    name = Path(path).name
                    keys.append(name[:-7] if name.endswith(".nii.gz") else Path(path).stem)
            table = pd.DataFrame({join_key: keys, view.path_column: paths})
        if table[join_key].duplicated().any():
            raise ValueError(f"Duplicate join keys in source for view {view.name!r}.")
        tables.append(table)
    result = tables[0]
    for table in tables[1:]:
        result = result.merge(table, on=join_key, how="outer", validate="one_to_one")
    return result.sort_values(join_key).reset_index(drop=True)


def _save_hybrid_metric_plots(csv_path: Path, out_dir: Path) -> None:
    """Write global and per-view training/validation metric figures."""
    if not csv_path.exists():
        return
    try:
        import matplotlib.pyplot as plt

        frame = pd.read_csv(csv_path)
        if len(frame) < 2:
            return
        plot_groups = [
            ([c for c in ("loss", "align", "val_loss") if c in frame],
             "objective", "objectives.png"),
            ([c for c in frame if c.startswith("bpd_")],
             "bits per dimension", "bpd_by_view.png"),
            ([c for c in frame if c.startswith("val_bpd_")],
             "validation bits per dimension", "val_bpd_by_view.png"),
        ]
        for columns, ylabel, filename in plot_groups:
            if not columns:
                continue
            figure, axis = plt.subplots()
            for column in columns:
                axis.plot(frame["iter"], frame[column], label=column)
            axis.set_xlabel("iteration")
            axis.set_ylabel(ylabel)
            axis.legend()
            figure.tight_layout()
            figure.savefig(out_dir / filename)
            plt.close(figure)
    except Exception as error:
        tqdm.write(f"[metrics] plot generation skipped: {error}")


def _is_present(value: Any) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def _view_availability(frame: pd.DataFrame, view: HybridViewSpec) -> pd.Series:
    if view.kind == "tabular":
        numeric = frame[view.columns].apply(pd.to_numeric, errors="coerce")
        return pd.Series(np.isfinite(numeric.to_numpy()).all(axis=1), index=frame.index)
    return frame[view.path_column].map(
        lambda value: _is_present(value) and Path(str(value)).expanduser().exists()
    )


def _as_channel_first(array: np.ndarray, channels: int, spatial_dims: int) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if t.ndim == spatial_dims:
        t = t.unsqueeze(0)
    elif t.ndim == spatial_dims + 1 and t.shape[0] != channels and t.shape[-1] == channels:
        order = [t.ndim - 1] + list(range(t.ndim - 1))
        t = t.permute(*order)
    if t.ndim != spatial_dims + 1:
        raise ValueError(
            f"Expected {spatial_dims}D data with an optional channel axis; got {tuple(t.shape)}."
        )
    if t.shape[0] != channels:
        if channels == 1:
            t = t[:1]
        else:
            raise ValueError(f"Expected {channels} channels; got {t.shape[0]}.")
    return t.contiguous()


def _load_image(path: str, spec: HybridViewSpec) -> torch.Tensor:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)

    lower = p.name.lower()
    if lower.endswith(".npy"):
        array = np.load(p)
    elif lower.endswith(".pt") or lower.endswith(".pth"):
        obj = torch.load(p, map_location="cpu", weights_only=False)
        array = obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
    elif spec.kind == "image2d" and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        from PIL import Image

        array = np.asarray(Image.open(p).convert("L" if spec.channels == 1 else "RGB"))
    else:
        import ants

        array = ants.image_read(str(p)).numpy()

    if spec.kind == "image2d" and np.asarray(array).ndim == 3 and spec.channels == 1:
        axis = spec.slice_axis % 3
        index = spec.slice_index
        if index is None:
            index = int(np.asarray(array).shape[axis] // 2)
        array = np.take(array, index, axis=axis)

    spatial_dims = 2 if spec.kind == "image2d" else 3
    tensor = _as_channel_first(array, spec.channels, spatial_dims)
    target = tuple(spec.shape)
    if tuple(tensor.shape[1:]) != target:
        mode = "bilinear" if spatial_dims == 2 else "trilinear"
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=target, mode=mode, align_corners=False
        ).squeeze(0)
    return tensor


def _default_augmentation_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate the specialized-trainer augmentation CLI into one config."""
    if args.augmentation_noise_model == "additivegaussian":
        noise_parameters: Any = [0.0, float(args.image_noise_std)]
    elif args.augmentation_noise_model == "saltandpepper":
        noise_parameters = [float(args.image_noise_std), 1.0, 0.0]
    else:
        noise_parameters = float(args.image_noise_std)
    return {
        "enabled": not bool(args.disable_augmentation),
        "transform_type": args.augmentation_transform_type,
        "sd_affine": float(args.augmentation_sd_affine),
        "sd_deformation": float(args.augmentation_sd_deformation),
        "noise_model": args.augmentation_noise_model,
        "noise_parameters": noise_parameters,
        "sd_simulated_bias_field": float(args.augmentation_sd_bias_field),
        "sd_histogram_warping": float(args.augmentation_sd_histogram_warping),
        "horizontal_flip_probability": float(args.horizontal_flip_probability),
        "schedules": "" if args.disable_aug_anneal else args.aug_schedules,
    }


def _merge_augmentation_config(
    base: Dict[str, Any], override: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def _scheduled_augmentation_config(config: Dict[str, Any], step: int) -> Dict[str, Any]:
    current = dict(config)
    schedule_spec = str(current.get("schedules", "") or "").strip()
    if schedule_spec:
        scheduler = antstorch.MultiParamScheduler(antstorch.parse_schedules(schedule_spec))
        values = scheduler.step(int(step))
        for key in (
            "sd_affine",
            "sd_deformation",
            "sd_simulated_bias_field",
            "sd_histogram_warping",
            "horizontal_flip_probability",
            "tabular_noise_std",
        ):
            if key in values:
                current[key] = float(values[key])
        if "noise_std" in values:
            scheduled_noise = max(0.0, float(values["noise_std"]))
            if current.get("noise_model", "additivegaussian") == "additivegaussian":
                params = list(current.get("noise_parameters", [0.0, 0.0]))
                if len(params) < 2:
                    params = [0.0, float(params[0]) if params else 0.0]
                params[1] = scheduled_noise
                current["noise_parameters"] = params
            else:
                current["noise_parameters"] = scheduled_noise
    return current


def _augment_image_group(
    tensors: List[torch.Tensor],
    config: Dict[str, Any],
) -> List[torch.Tensor]:
    """Apply one shared spatial transform to a group of image views.

    Channels and views are passed as ANTs modalities in one simulation, which
    guarantees a common affine/deformation while allowing the ANTs intensity
    augmentations to operate on every modality.
    """
    if not tensors or not bool(config.get("enabled", True)):
        return tensors

    ndim = tensors[0].ndim - 1
    spatial_shape = tuple(tensors[0].shape[1:])
    if any(t.ndim - 1 != ndim or tuple(t.shape[1:]) != spatial_shape for t in tensors):
        raise ValueError(
            "All present image views in an augmentation_group must have the "
            "same dimensionality and spatial shape."
        )

    # Raster-style flip from the specialized 2D trainer. The draw is shared
    # across the entire group, preserving correspondence.
    flip_probability = float(config.get("horizontal_flip_probability", 0.0))
    if flip_probability > 0.0 and torch.rand(()).item() < flip_probability:
        tensors = [torch.flip(t, dims=(-1,)) for t in tensors]

    noise_parameters = config.get("noise_parameters", [0.0, 0.0])
    noise_values = (
        [float(noise_parameters)]
        if isinstance(noise_parameters, (int, float))
        else [float(x) for x in noise_parameters]
    )
    active = any(
        abs(float(config.get(key, 0.0))) > 0.0
        for key in (
            "sd_affine",
            "sd_deformation",
            "sd_simulated_bias_field",
            "sd_histogram_warping",
        )
    ) or any(abs(x) > 0.0 for x in noise_values)
    if not active:
        return tensors

    import ants

    modalities = []
    channel_counts = []
    for tensor in tensors:
        channel_counts.append(int(tensor.shape[0]))
        modalities.extend(
            ants.from_numpy(channel.detach().cpu().numpy().astype(np.float32))
            for channel in tensor
        )
    result, _ = antstorch.augment_image_modalities(
        modalities,
        modalities[0],
        transform_type=str(config.get("transform_type", "affineAndDeformation")),
        noise_model=config.get("noise_model", "additivegaussian"),
        noise_parameters=(
            float(noise_parameters)
            if isinstance(noise_parameters, (int, float))
            else tuple(noise_parameters)
        ),
        sd_simulated_bias_field=float(config.get("sd_simulated_bias_field", 0.0)),
        sd_histogram_warping=float(config.get("sd_histogram_warping", 0.0)),
        sd_affine=float(config.get("sd_affine", 0.0)),
        sd_deformation=float(config.get("sd_deformation", 0.0)),
    )

    augmented = []
    offset = 0
    for channels in channel_counts:
        arrays = [result[offset + c].numpy() for c in range(channels)]
        augmented.append(torch.from_numpy(np.stack(arrays).astype(np.float32)))
        offset += channels
    return augmented


class HybridManifestDataset(Dataset):
    """Lazy row-aligned heterogeneous dataset with explicit availability masks."""

    def __init__(
        self,
        frame: pd.DataFrame,
        views: Sequence[HybridViewSpec],
        normalizers: Dict[str, TabularNormalizer],
        image_noise_std: float = 0.0,
        tabular_noise_std: float = 0.0,
        augmentation_groups: Optional[Dict[str, Dict[str, Any]]] = None,
        do_augmentation: bool = False,
        number_of_samples: Optional[int] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.views = list(views)
        self.normalizers = normalizers
        self.image_noise_std = float(image_noise_std)
        self.tabular_noise_std = float(tabular_noise_std)
        self.augmentation_groups = augmentation_groups or {}
        self.do_augmentation = bool(do_augmentation)
        self.number_of_samples = (
            len(self.frame) if number_of_samples is None else int(number_of_samples)
        )
        if self.number_of_samples <= 0:
            raise ValueError("number_of_samples must be positive.")
        self.global_step_ref = None

    def _global_step(self) -> int:
        return int(self.global_step_ref.value) if self.global_step_ref is not None else 0

    def __len__(self) -> int:
        return self.number_of_samples

    def __getitem__(self, index: int):
        index = int(index) % len(self.frame)
        row = self.frame.iloc[index]
        values: List[torch.Tensor] = []
        masks: List[bool] = []
        step = self._global_step()
        for spec in self.views:
            if spec.kind == "tabular":
                raw = pd.to_numeric(row[spec.columns], errors="coerce").to_numpy(dtype=np.float64)
                present = bool(np.isfinite(raw).all())
                if present:
                    noise_std = 0.0
                    if self.do_augmentation:
                        tab_config = _scheduled_augmentation_config(
                            _merge_augmentation_config(
                                {"tabular_noise_std": self.tabular_noise_std},
                                spec.augmentation,
                            ),
                            step,
                        )
                        noise_std = float(tab_config.get("tabular_noise_std", 0.0))
                    value = self.normalizers[spec.name].transform(
                        raw[None, :], noise_std=noise_std
                    ).squeeze(0)
                else:
                    value = torch.zeros(len(spec.columns), dtype=torch.float32)
            else:
                path_value = row[spec.path_column]
                present = _is_present(path_value) and Path(str(path_value)).expanduser().exists()
                if present:
                    value = _load_image(str(path_value), spec)
                else:
                    value = torch.zeros((spec.channels, *spec.shape), dtype=torch.float32)
            values.append(value.float())
            masks.append(present)

        if self.do_augmentation:
            grouped_indices: Dict[str, List[int]] = {}
            for vi, spec in enumerate(self.views):
                if spec.kind == "tabular" or not masks[vi]:
                    continue
                group = spec.augmentation_group or f"__view__{spec.name}"
                grouped_indices.setdefault(group, []).append(vi)
            for group, indices in grouped_indices.items():
                config = _scheduled_augmentation_config(
                    self.augmentation_groups.get(group, {}), step
                )
                augmented = _augment_image_group([values[i] for i in indices], config)
                for vi, tensor in zip(indices, augmented):
                    values[vi] = tensor
        return values, torch.tensor(masks, dtype=torch.bool)


def hybrid_collate(samples):
    n_views = len(samples[0][0])
    values = [torch.stack([sample[0][vi] for sample in samples]) for vi in range(n_views)]
    masks_matrix = torch.stack([sample[1] for sample in samples])
    masks = [masks_matrix[:, vi] for vi in range(n_views)]
    return values, masks


def _split_manifest(
    frame: pd.DataFrame,
    val_fraction: float,
    seed: int,
    subject_column: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if subject_column:
        if subject_column not in frame:
            raise ValueError(f"subject_column {subject_column!r} is absent from the manifest.")
        units = frame[subject_column].drop_duplicates().to_numpy()
        rng.shuffle(units)
        n_val = min(max(0, int(round(val_fraction * len(units)))), max(len(units) - 1, 0))
        val_units = set(units[:n_val].tolist())
        is_val = frame[subject_column].isin(val_units)
    else:
        indices = np.arange(len(frame))
        rng.shuffle(indices)
        n_val = min(max(0, int(round(val_fraction * len(frame)))), max(len(frame) - 1, 0))
        val_rows = set(indices[:n_val].tolist())
        is_val = pd.Series([i in val_rows for i in range(len(frame))], index=frame.index)
    train = frame.loc[~is_val].reset_index(drop=True)
    val = frame.loc[is_val].reset_index(drop=True)
    return train, val if len(val) else train.copy()


class HybridLAMNrTrainer:
    """One training loop for heterogeneous LAMNr observation spaces."""

    def setup(self, args: argparse.Namespace) -> None:
        self.args = args
        set_deterministic(args.seed)
        # Optional: torch.autograd.set_detect_anomaly(True) pinpoints the
        # exact forward op behind a NaN/Inf gradient at the cost of a large
        # slowdown. Off by default; mirrors BaseLAMNrTrainer's --detect-anomaly.
        if bool(getattr(args, "detect_anomaly", False)):
            torch.autograd.set_detect_anomaly(True)
            tqdm.write(
                "[debug] torch.autograd.set_detect_anomaly(True) enabled — "
                "expect a large slowdown"
            )
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_ddp = self.world_size > 1
        if self.is_ddp:
            if not torch.cuda.is_available():
                raise RuntimeError("Hybrid DDP currently requires CUDA/NCCL.")
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
            self.dev = torch.device(f"cuda:{self.local_rank}")
        elif args.devices == "mps" and torch.backends.mps.is_available():
            self.dev = torch.device("mps")
        elif args.devices.startswith("cuda") and torch.cuda.is_available():
            self.dev = torch.device(args.devices.split(",")[0])
        else:
            self.dev = torch.device("cpu")

        self.run_dir = Path(args.out_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "training_state.pt"
        self.metrics_path = self.run_dir / "metrics.csv"

        self.views, config = load_hybrid_config(args.config)

        # Resolve the resume checkpoint (if any) *before* the manifest and
        # models are built, so --use-ckpt-config can steer view/architecture
        # construction instead of only being reconciled after models already
        # exist (which risks a confusing state_dict shape-mismatch error, or
        # worse, a silent load onto the wrong architecture).
        self._resume_path: Optional[Path] = None
        if args.resume:
            self._resume_path = Path(args.resume)
        elif args.auto_resume and self.state_path.exists():
            self._resume_path = self.state_path
        if self._resume_path is not None and self._resume_path.exists():
            self._reconcile_config_with_checkpoint(self._resume_path)

        frame = (
            pd.read_csv(args.manifest)
            if args.manifest
            else build_manifest_from_config(self.views, config)
        )
        self.full_frame = frame.reset_index(drop=True)
        required = []
        for view in self.views:
            required.extend(view.columns if view.kind == "tabular" else [view.path_column])
        missing_columns = sorted(set(required) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {missing_columns}")

        subject_column = config.get("subject_column", args.subject_column or None)
        if args.subject_limit > 0:
            if subject_column and subject_column in frame.columns:
                keep_ids = frame[subject_column].drop_duplicates().iloc[: args.subject_limit]
                frame = frame[frame[subject_column].isin(keep_ids)].reset_index(drop=True)
            else:
                frame = frame.iloc[: args.subject_limit].reset_index(drop=True)
            self.full_frame = frame.reset_index(drop=True)
        train_frame, val_frame = _split_manifest(
            frame, args.val_fraction, args.seed, subject_column
        )
        self.normalizers: Dict[str, TabularNormalizer] = {}
        for view in self.views:
            available = _view_availability(train_frame, view)
            if not available.any():
                raise ValueError(f"View {view.name!r} has no available training observations.")
            if view.kind != "tabular":
                continue
            matrix = train_frame[view.columns].apply(pd.to_numeric, errors="coerce").to_numpy()
            complete = matrix[np.isfinite(matrix).all(axis=1)]
            if len(complete) == 0:
                raise ValueError(f"Tabular view {view.name!r} has no complete training rows.")
            normalizer = TabularNormalizer(view.normalization)
            normalizer.fit(complete)
            self.normalizers[view.name] = normalizer

        default_augmentation = _default_augmentation_config(args)
        configured_groups = dict(config.get("augmentation_groups", {}))
        group_members: Dict[str, List[HybridViewSpec]] = {}
        for view in self.views:
            if view.kind != "tabular":
                group = view.augmentation_group or f"__view__{view.name}"
                group_members.setdefault(group, []).append(view)
        self.augmentation_groups: Dict[str, Dict[str, Any]] = {}
        for group, members in group_members.items():
            geometries = {(v.kind, v.shape) for v in members}
            if len(geometries) != 1:
                raise ValueError(
                    f"augmentation_group {group!r} mixes incompatible image "
                    f"geometries: {sorted(geometries)}"
                )
            per_view_overrides = [v.augmentation for v in members if v.augmentation]
            if len(per_view_overrides) > 1 and any(
                override != per_view_overrides[0]
                for override in per_view_overrides[1:]
            ):
                raise ValueError(
                    f"Views in augmentation_group {group!r} specify conflicting "
                    "augmentation dictionaries; configure the group once under "
                    "'augmentation_groups'."
                )
            group_config = _merge_augmentation_config(
                default_augmentation, configured_groups.get(group)
            )
            if per_view_overrides:
                group_config = _merge_augmentation_config(
                    group_config, per_view_overrides[0]
                )
            self.augmentation_groups[group] = group_config

        self.train_dataset = HybridManifestDataset(
            train_frame, self.views, self.normalizers,
            image_noise_std=args.image_noise_std,
            tabular_noise_std=args.tabular_noise_std,
            augmentation_groups=self.augmentation_groups,
            do_augmentation=not args.disable_augmentation,
            number_of_samples=(args.train_samples or None),
        )
        self.val_dataset = HybridManifestDataset(
            val_frame, self.views, self.normalizers,
            augmentation_groups=self.augmentation_groups,
            do_augmentation=False,
            number_of_samples=(args.val_samples or None),
        )
        self.global_step = Value("i", 0)
        self.train_dataset.global_step_ref = self.global_step
        train_sampler = (
            DistributedSampler(
                self.train_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True, seed=args.seed,
            )
            if self.is_ddp else None
        )
        val_sampler = (
            DistributedSampler(
                self.val_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=False,
            )
            if self.is_ddp else None
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=args.num_workers, collate_fn=hybrid_collate,
        )
        self.train_sampler = train_sampler
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=val_sampler, num_workers=args.num_workers,
            collate_fn=hybrid_collate,
        )

        bare_models = nn.ModuleList([self._build_model(v) for v in self.views])
        self.models = nn.ModuleList([
            HybridViewStep(
                model, view.kind, args.alignment_latents,
                args.alignment_pool_size,
            ).to(self.dev)
            for model, view in zip(bare_models, self.views)
        ])
        # Prime data-dependent ActNorm and latent shapes with real observations
        # before DDP wraps/broadcasts rank-0 parameters and buffers.
        probe_batch = next(iter(self.train_loader))
        feature_dims = [
            self._probe_feature_dim(i, probe_batch) for i in range(len(self.views))
        ]
        self.projectors = None
        if args.align != "none":
            self.projectors = nn.ModuleList([
                Projector(d, args.proj_hidden, args.proj_dim).to(self.dev)
                for d in feature_dims
            ])
        if self.is_ddp:
            self.models = nn.ModuleList([
                DistributedDataParallel(
                    model, device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=args.ddp_find_unused,
                )
                for model in self.models
            ])
            if self.projectors is not None:
                self.projectors = nn.ModuleList([
                    DistributedDataParallel(
                        projector, device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=args.ddp_find_unused,
                    )
                    for projector in self.projectors
                ])
        elif self.dev.type == "cuda" and "," in args.devices:
            device_ids = [int(device.split(":")[-1]) for device in args.devices.split(",")]
            self.models = nn.ModuleList([
                DataParallel(model, device_ids=device_ids) for model in self.models
            ])
            if self.projectors is not None:
                self.projectors = nn.ModuleList([
                    DataParallel(projector, device_ids=device_ids)
                    for projector in self.projectors
                ])
        self.align_mgr = LatentAlignmentLossManager(args, self.projectors, self.dev)

        self.s_nll = self.s_align = None
        parameters = list(self.models.parameters())
        if self.projectors is not None:
            parameters += list(self.projectors.parameters())
        if args.weighting == "kendall" and args.align != "none":
            self.s_nll = nn.Parameter(
                torch.tensor([args.init_logvar_nll], device=self.dev)
            )
            self.s_align = nn.Parameter(
                torch.tensor([args.init_logvar_align], device=self.dev)
            )
            parameters += [self.s_nll, self.s_align]
        self.opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        self.warm = make_warmup(
            self.opt, args.warmup_iters, args.lr_decay_gamma, args.lr_decay_steps
        )
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.plateau_threshold,
            cooldown=args.plateau_cooldown,
            min_lr=args.min_lr,
        )
        self.amp_enabled = args.precision == "mixed" and self.dev.type == "cuda"
        self.amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(
            enabled=(self.amp_enabled and self.amp_dtype == torch.float16)
        )
        self.ema_models: Optional[nn.ModuleList] = None
        self.ema_projectors: Optional[nn.ModuleList] = None
        self.anomaly_streak = 0

        self.start_iter = 1
        if self._resume_path is not None:
            self.start_iter = self._load_checkpoint(self._resume_path)
        if args.extra_iters > 0:
            args.max_iter = (self.start_iter - 1) + args.extra_iters
        if self.rank == 0 and not self.metrics_path.exists():
            columns = ["iter", "loss", "align", "sum_bpd", "val_loss", "lr"]
            columns.extend(f"bpd_{view.name}" for view in self.views)
            columns.extend(f"val_bpd_{view.name}" for view in self.views)
            self.metrics_path.write_text(",".join(columns) + "\n")
        self.last_val_bpds = [float("nan")] * len(self.views)

        availability = {
            view.name: int(_view_availability(train_frame, view).sum())
            for view in self.views
        }
        if self.rank == 0:
            (self.run_dir / "run_config.json").write_text(json.dumps({
                "trainer": "hybrid",
                "world_size": self.world_size,
                "device": str(self.dev),
                "arguments": vars(args),
                "views": [view.__dict__ for view in self.views],
                "availability": availability,
                "projector_input_dimensions": feature_dims,
            }, indent=2, default=str))
            summary = self._format_run_summary(
                args=args,
                train_frame=train_frame,
                val_frame=val_frame,
                subject_column=subject_column,
                feature_dims=feature_dims,
                availability=availability,
            )
            print("\n" + summary)
            (self.run_dir / "run_config.txt").write_text(summary + "\n")

    def _format_run_summary(
        self,
        args: argparse.Namespace,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        subject_column: Optional[str],
        feature_dims: Sequence[int],
        availability: Dict[str, int],
    ) -> str:
        """Return the human-readable startup dump for a hybrid run."""
        from datetime import datetime

        rows = [
            f"[run] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Py {platform.python_version()} | torch {torch.__version__} | "
            f"cuda={str(torch.cuda.is_available()).lower()} "
            f"(n={torch.cuda.device_count()})",
            "[note] hybrid heterogeneous-view trainer",
        ]

        def add(label: str, value: Any) -> None:
            rows.append(f"{label:>28}: {'None' if value is None else value}")

        effective_global_batch = (
            int(args.batch_size) * int(args.accum_steps) * int(self.world_size)
        )
        val_mode = (
            "training subjects, clean/no augmentation"
            if float(args.val_fraction) == 0.0
            else "held-out subjects"
        )

        add("out_dir", args.out_dir)
        add("manifest / config", f"{args.manifest} / {args.config}")
        add("world_size / device", f"{self.world_size} / {self.dev}")
        add("views", len(self.views))
        add("precision / amp_dtype", f"{args.precision} / {args.amp_dtype}")
        add("num_workers / seed", f"{args.num_workers} / {args.seed}")
        add("batch local per GPU", args.batch_size)
        add("grad_accum", args.accum_steps)
        add("effective global batch", effective_global_batch)
        add("max_iter / extra", f"{args.max_iter} / {args.extra_iters}")
        add("eval / preview interval", f"{args.eval_interval} / {args.preview_interval}")
        add("lr / warmup", f"{args.lr} / {args.warmup_iters}")
        add("grad_clip / weight_decay", f"{args.grad_clip} / {args.weight_decay}")
        add("ema / decay", f"{args.ema} / {args.ema_decay}")
        add("lr_decay gamma / steps", f"{args.lr_decay_gamma} / {args.lr_decay_steps}")
        add(
            "plateau fac/pat/thr/cd",
            f"{args.plateau_factor} / {args.plateau_patience} / "
            f"{args.plateau_threshold} / {args.plateau_cooldown}",
        )
        add("min_lr", args.min_lr)
        add("resume", args.resume or None)
        add("auto_resume / ckpt config", f"{args.auto_resume} / {args.use_ckpt_config}")
        add("manifest rows", len(self.full_frame))
        add("subject column", subject_column)
        add("train / val rows", f"{len(train_frame)} / {len(val_frame)}")
        add("validation mode", val_mode)
        add("train / val virtual len", f"{len(self.train_dataset)} / {len(self.val_dataset)}")
        add("train / val samples", f"{args.train_samples} / {args.val_samples}")
        add("augmentation disabled", args.disable_augmentation)
        add("aug anneal disabled", args.disable_aug_anneal)
        add("augmentation groups", sorted(self.augmentation_groups))
        for group_name, group_config in sorted(self.augmentation_groups.items()):
            add(f"aug[{group_name}] schedules", group_config.get("schedules"))

        add("align / weighting", f"{args.align} / {args.weighting}")
        add("align weight / warmup", f"{args.align_weight} / {args.align_warmup}")
        add("alignment latents / pool", f"{args.alignment_latents} / {args.alignment_pool_size}")
        add("proj dim / hidden", f"{args.proj_dim} / {args.proj_hidden}")
        add(
            "vicreg inv/var/cov/gamma",
            f"{args.vicreg_inv} / {args.vicreg_var} / "
            f"{args.vicreg_cov} / {args.vicreg_gamma}",
        )
        add("screen / fraction", f"{args.screen} / {args.screen_frac}")
        add("screen warmup / refresh", f"{args.screen_warmup} / {args.screen_refresh}")
        add("cca ridge / prefilter", f"{args.cca_ridge} / {args.prefilter_frac}")
        add("sample mode / temp", f"{args.sample_mode} / {args.sample_temp}")
        add("grad checkpoint", args.grad_checkpoint)

        rows.append("-" * 72)
        for index, (view, feature_dim, model) in enumerate(
            zip(self.views, feature_dims, self.models)
        ):
            add(f"view[{index}] name / type", f"{view.name} / {view.kind}")
            add(f"view[{index}] observed train", availability[view.name])
            add(f"view[{index}] flow parameters", f"{n_params(self._unwrap(model)):,}")
            add(f"view[{index}] projector input", feature_dim)
            if view.kind == "tabular":
                add(f"view[{index}] columns", view.columns)
                add(f"view[{index}] normalization", view.normalization)
            else:
                add(f"view[{index}] shape / channels", f"{view.shape} / {view.channels}")
                add(f"view[{index}] augmentation group", view.augmentation_group)
            add(f"view[{index}] model", view.model)

        rows.append("-" * 72)
        return "\n".join(rows)

    def _cfg(self, view: HybridViewSpec, name: str, default: Any) -> Any:
        return view.model.get(name, default)

    def _build_model(self, view: HybridViewSpec) -> nn.Module:
        if view.kind == "tabular":
            dim = len(view.columns)
            base_name = self._cfg(view, "base_distribution", self.args.tabular_base)
            q0 = _build_base_distribution(
                D=dim,
                base=base_name,
                pca_latent_dim=min(int(self._cfg(view, "pca_latent_dimension", 4)), dim),
                base_min_log=float(self._cfg(view, "base_min_log", -5.0)),
                base_max_log=float(self._cfg(view, "base_max_log", 5.0)),
                base_sigma=float(self._cfg(view, "base_sigma", 0.1)),
            )
            model = create_real_nvp_normalizing_flow_model(
                latent_size=dim,
                K=int(self._cfg(view, "K", self.args.tabular_K)),
                q0=q0,
                mlp_width=self._cfg(view, "hidden", self.args.tabular_hidden),
                scale_cap=float(self._cfg(view, "scale_cap", self.args.scale_cap)),
                mask_mode=str(self._cfg(view, "mask_mode", "alternating")),
                leaky_relu_negative_slope=float(
                    self._cfg(view, "leaky_relu_negative_slope", 0.0)
                ),
                spectral_norm_scales=bool(self._cfg(view, "spectral_norm_scales", False)),
                additive_first_n=int(self._cfg(view, "additive_first_n", 0)),
                actnorm_every=int(self._cfg(view, "actnorm_every", 1)),
            )
        else:
            input_shape = (view.channels, *view.shape)
            hidden = self._cfg(view, "hidden", self.args.image_hidden)
            hidden = (
                [int(value) for value in hidden]
                if isinstance(hidden, (list, tuple))
                else int(hidden)
            )
            base = str(self._cfg(view, "base", self.args.image_base)).lower()
            base = {"glowbase": "glow", "diaggaussian": "diag"}.get(base, base)
            if base not in {"glow", "diag"}:
                raise ValueError(f"Unsupported image base distribution {base!r}.")
            kwargs = dict(
                input_shape=input_shape,
                L=int(self._cfg(view, "L", self.args.image_L)),
                K=self._cfg(view, "K", self.args.image_K),
                hidden_channels=hidden,
                base=base,
                glowbase_logscale_factor=float(
                    self._cfg(view, "glowbase_logscale_factor", 1.0)
                ),
                glowbase_min_log=float(self._cfg(view, "glowbase_min_log", -5.0)),
                glowbase_max_log=float(self._cfg(view, "glowbase_max_log", 5.0)),
                split_mode="channel", scale=True,
                scale_map=str(self._cfg(view, "scale_map", "sigmoid")),
                leaky=0.0, net_actnorm=bool(self._cfg(view, "net_actnorm", False)),
                scale_cap=float(self._cfg(view, "scale_cap", self.args.scale_cap)),
                actnorm_scale_cap=self._cfg(view, "actnorm_scale_cap", None),
                legacy_conv_cap=self._cfg(view, "legacy_conv_cap", None),
                grad_checkpoint={"auto": None, "on": True, "off": False}[
                    str(self._cfg(view, "grad_checkpoint", self.args.grad_checkpoint))
                ],
            )
            factory = (
                create_glow_normalizing_flow_model_2d
                if view.kind == "image2d"
                else create_glow_normalizing_flow_model_3d
            )
            if view.kind == "image3d":
                kwargs.update(
                    shift_cap=self._cfg(view, "shift_cap", None),
                    gen_clamp=float(self._cfg(view, "gen_clamp", 1.0e4)),
                )
            model = factory(**kwargs)
        model = model.to(device=self.dev, dtype=torch.float32).train()
        if self.rank == 0:
            print(f"[init] {view.name} ({view.kind}): {n_params(model):,} parameters")
        return model

    @staticmethod
    def _unwrap(module: nn.Module) -> nn.Module:
        while isinstance(module, (DistributedDataParallel, DataParallel)):
            module = module.module
        return module

    def _flow(self, module: nn.Module) -> nn.Module:
        module = self._unwrap(module)
        return module.flow if isinstance(module, HybridViewStep) else module

    def _prepare(self, tensor: torch.Tensor, view: HybridViewSpec) -> torch.Tensor:
        tensor = tensor.to(device=self.dev, dtype=torch.float32)
        return tensor if view.kind == "tabular" else to01(tensor)

    def _encode(self, model: nn.Module, x: torch.Tensor, view: HybridViewSpec):
        if view.kind == "tabular":
            z, _ = _inverse_with_guard(model, x)
            return z, z.flatten(1)
        z, _ = model.inverse_and_log_det(x)
        flat = flatten_latents(
            z,
            strategy=self.args.alignment_latents,
            target_pool_size=self.args.alignment_pool_size,
        )
        return z, flat

    @torch.no_grad()
    def _probe_feature_dim(self, index: int, batch=None) -> int:
        view = self.views[index]
        x = None
        if batch is not None:
            values, masks = batch
            present = masks[index].bool()
            if present.any():
                x = self._prepare(values[index][present], view)
        if x is None:
            shape = (
                (2, len(view.columns))
                if view.kind == "tabular"
                else (2, view.channels, *view.shape)
            )
            x = torch.rand(shape, device=self.dev) * 0.9 + 0.05
        elif view.kind == "tabular" and len(x) == 1:
            # Flat ActNorm needs at least two non-identical samples.
            x = torch.cat([x, x + 1e-4 * torch.randn_like(x)], dim=0)
        _, flat = self.models[index](x)
        return int(flat.shape[1])

    def _batch_loss(
        self,
        batch,
        iteration: int,
        models: Optional[Sequence[nn.Module]] = None,
        projectors: Optional[Sequence[nn.Module]] = None,
    ):
        values, masks = batch
        models = self.models if models is None else models
        L_nll = torch.tensor(0.0, device=self.dev)
        latents, device_masks, bpds = [], [], []
        observed_views = 0
        for value, mask, view, model in zip(values, masks, self.views, models):
            x = self._prepare(value, view)
            mask = mask.to(device=self.dev, dtype=torch.bool)
            device_masks.append(mask)
            logp, flat = model(x)
            if mask.any():
                dimensions = float(x[0].numel())
                bpd = (-logp[mask] / (math.log(2.0) * dimensions)).mean()
                L_nll = L_nll + bpd
                observed_views += 1
                bpds.append(float(bpd.detach().cpu()))
            else:
                bpds.append(float("nan"))
            latents.append(torch.nan_to_num(flat.float()))
        if observed_views == 0:
            raise RuntimeError("Batch contains no observed views.")
        # Preserve the ordinary all-view screening path for complete batches;
        # use pairwise masked alignment only when at least one view is absent.
        alignment_masks = (
            None if all(bool(mask.all().item()) for mask in device_masks) else device_masks
        )
        align_mgr = self.align_mgr
        if projectors is not None and projectors is not self.projectors:
            align_mgr = LatentAlignmentLossManager(self.args, projectors, self.dev)
        loss, align, _, _ = align_mgr.compute(
            latents, L_nll, iteration, self.s_nll, self.s_align,
            masks=alignment_masks,
        )
        return loss, align, bpds

    def _sync_bad(self, bad: bool) -> bool:
        if not self.is_ddp:
            return bad
        flag = torch.tensor(int(bad), device=self.dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _sync_kendall_gradients(self) -> None:
        if not self.is_ddp:
            return
        for parameter in (self.s_nll, self.s_align):
            if parameter is not None and parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)

    @torch.no_grad()
    def _init_or_update_ema(self) -> None:
        if not self.args.ema:
            return
        sources = [self._unwrap(m) for m in self.models]
        projector_sources = (
            [self._unwrap(p) for p in self.projectors]
            if self.projectors is not None else []
        )
        if self.ema_models is None:
            self.ema_models = nn.ModuleList([
                copy.deepcopy(m).eval().requires_grad_(False) for m in sources
            ])
            if projector_sources:
                self.ema_projectors = nn.ModuleList([
                    copy.deepcopy(p).eval().requires_grad_(False)
                    for p in projector_sources
                ])
            return
        decay = float(self.args.ema_decay)
        for target, source in zip(self.ema_models, sources):
            for p_target, p_source in zip(target.parameters(), source.parameters()):
                p_target.lerp_(p_source.detach(), 1.0 - decay)
            for b_target, b_source in zip(target.buffers(), source.buffers()):
                b_target.copy_(b_source)
        if self.ema_projectors is not None:
            for target, source in zip(self.ema_projectors, projector_sources):
                for p_target, p_source in zip(target.parameters(), source.parameters()):
                    p_target.lerp_(p_source.detach(), 1.0 - decay)
                for b_target, b_source in zip(target.buffers(), source.buffers()):
                    b_target.copy_(b_source)

    def _recover_after_anomaly(self, iteration: int, reason: str) -> None:
        self.anomaly_streak += 1
        def _backed_off(current_lr: float) -> float:
            # During warmup the current LR may legitimately be below min_lr.
            # An anomaly "backoff" must never increase it to the plateau floor.
            return min(
                current_lr,
                max(self.args.min_lr, current_lr * self.args.nan_lr_backoff),
            )

        backed_off_lr = _backed_off(self.opt.param_groups[0]["lr"])
        for group in self.opt.param_groups:
            group["lr"] = _backed_off(group["lr"])
        if self.rank == 0:
            tqdm.write(
                f"[anomaly] iter={iteration}: {reason}; update rejected, "
                f"streak={self.anomaly_streak}, lr={self.opt.param_groups[0]['lr']:.3g}"
            )
        if self.anomaly_streak >= self.args.anomaly_reload_after and self.state_path.exists():
            self._load_checkpoint(self.state_path, load_iteration=False)
            # Loading restores the checkpoint optimizer LR; retain the safety
            # backoff that triggered this recovery.
            for group in self.opt.param_groups:
                group["lr"] = min(group["lr"], backed_off_lr)
            self.anomaly_streak = 0
            if self.rank == 0:
                tqdm.write("[anomaly] restored the last validated checkpoint")

    def train(self) -> None:
        iterator = iter(self.train_loader)
        epoch = 0
        smooth_alpha = float(self.args.smooth_alpha)
        ema_loss_disp: Optional[float] = None
        ema_align_disp: Optional[float] = None
        pbar = tqdm(
            range(self.start_iter, self.args.max_iter + 1), desc="train-hybrid",
            disable=self.rank != 0,
        )
        for iteration in pbar:
            self.opt.zero_grad(set_to_none=True)
            loss_sum = align_sum = 0.0
            bpd_sum = np.zeros(len(self.views), dtype=np.float64)
            bpd_count = np.zeros(len(self.views), dtype=np.int64)
            bad = False
            for _micro in range(self.args.accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if self.train_sampler is not None:
                        self.train_sampler.set_epoch(epoch)
                    iterator = iter(self.train_loader)
                    batch = next(iterator)
                try:
                    with torch.amp.autocast(
                        device_type=self.dev.type,
                        dtype=self.amp_dtype,
                        enabled=self.amp_enabled,
                    ):
                        loss, align, bpds = self._batch_loss(batch, iteration)
                        scaled_loss = loss / self.args.accum_steps
                    if self._sync_bad(not bool(torch.isfinite(loss).item())):
                        bad = True
                        break
                    self.scaler.scale(scaled_loss).backward()
                    loss_sum += float(loss.detach())
                    align_sum += float(align.detach())
                    for vi, bpd in enumerate(bpds):
                        if math.isfinite(bpd):
                            bpd_sum[vi] += bpd
                            bpd_count[vi] += 1
                except (FloatingPointError, RuntimeError) as error:
                    if "out of memory" in str(error).lower() and self.dev.type == "cuda":
                        torch.cuda.empty_cache()
                    if self._sync_bad(True):
                        bad = True
                    if self.rank == 0:
                        tqdm.write(f"[anomaly] forward/backward error: {error}")
                    break

            if bad:
                self.opt.zero_grad(set_to_none=True)
                self._recover_after_anomaly(iteration, "non-finite loss or failed batch")
                continue
            self.scaler.unscale_(self.opt)
            self._sync_kendall_gradients()
            params = [p for group in self.opt.param_groups for p in group["params"]]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip)
            bad_grad = not bool(torch.isfinite(grad_norm).item())
            if self.args.exploding_grad_norm > 0:
                bad_grad = bad_grad or float(grad_norm) > self.args.exploding_grad_norm
            if self._sync_bad(bad_grad):
                self.opt.zero_grad(set_to_none=True)
                self._recover_after_anomaly(iteration, f"gradient norm={float(grad_norm):.4g}")
                continue
            parameter_snapshot = [p.detach().clone() for p in params]
            optimizer_snapshot = {
                parameter: {
                    key: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
                    for key, value in self.opt.state[parameter].items()
                }
                for parameter in params if parameter in self.opt.state
            }
            self.scaler.step(self.opt)
            self.scaler.update()
            bad_params = any(
                not bool(torch.isfinite(p).all().item()) for p in params
            )
            update_norm = math.sqrt(sum(
                float(torch.sum((p.detach() - old) ** 2).item())
                for p, old in zip(params, parameter_snapshot)
            ))
            bad_update = bad_params or (
                self.args.max_update_norm > 0
                and (not math.isfinite(update_norm) or update_norm > self.args.max_update_norm)
            )
            if self._sync_bad(bad_update):
                with torch.no_grad():
                    for parameter, old in zip(params, parameter_snapshot):
                        parameter.copy_(old)
                    for parameter, state in optimizer_snapshot.items():
                        for key, old in state.items():
                            if torch.is_tensor(old):
                                self.opt.state[parameter][key].copy_(old)
                            else:
                                self.opt.state[parameter][key] = old
                self._recover_after_anomaly(
                    iteration,
                    "non-finite parameter after step" if bad_params
                    else f"parameter update norm={update_norm:.4g}",
                )
                continue
            self.anomaly_streak = 0
            self._init_or_update_ema()
            if self.warm is not None and iteration <= self.args.warmup_iters:
                self.warm.step()
            with self.global_step.get_lock():
                self.global_step.value = iteration

            denom = float(self.args.accum_steps)
            mean_loss, mean_align = loss_sum / denom, align_sum / denom
            bpds = [
                float(bpd_sum[i] / bpd_count[i]) if bpd_count[i] else float("nan")
                for i in range(len(self.views))
            ]
            sum_bpd = sum(x for x in bpds if math.isfinite(x))
            lr = self.opt.param_groups[0]["lr"]
            val_loss = float("nan")
            do_eval = iteration % self.args.eval_interval == 0 or iteration == self.args.max_iter
            if do_eval:
                val_loss = self.validate(iteration)
                self.plateau.step(val_loss)
            if self.rank == 0:
                if smooth_alpha > 0:
                    ema_loss_disp = (
                        mean_loss if ema_loss_disp is None
                        else (1.0 - smooth_alpha) * ema_loss_disp + smooth_alpha * mean_loss
                    )
                    ema_align_disp = (
                        mean_align if ema_align_disp is None
                        else (1.0 - smooth_alpha) * ema_align_disp + smooth_alpha * mean_align
                    )
                    disp_loss, disp_align = ema_loss_disp, ema_align_disp
                else:
                    disp_loss, disp_align = mean_loss, mean_align
                pbar.set_postfix(loss=f"{disp_loss:.4f}", align=f"{disp_align:.4f}")
                row = [
                    iteration, mean_loss, mean_align, sum_bpd, val_loss, lr,
                    *bpds, *self.last_val_bpds,
                ]
                with open(self.metrics_path, "a") as stream:
                    stream.write(",".join(f"{x:.8g}" for x in row) + "\n")
                if do_eval:
                    self.save_checkpoint(iteration)
                    self._save_previews(iteration)
                    _save_hybrid_metric_plots(self.metrics_path, self.run_dir)
            if do_eval and self.is_ddp:
                dist.barrier()
            gc.collect()
        if self.rank == 0:
            self.export()
        if self.is_ddp:
            dist.barrier()
            dist.destroy_process_group()

    @torch.no_grad()
    def validate(self, iteration: int) -> float:
        models = self.ema_models if self.ema_models is not None else self.models
        projectors = (
            self.ema_projectors
            if self.ema_projectors is not None else self.projectors
        )
        for model in models:
            model.eval()
        total = torch.zeros(2 + 2 * len(self.views), device=self.dev, dtype=torch.float64)
        for batch_index, batch in enumerate(self.val_loader):
            with torch.amp.autocast(
                device_type=self.dev.type, dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                loss, _, bpds = self._batch_loss(
                    batch, iteration, models=models, projectors=projectors
                )
            if torch.isfinite(loss):
                total[0] += loss.double()
                total[1] += 1
            for vi, bpd in enumerate(bpds):
                if math.isfinite(bpd):
                    total[2 + 2 * vi] += bpd
                    total[3 + 2 * vi] += 1
            if batch_index + 1 >= self.args.val_batches:
                break
        if self.is_ddp:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        for model in self.models:
            model.train()
        value = float((total[0] / total[1]).item()) if total[1] else float("nan")
        self.last_val_bpds = [
            float((total[2 + 2 * vi] / total[3 + 2 * vi]).item())
            if total[3 + 2 * vi] else float("nan")
            for vi in range(len(self.views))
        ]
        if self.rank == 0:
            details = ", ".join(
                f"{view.name}={bpd:.4g}"
                for view, bpd in zip(self.views, self.last_val_bpds)
            )
            tqdm.write(
                f"[val] iter={iteration} loss={value:.6g} "
                f"({'EMA' if self.ema_models else 'base'}; {details})"
            )
        return value

    def _checkpoint_blob(self, iteration: int) -> Dict[str, Any]:
        return {
            "iter": iteration + 1,
            "models": [self._unwrap(model).state_dict() for model in self.models],
            "ema_models": (
                [model.state_dict() for model in self.ema_models]
                if self.ema_models is not None else None
            ),
            "projectors": (
                [self._unwrap(p).state_dict() for p in self.projectors]
                if self.projectors is not None else None
            ),
            "ema_projectors": (
                [p.state_dict() for p in self.ema_projectors]
                if self.ema_projectors is not None else None
            ),
            "optimizer": self.opt.state_dict(),
            "warmup": self.warm.state_dict() if self.warm is not None else None,
            "plateau": self.plateau.state_dict(),
            "scaler": self.scaler.state_dict(),
            "normalizers": {k: v.state_dict() for k, v in self.normalizers.items()},
            "views": [view.__dict__ for view in self.views],
            "augmentation_groups": self.augmentation_groups,
            "config": {**vars(self.args), "s_cap_wired_to_conv": True},
            "kendall": {
                "s_nll": None if self.s_nll is None else float(self.s_nll.detach().cpu()),
                "s_align": None if self.s_align is None else float(self.s_align.detach().cpu()),
            },
        }

    def save_checkpoint(self, iteration: int) -> None:
        blob = self._checkpoint_blob(iteration)
        torch.save(blob, self.state_path)
        milestone = self.run_dir / f"training_state_it{iteration:06d}.pt"
        torch.save(blob, milestone)
        self.cleanup_checkpoints()
        free_gb = shutil.disk_usage(self.run_dir).free / 2**30
        if free_gb < self.args.disk_warning_gb:
            tqdm.write(f"[disk warning] only {free_gb:.1f} GiB free in {self.run_dir}")

    def cleanup_checkpoints(self) -> None:
        files = sorted(self.run_dir.glob("training_state_it*.pt"))
        keep = set(files[-self.args.keep_last :]) if self.args.keep_last else set()
        for path in files:
            try:
                iteration = int(path.stem.rsplit("it", 1)[1])
            except (ValueError, IndexError):
                keep.add(path)
                continue
            if self.args.keep_every > 0 and iteration % self.args.keep_every == 0:
                keep.add(path)
        for path in files:
            if path not in keep:
                path.unlink()

    def _reconcile_config_with_checkpoint(self, resume_path: Path) -> None:
        """Compare the current --config views/architecture against a checkpoint.

        Without --use-ckpt-config, a resumed run whose --config drifted from
        the one used to produce the checkpoint (different shape, K, hidden
        width, base distribution, ...) would previously proceed silently and
        either crash deep inside `load_state_dict` with a confusing shape
        error, or -- if shapes happened to coincide -- load weights onto the
        wrong architecture without any warning. This fails fast instead, with
        --use-ckpt-config as the explicit opt-in to adopt the checkpoint's
        architecture (mirroring BaseLAMNrTrainer's --use-ckpt-config).
        """
        blob_cpu = torch.load(resume_path, map_location="cpu", weights_only=False)
        ckpt_views = blob_cpu.get("views")
        ckpt_cfg = blob_cpu.get("config", {})
        if not ckpt_views:
            return

        current_by_name = {v.name: v for v in self.views}
        view_mismatches: List[Tuple[str, str, Any, Any]] = []
        for raw in ckpt_views:
            name = raw.get("name")
            current = current_by_name.get(name)
            if current is None:
                view_mismatches.append((name, "<view>", "absent from --config", "present in checkpoint"))
                continue
            for field_name in ("kind", "shape", "channels", "columns", "model"):
                current_value = getattr(current, field_name)
                ckpt_value = raw.get(field_name)
                # Export-only diagnostics may deliberately tighten these
                # parameter-free numerical guards on an existing checkpoint.
                # Keep strict matching for resumed training, where changing
                # the transform semantics mid-run would be unsafe.
                if field_name == "model" and self.args.export_only:
                    diagnostic_keys = {"shift_cap", "gen_clamp"}
                    current_value = {
                        k: v for k, v in current_value.items()
                        if k not in diagnostic_keys
                    }
                    ckpt_value = {
                        k: v for k, v in (ckpt_value or {}).items()
                        if k not in diagnostic_keys
                    }
                if current_value != ckpt_value:
                    view_mismatches.append((name, field_name, current_value, ckpt_value))

        global_arch_keys = [
            "image_L", "image_K", "image_hidden", "image_base", "grad_checkpoint",
            "scale_cap", "tabular_K", "tabular_hidden", "tabular_base",
        ]
        arg_mismatches = [
            k for k in global_arch_keys
            if k in ckpt_cfg and getattr(self.args, k, None) != ckpt_cfg[k]
        ]

        if not view_mismatches and not arg_mismatches:
            return

        if self.args.use_ckpt_config:
            if view_mismatches:
                self.views = [HybridViewSpec.from_dict(raw) for raw in ckpt_views]
            for k in arg_mismatches:
                setattr(self.args, k, ckpt_cfg[k])
            if self.rank == 0:
                tqdm.write(
                    f"[resume] --use-ckpt-config: adopted the checkpoint's architecture "
                    f"({len(view_mismatches)} view field(s), {len(arg_mismatches)} "
                    f"global arg(s) overridden from {resume_path})."
                )
        else:
            details = "; ".join(
                f"view {name!r} field {field!r}: --config={current!r} vs checkpoint={ckpt!r}"
                for name, field, current, ckpt in view_mismatches
            )
            if arg_mismatches:
                arg_details = "; ".join(
                    f"{k}: --config={getattr(self.args, k)!r} vs checkpoint={ckpt_cfg[k]!r}"
                    for k in arg_mismatches
                )
                details = f"{details}; {arg_details}" if details else arg_details
            raise ValueError(
                "Resume architecture mismatch between --config and the checkpoint "
                f"at {resume_path}: {details}. Fix --config to match the checkpoint, "
                "or pass --use-ckpt-config to adopt the checkpoint's architecture."
            )

    def _load_checkpoint(self, path: Path, load_iteration: bool = True) -> int:
        blob = torch.load(path, map_location=self.dev, weights_only=False)
        for model, state in zip(self.models, blob["models"]):
            self._unwrap(model).load_state_dict(state)
        if self.projectors is not None and blob.get("projectors") is not None:
            for projector, state in zip(self.projectors, blob["projectors"]):
                self._unwrap(projector).load_state_dict(state)
        self.opt.load_state_dict(blob["optimizer"])
        if self.warm is not None and blob.get("warmup") is not None:
            self.warm.load_state_dict(blob["warmup"])
        if blob.get("plateau") is not None:
            self.plateau.load_state_dict(blob["plateau"])
        # A disabled GradScaler serializes to an empty dict.  Passing that
        # empty state back to load_state_dict raises, even though there is no
        # scaling state to restore (for example, a float32 checkpoint loaded
        # by an export-only diagnostic run).
        scaler_state = blob.get("scaler")
        if scaler_state:
            self.scaler.load_state_dict(scaler_state)
        if self.args.ema and blob.get("ema_models") is not None:
            self.ema_models = nn.ModuleList([
                copy.deepcopy(self._unwrap(m)).eval().requires_grad_(False)
                for m in self.models
            ])
            for model, state in zip(self.ema_models, blob["ema_models"]):
                model.load_state_dict(state)
            if self.projectors is not None and blob.get("ema_projectors") is not None:
                self.ema_projectors = nn.ModuleList([
                    copy.deepcopy(self._unwrap(p)).eval().requires_grad_(False)
                    for p in self.projectors
                ])
                for projector, state in zip(self.ema_projectors, blob["ema_projectors"]):
                    projector.load_state_dict(state)
        for name, state in blob.get("normalizers", {}).items():
            if name in self.normalizers:
                self.normalizers[name].load_state_dict(state)
        kendall = blob.get("kendall", {})
        if self.s_nll is not None and kendall.get("s_nll") is not None:
            self.s_nll.data.fill_(float(kendall["s_nll"]))
        if self.s_align is not None and kendall.get("s_align") is not None:
            self.s_align.data.fill_(float(kendall["s_align"]))
        return int(blob.get("iter", 1)) if load_iteration else self.start_iter

    def _reconstruct(self, model: nn.Module, x: torch.Tensor, view: HybridViewSpec):
        flow = self._flow(model)
        if view.kind == "tabular":
            z, _ = _inverse_with_guard(flow, x)
        else:
            z, _ = flow.inverse_and_log_det(x)
        reconstructed, _ = flow.forward_and_log_det(z)
        return z, reconstructed

    @staticmethod
    def _display_slice(tensor: torch.Tensor, mode: str = "clamp") -> np.ndarray:
        array = tensor.detach().float().cpu().numpy()
        if array.ndim == 4:  # C,H,W,D
            array = array[0, :, :, array.shape[-1] // 2]
        elif array.ndim == 3:  # C,H,W
            array = array[0]
        array = np.nan_to_num(array)

        def _minmax(a: np.ndarray) -> np.ndarray:
            lo, hi = float(a.min()), float(a.max())
            return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        def _pclamp(a: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(a, [1, 99])
            return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        # "clamp" (default) preserves the previous reconstruction-preview
        # behavior exactly. "to01"/"both" mirror --sample-grid-norm from the
        # 2D/3D trainers and are used for the *generated-samples* grid only.
        if mode == "to01":
            return _minmax(array)
        if mode == "both":
            return _minmax(_pclamp(array))
        return _pclamp(array)

    @staticmethod
    def _write_grid(images: Sequence[np.ndarray], path: Path, columns: int = 4) -> None:
        from PIL import Image, ImageDraw

        if not images:
            return
        tiles = [Image.fromarray((image * 255).astype(np.uint8), mode="L") for image in images]
        width, height = max(x.width for x in tiles), max(x.height for x in tiles)
        rows = math.ceil(len(tiles) / columns)
        canvas = Image.new("L", (columns * width, rows * height), color=0)
        draw = ImageDraw.Draw(canvas)
        for index, tile in enumerate(tiles):
            canvas.paste(tile, ((index % columns) * width, (index // columns) * height))
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)

    @torch.no_grad()
    def _save_previews(self, iteration: int) -> None:
        if self.args.preview_interval <= 0 or iteration % self.args.preview_interval:
            return
        models = self.ema_models if self.ema_models is not None else self.models
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return
        values, masks = batch
        preview_dir = self.run_dir / "previews"
        for value, mask, view, model in zip(values, masks, self.views, models):
            if view.kind == "tabular":
                continue
            x = self._prepare(value, view)
            present = mask.to(self.dev).bool()
            if not present.any():
                continue
            x = x[present][: self.args.preview_samples]
            z, recon = self._reconstruct(model, x, view)
            finite = torch.isfinite(recon)
            safe_recon = torch.nan_to_num(recon.float())
            error = (safe_recon - x.float()).abs()
            latent_levels = z if isinstance(z, (list, tuple)) else [z]
            latent_abs_max = max(
                float(torch.nan_to_num(level.detach().float()).abs().max().cpu())
                for level in latent_levels
            )
            tqdm.write(
                f"[recon] iter={iteration} view={view.name} "
                f"mae={float(error.mean().cpu()):.6g} "
                f"rmse={float(error.square().mean().sqrt().cpu()):.6g} "
                f"max={float(error.max().cpu()):.6g} "
                f"finite={float(finite.float().mean().cpu()):.6f} "
                f"x_range=[{float(x.min().cpu()):.6g},{float(x.max().cpu()):.6g}] "
                f"recon_range=[{float(safe_recon.min().cpu()):.6g},"
                f"{float(safe_recon.max().cpu()):.6g}] "
                f"latent_abs_max={latent_abs_max:.6g}"
            )
            panels = []
            for original, restored in zip(x, recon):
                panels.extend([self._display_slice(original), self._display_slice(restored)])
            self._write_grid(
                panels, preview_dir / f"{view.name}_recon_it{iteration:06d}.png",
                columns=2,
            )
            if self.args.sample_mode == "model":
                flow = self._flow(model)
                # Chunk generative sampling instead of drawing all
                # preview_samples in one shot, to bound the peak memory of
                # deep Glow flows at full resolution (mirrors
                # BaseLAMNrTrainer's --sample-chunk-size / _save_samples_grid).
                chunk_size = max(1, int(self.args.sample_chunk_size))
                remaining = int(self.args.preview_samples)
                sampled_chunks: List[torch.Tensor] = []
                while remaining > 0:
                    k = min(chunk_size, remaining)
                    chunk = flow.sample(k, temperature=self.args.sample_temp)
                    chunk = chunk[0] if isinstance(chunk, (tuple, list)) else chunk
                    sampled_chunks.append(chunk.detach().cpu())
                    remaining -= k
                    if self.dev.type == "cuda":
                        torch.cuda.empty_cache()
                sampled = torch.cat(sampled_chunks, dim=0)
                self._write_grid(
                    [
                        self._display_slice(item, mode=self.args.sample_grid_norm)
                        for item in sampled
                    ],
                    preview_dir / f"{view.name}_samples_it{iteration:06d}.png",
                )

    @torch.no_grad()
    def export(self) -> None:
        if not (self.args.save_z or self.args.save_whitened or self.args.save_recon):
            return
        models = self.ema_models if self.ema_models is not None else self.models
        dataset = HybridManifestDataset(
            self.full_frame, self.views, self.normalizers,
            augmentation_groups=self.augmentation_groups, do_augmentation=False,
        )
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers, collate_fn=hybrid_collate,
        )
        export_dir = self.run_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        tabular = {
            view.name: {"z": [], "whitened": [], "recon": [], "row": []}
            for view in self.views if view.kind == "tabular"
        }
        image_records: Dict[str, List[Dict[str, Any]]] = {
            view.name: [] for view in self.views if view.kind != "tabular"
        }
        offset = 0
        for values, masks in loader:
            batch_size = len(masks[0])
            for value, mask, view, model in zip(values, masks, self.views, models):
                present_cpu = mask.bool()
                present = present_cpu.to(self.dev)
                if not present.any():
                    continue
                rows = np.flatnonzero(present_cpu.numpy()) + offset
                x = self._prepare(value, view)[present]
                z, recon = self._reconstruct(model, x, view)
                if view.kind == "tabular":
                    record = tabular[view.name]
                    record["row"].extend(rows.tolist())
                    if self.args.save_z:
                        record["z"].append(z.detach().float().cpu().numpy())
                    if self.args.save_whitened:
                        record["whitened"].append(
                            _extract_whitened(self._flow(model), z).detach().float().cpu().numpy()
                        )
                    if self.args.save_recon:
                        record["recon"].append(
                            self.normalizers[view.name].inverse_transform(
                                recon.detach().float().cpu()
                            ).numpy()
                        )
                elif self.args.save_recon:
                    view_dir = export_dir / view.name / "reconstructions"
                    view_dir.mkdir(parents=True, exist_ok=True)
                    remaining = max(0, self.args.export_max_samples - len(image_records[view.name]))
                    for row, image in list(zip(rows, recon.detach().float().cpu()))[:remaining]:
                        output = view_dir / f"row_{int(row):06d}.npy"
                        np.save(output, image.numpy())
                        image_records[view.name].append({"row": int(row), "path": str(output)})
            offset += batch_size
        for view in (v for v in self.views if v.kind == "tabular"):
            record = tabular[view.name]
            ids = pd.DataFrame({"row": record["row"]})
            if self.args.save_z and record["z"]:
                array = np.concatenate(record["z"])
                ids.join(pd.DataFrame(array, columns=[f"z{i}" for i in range(array.shape[1])])).to_csv(
                    export_dir / f"{view.name}_latents.csv", index=False
                )
            if self.args.save_whitened and record["whitened"]:
                array = np.concatenate(record["whitened"])
                ids.join(pd.DataFrame(array, columns=[f"epsilon{i}" for i in range(array.shape[1])])).to_csv(
                    export_dir / f"{view.name}_whitened.csv", index=False
                )
            if self.args.save_recon and record["recon"]:
                array = np.concatenate(record["recon"])
                ids.join(pd.DataFrame(array, columns=view.columns)).to_csv(
                    export_dir / f"{view.name}_reconstructions.csv", index=False
                )
        for name, records in image_records.items():
            if records:
                pd.DataFrame(records).to_csv(export_dir / f"{name}_reconstructions.csv", index=False)
        print(f"[export] wrote EMA-aware outputs to {export_dir}")


def _build_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Hybrid tabular/2D/3D LAMNr trainer")
    parser.add_argument(
        "--manifest", default="",
        help="Row-aligned CSV. If omitted, build it from each view's glob/CSV config.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="runs_hybrid")
    parser.add_argument("--subject-column", default="")
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=0,
        help="Virtual epoch length; 0 uses the physical training-set length.")
    parser.add_argument("--val-samples", type=int, default=0,
        help="Virtual validation length; 0 uses the physical validation-set length.")
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--lr-decay-gamma", type=float, default=1.0)
    parser.add_argument("--lr-decay-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--exploding-grad-norm", type=float, default=0.0,
        help=(
            "Optional pre-clipping gradient-norm rejection threshold; 0 disables it. "
            "Finite gradients are still clipped by --grad-clip."
        ),
    )
    parser.add_argument("--max-update-norm", type=float, default=1e3)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument(
        "--ddp-find-unused", action=argparse.BooleanOptionalAction, default=False,
        help="Enable DDP's unused-parameter graph traversal only for conditional models.",
    )
    parser.add_argument("--precision", default="mixed", choices=["float", "mixed"])
    parser.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=4)
    parser.add_argument("--plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--plateau-cooldown", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--nan-lr-backoff", type=float, default=0.5)
    parser.add_argument("--anomaly-reload-after", type=int, default=5)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--use-ckpt-config", action="store_true",
        help="On resume, adopt the checkpoint's view/architecture config "
             "instead of failing on a mismatch with the current --config.")
    parser.add_argument("--extra-iters", type=int, default=0,
        help="After resuming, run this many additional iterations past the "
             "checkpoint's iteration instead of stopping at --max-iter.")
    parser.add_argument("--subject-limit", type=int, default=0,
        help="Debug: cap the number of subjects/rows loaded from the "
             "manifest to this many; 0 disables the cap.")
    parser.add_argument("--detect-anomaly", action="store_true",
        help="Enable torch.autograd.set_detect_anomaly(True) to pinpoint the "
             "exact forward op behind a NaN/Inf gradient. Substantially slower.")
    parser.add_argument("--smooth-alpha", type=float, default=0.1,
        help="EMA smoothing factor for the loss/align values shown in the "
             "progress bar; does not affect the raw values logged to metrics.csv.")
    parser.add_argument("--image-noise-std", type=float, default=0.05)
    parser.add_argument("--tabular-noise-std", type=float, default=0.0)
    parser.add_argument("--disable-augmentation", action="store_true",
        help="Disable all training-data augmentation. Validation is always clean.")
    parser.add_argument("--augmentation-transform-type",
        default="affineAndDeformation",
        choices=["translation", "rigid", "scaleShear", "affine",
                 "deformation", "affineAndDeformation"])
    parser.add_argument("--augmentation-sd-affine", type=float, default=0.05)
    parser.add_argument("--augmentation-sd-deformation", type=float, default=10.0)
    parser.add_argument("--augmentation-noise-model", default="additivegaussian",
        choices=["additivegaussian", "saltandpepper", "shot", "speckle"])
    parser.add_argument("--augmentation-sd-bias-field", type=float, default=1e-8)
    parser.add_argument("--augmentation-sd-histogram-warping", type=float, default=0.025)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument("--aug-schedules", default=(
        "noise_std:cos:0.05->0.00@150k,"
        "sd_affine:linear:0.05->0.00@80k,"
        "sd_deformation:cos:0.20->0.00@100k,"
        "sd_simulated_bias_field:cos:1.00->0.00@120k,"
        "sd_histogram_warping:exp:0.05->0.00@120k"
    ))
    parser.add_argument("--disable-aug-anneal", action="store_true")

    parser.add_argument("--image-L", type=int, default=3)
    parser.add_argument("--image-K", type=int, default=16)
    parser.add_argument("--image-hidden", type=int, default=128)
    parser.add_argument("--image-base", default="glow", choices=["glow", "diag"])
    parser.add_argument("--grad-checkpoint", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--scale-cap", type=float, default=3.0)
    parser.add_argument("--tabular-K", type=int, default=32)
    parser.add_argument("--tabular-hidden", type=int, default=None)
    parser.add_argument("--tabular-base", default="GaussianPCA",
                        choices=["DiagGaussian", "GaussianPCA"],
                        help="Matches the standalone tabular trainer's default "
                             "(GaussianPCA). Override per view via the config's "
                             '"model": {"base_distribution": ...}.')

    parser.add_argument("--align", default="vicreg",
        choices=["none", "infonce", "barlow", "vicreg", "hsic", "pearson", "mse"])
    parser.add_argument("--align-weight", type=float, default=0.05)
    parser.add_argument("--align-warmup", type=int, default=200)
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--proj-hidden", type=int, default=128)
    parser.add_argument("--alignment-latents", default="all-pooled",
        choices=["all-pooled", "all-flat"])
    parser.add_argument("--alignment-pool-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--barlow-lambda", type=float, default=5e-3)
    parser.add_argument("--weighting", default="fixed", choices=["fixed", "kendall"])
    parser.add_argument("--init-logvar-nll", type=float, default=0.0,
        help="Initial value of the learnable NLL log-variance under --weighting kendall.")
    parser.add_argument("--init-logvar-align", type=float, default=0.0,
        help="Initial value of the learnable alignment log-variance under --weighting kendall.")
    parser.add_argument("--vicreg-inv", type=float, default=25.0)
    parser.add_argument("--vicreg-cov", type=float, default=1.0)
    parser.add_argument("--vicreg-var", type=float, nargs="+", default=[25.0])
    parser.add_argument("--vicreg-gamma", type=float, nargs="+", default=[1.0])
    parser.add_argument("--hsic-sigma", type=float, default=0.0)
    parser.add_argument("--screen", default="none", choices=["none", "cca", "hsic"])
    parser.add_argument("--screen-warmup", type=int, default=500)
    parser.add_argument("--screen-refresh", type=int, default=0)
    parser.add_argument("--screen-frac", type=float, default=0.5)
    parser.add_argument("--cca-ridge", type=float, default=1e-3)
    parser.add_argument("--prefilter-frac", type=float, default=0.5)
    parser.add_argument("--preview-interval", type=int, default=500)
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--sample-mode", default="model", choices=["off", "model"])
    parser.add_argument("--sample-temp", type=float, default=1.0)
    parser.add_argument("--sample-grid-norm", default="to01",
        choices=["to01", "clamp", "both"],
        help="Display normalization for the generated-samples preview grid "
             "(reconstruction previews always use percentile clamping).")
    parser.add_argument("--sample-chunk-size", type=int, default=20,
        help="Max number of samples generated per flow.sample() call during "
             "preview grids, to bound the peak memory of generative sampling.")
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--keep-every", type=int, default=5000)
    parser.add_argument("--disk-warning-gb", type=float, default=10.0)
    parser.add_argument("--save-z", action="store_true")
    parser.add_argument("--save-whitened", action="store_true")
    parser.add_argument("--save-recon", action="store_true")
    parser.add_argument("--export-max-samples", type=int, default=100)
    parser.add_argument("--export-only", action="store_true",
        help="Load --resume/--auto-resume and run reconstruction/latent export only.")
    parser.add_argument("--diagnose-invertibility", action="store_true",
        help="Build/load the configured models, print image round-trip metrics, "
             "write reconstruction previews at iteration 0, and exit without training.")
    args = parser.parse_args(argv)
    if args.accum_steps < 1:
        parser.error("--accum-steps must be >= 1")
    if args.keep_last < 0 or args.keep_every < 0:
        parser.error("checkpoint retention values must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    trainer = HybridLAMNrTrainer()
    args = _build_args(argv)
    trainer.setup(args)
    if args.diagnose_invertibility:
        trainer._save_previews(0)
        if trainer.is_ddp:
            dist.barrier()
            dist.destroy_process_group()
    elif args.export_only:
        if not (args.resume or args.auto_resume):
            raise ValueError("--export-only requires --resume or --auto-resume.")
        trainer.export()
        trainer._save_previews(max(0, trainer.start_iter - 1))
    else:
        trainer.train()


if __name__ == "__main__":
    main()
