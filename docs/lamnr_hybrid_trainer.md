# Hybrid LAMNr trainer

`train_lamnr_flows_hybrid.py` trains heterogeneous LAMNr views in one loop:

- tabular vectors with RealNVP;
- 2D images with Glow 2D;
- 3D images with Glow 3D.

Each view preserves its own exact-likelihood flow and has its own projection
head. Projection heads map heterogeneous latent dimensions to the common
`--proj-dim` used by VICReg, InfoNCE, Barlow Twins, HSIC, Pearson, or MSE.

## Manifest

Use one CSV row per biologically paired observation. Include a subject column
so all visits from one participant remain in the same train/validation split.

```csv
SubjectID,TimepointID,T1MTLFile,T2MTLFile,Braak1_2,Braak3_4,Braak5_6
OAS30001,d3746,/path/t1.nii.gz,/path/t2.nii.gz,1.153,1.163,1.259
OAS30002,d1200,/path/t1.nii.gz,,,,
```

Empty paths or incomplete tabular vectors mark a missing view. NLL is computed
only for observed rows. Alignment is computed separately for every pair of
views on the intersection of rows where both views are observed. A pair needs
at least two observations in a batch; otherwise its alignment term is skipped
for that batch.

The trainer does not perform temporal matching. For OASIS-3, construct each row
after selecting the PET/clinical observation appropriate to the MRI visit and
retain the day difference in the manifest for auditing.

## View configuration

```json
{
  "subject_column": "SubjectID",
  "augmentation_groups": {
    "MTL": {
      "transform_type": "affineAndDeformation",
      "sd_affine": 0.05,
      "sd_deformation": 10.0,
      "noise_model": "additivegaussian",
      "noise_parameters": [0.0, 0.05],
      "sd_simulated_bias_field": 0.00000001,
      "sd_histogram_warping": 0.025,
      "schedules": "noise_std:cos:0.05->0@150k,sd_affine:linear:0.05->0@80k,sd_deformation:cos:0.20->0@100k,sd_simulated_bias_field:cos:1.0->0@120k,sd_histogram_warping:exp:0.05->0@120k"
    }
  },
  "views": [
    {
      "name": "T1_MTL",
      "type": "image3d",
      "path_column": "T1MTLFile",
      "shape": [40, 40, 64],
      "augmentation_group": "MTL",
      "model": {"L": 3, "K": 16, "hidden": 128}
    },
    {
      "name": "T2_MTL",
      "type": "image3d",
      "path_column": "T2MTLFile",
      "shape": [40, 40, 64],
      "augmentation_group": "MTL",
      "model": {"L": 3, "K": 16, "hidden": 128}
    },
    {
      "name": "tau",
      "type": "tabular",
      "columns": ["Braak1_2", "Braak3_4", "Braak5_6"],
      "normalization": "0mean",
      "augmentation": {
        "tabular_noise_std": 0.02,
        "schedules": "tabular_noise_std:cos:0.02->0@100k"
      },
      "model": {"K": 32, "hidden": 64, "base_distribution": "DiagGaussian"}
    }
  ]
}
```

Every tabular view currently needs at least two columns because the RealNVP
affine-coupling masks are degenerate in one dimension. Image files may be
NIfTI, NumPy (`.npy`), or Torch tensors (`.pt`/`.pth`); common raster formats
are also accepted for 2D views.

## Training

```bash
python -m antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid \
  --manifest oasis3_hybrid_manifest.csv \
  --config oasis3_hybrid_views.json \
  --out-dir runs/oasis3_hybrid \
  --devices mps \
  --batch-size 4 \
  --align vicreg \
  --alignment-latents all-pooled \
  --alignment-pool-size 2
```

`all-pooled` includes every multiscale image latent level while controlling the
projector size. `all-flat` includes every latent coordinate and can require
substantially more memory.

For one process using several local GPUs, pass a comma-separated device list,
for example `--devices cuda:0,cuda:1` (DataParallel). For production training,
DDP is preferred:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid \
  --manifest oasis3_hybrid_manifest.csv \
  --config oasis3_hybrid_views.json \
  --devices cuda --out-dir runs/oasis3_hybrid
```

DDP shards both loaders, synchronizes anomaly decisions and Kendall gradients,
and evaluates collectively. ActNorm is initialized from real observations
before DDP broadcasts rank 0's parameters and buffers. Only rank 0 writes
metrics, previews, exports, and checkpoints.

## Optimization and resilience

The hybrid loop includes the mechanisms used by the specialized trainers:

- CUDA AMP (`--precision mixed`, with `--amp-dtype fp16|bf16`);
- gradient accumulation (`--accum-steps`), clipping, and explicit exploding
  gradient/update thresholds;
- EMA flows and projectors, enabled by default and used for validation,
  previews, and export (`--no-ema` disables them);
- warmup followed by `ReduceLROnPlateau`;
- synchronized rejection of non-finite losses/gradients/parameters, learning
  rate backoff, and restoration of the last validated checkpoint after a
  repeated anomaly streak;
- Glow gradient checkpointing (`--grad-checkpoint auto|on|off`);
- milestone retention with `--keep-last` and `--keep-every`, plus a free-disk
  warning.

`metrics.csv` contains training and validation BPD for every named view. Metric
plots and per-view original/reconstruction grids are refreshed at evaluation
time. `--sample-mode model` also writes unconditional image samples; use
`--sample-mode off` when previews would be too expensive.

## Glob sources and virtual sampling

The manifest may be generated in memory by omitting `--manifest`. In that
case, set top-level `join_key`, and give each image view `glob` (a string or
list), optionally `key_regex`, and each tabular view `csv` plus `key_column`:

```json
{
  "join_key": "visit_id",
  "views": [
    {"name": "T1", "type": "image3d", "path_column": "T1File",
     "shape": [40, 40, 64], "glob": ["/data/T1/**/*.nii.gz"],
     "key_regex": "/(?P<key>OAS3[^/]+)_T1w"},
    {"name": "tau", "type": "tabular", "columns": ["Braak1_2", "Braak3_4"],
     "csv": "/data/tau.csv", "key_column": "visit_id"}
  ]
}
```

Sources are outer-joined, so unmatched records become missing views rather
than being discarded. `--train-samples` and `--val-samples` set virtual loader
lengths; indices wrap over the physical manifest, matching ImageDataset-style
virtual sampling.

## Export and image reconstruction

At the end of training, `--save-z`, `--save-whitened`, and `--save-recon`
write EMA-based results under `OUT/export`. Tabular outputs are CSV files with
the original manifest row. Image reconstructions are NumPy tensors plus a CSV
index, capped by `--export-max-samples`. The PNG grids under `OUT/previews`
provide central-slice reconstruction analysis for 2D and 3D views.

An existing checkpoint can be analyzed without training:

```bash
python -m antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid \
  --manifest oasis3_hybrid_manifest.csv --config oasis3_hybrid_views.json \
  --out-dir runs/oasis3_hybrid --resume runs/oasis3_hybrid/training_state.pt \
  --export-only --save-z --save-whitened --save-recon
```

## Data augmentation

Training supports the augmentation mechanisms from the specialized trainers:

- shared translation, rigid, scale/shear, affine, deformation, or
  affine-and-deformation transforms through ANTs;
- additive Gaussian, salt-and-pepper, shot, or speckle noise;
- simulated bias fields;
- histogram warping;
- shared horizontal flips for raster-style 2D data;
- normalized-space Gaussian jitter for tabular views;
- deterministic parameter schedules driven by the optimizer global step.

Views with the same `augmentation_group` must have the same spatial dimension
and shape. They receive one common geometric transform, preserving T1/T2 or
other voxelwise correspondence. Intensity augmentation is applied to every
modality in the group. Views without a group receive an implicit private group.

Validation is always unaugmented. Use `--disable-augmentation` to disable all
training augmentation or `--disable-aug-anneal` to keep the configured
augmentation strengths constant. Group settings in JSON override the CLI
defaults; a view-level `augmentation` dictionary is convenient for an
ungrouped image or a tabular view.

The implementation deliberately shares its data primitives with the
specialized datasets. `ImageDataset` and the hybrid image loader both call
`augment_image_modalities()` from `antstorch.utilities.image_dataset`.
`MultiViewDataFrameDataset`, `TabularNormalizer`, and hybrid tabular views all
call `transform_tabular_numeric()` from
`antstorch.utilities.dataframe_dataset`. The hybrid dataset adds only manifest
dispatch, availability masks, and grouping across heterogeneous view types.
