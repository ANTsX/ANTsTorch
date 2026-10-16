# Benchmarks

- **[Registration](#registration)** using [Mindboggle-101](frontiersin.org/Brain_Imaging_Methods/10.3389/fnins.2012.00171/full) data
    * [ANTsTorch](https://github.com/ANTsX/ANTsTorch) 
    * [syntx](https://github.com/stnava/syntx)
    * [FireANTs](https://github.com/rohitrango/fireants) 

- **[Primitives](#primitive-comparisons-n4-denoising-cortical-thickness-b-spline-fitting)** — ANTs vs ANTsTorch
    * N4 bias-field correction
    * denoising
    * cortical thickness (KellyKapowski)
    * B-spline scattered-data approximation
    

## Registration

1. [Overview](#overview)
    * [Models](#models)
    * [Optimizers](#optimizers)
2. [FireANTs](#fireants)
3. [The Mindboggle benchmark](#the-mindboggle-benchmark)
4. [Results](#results)
5. [Reproducing the runs](#reproducing-the-runs)


### Overview

Most of ANTsTorch's registration code follows [syntx](https://github.com/stnava/syntx).  ANTsTorch also includes a stationary velocity flow (SVF) framework.

- **The dense SyN formulation:** symmetric, midpoint-based, Eulerian fields, local cross-correlation (`cc2`), 4-2-1 pyramid, and the fluid regularizers (Gaussian, Sobolev, DST-I).
- **`greedy`:** the one-directional compositive model of `syntx.greedy`, including syntx's millimetre-isotropic deformable pyramid (`antstorch.greedy`, bit-identical to syntx on CPU).
- **The benchmark protocol:** the 90 Mindboggle pairs, N4 pre-processing, the DKT31 Dice measures, and the
  model names (`gaussian`, `sobolev`, `dsti`, ...).
- **The syntx default hyper-parameters** (`syntx_defaults=True`), selected per regularizer and optimizer.

<!-- Where ANTsTorch differs, it is by implementation (PyTorch-native sampling through`antstorch._torch_compat`, autograd or analytical gradients, MPS/CPU/CUDA support, the `antstorch.syn` / `antstorch.greedy` / `antstorch.bspline_flows` modules) and not by algorithm; the parity tests live in `tests/greedy` and the benchmark scripts of this directory. -->

### Models

| Family | ANTsTorch entry point | Model names | Inverse |
|---|---|---|---|
| Dense SyN | `antstorch.syn.syn_registration` | `gaussian_syn`, `sobolev_syn`, `dsti_syn`, `bspline_syn` | yes (symmetric) |
| Greedy (compositive) | `antstorch.greedy.greedy_registration` (alias `greedy`) | `greedy`, `greedy_regadam` | no (optional Anderson inverse with `return_inverse=True`) |
| Stationary velocity field (SVF) | `antstorch.bspline_flows.bspline_svf_registration`, `gaussian_svf_registration` | `bspline_svf` (alias `svf`), `gaussian_svf` | yes (by integration) |
| ANTs baselines | `ants.registration` | `ants_syn_quick`, any deformable-only `type_of_transform` | — |

- **Dense SyN** differs between variants only by the regularizer applied to the update field.
- **Greedy** estimates one forward displacement field and has no native inverse; its `dice_moving` and `dice_sym` depend on a post-hoc inverse, so `dice_fixed` is its primary measure.
- **SVF** parameterizes a velocity field (B-spline or dense Gaussian-smoothed) and exponentiates it.

### Optimizers

| Optimizer | Where | What it does |
|---|---|---|
| `gradient_descent` | `syn_registration` default | CFL-scaled step along the regularized gradient |
| `reg_adam` | `syn_registration`, `greedy` (name `regadam`) | `greedy.py` Adam: bias-corrected moments, the Adam quotient is smoothed (`regadam_sigma` 0.8); moments reset at each level |
| `syn_reg_adam` | `syn_registration` (sobolev, dsti only) | `syntx.syn()` variant: gradient regularized, Adam quotient, Green operator applied again, CFL step shared by both half-warps; used by the `dsti` fair-mode arm |

### FireANTs

[FireANTs](https://github.com/rohitrango/FireANTs) (Jena, Chaudhari, Gee; *Nature Communications*, 2024) is a
GPU-accelerated registration toolkit that optimizes with Riemannian Adam. In this benchmark it enters as an optional third arm (`--fireants`, CUDA only) through its `GreedyRegistration`: unidirectional, compositive, no native inverse.

[syntx's evaluation of FireANTs on Mindboggle](https://github.com/stnava/syntx/blob/main/docs/v_fire_ants/README.md). Main points: 
* FireANTs' default SyN plateaus near 0.39 DKT Dice,
* Greedy setting reaches about 0.62–0.63 at roughly 3.5x the speed, but 3.7–15.6 % folded voxels (vs 0.0005 % for `syntx.syn`), no reliable inverse. 
    

### The Mindboggle benchmark

[Mindboggle-101](https://mindboggle.info/data.html) (Klein & Tourville 2012, *Front. Neurosci.* 6:171): manually labeled T1-weighted brains with DKT31 cortical labels (`labels.DKT31.manual.nii.gz`), from the cohorts OASIS-TRT-20, NKI-RS-22, NKI-TRT-20 and MMRR-21.

- **Pairs:** 90 pairs (40 intra-subject and 50 inter-subject), defined in `antstorch/benchmark/pairs.csv`.
- **Pre-processing:** brain-extracted volumes, N4 bias correction (cached), images normalized as in syntx.
- **Location:** `ANTSTORCH_MINDBOGGLE_DATA_DIR` (default `~/data/mindboggle/volumes`; the script uses
  `DATA_DIR`). Directory layout: `antstorch/benchmark/data.py`.

#### Evaluation measures

| Measure | Definition |
|---|---|
| `dice_fixed` | Dice in the fixed space  |
| `dice_moving` | Dice in the moving space (if inverse exists) |
| `dice_sym` | mean of the two |
| time | deformable-only in second |
| Jacobian | min, max, mean, and σ(log J). |
| folding % | voxels with det ≤ 0 | 


### Results

> **Not yet filled in.** This section will hold only complete 90-pair runs. Numbers from earlier non-fair or
> partial runs are in the Reference part and are not comparable with it.

| Item | Source of the numbers |
|---|---|
| Dice (`dice_fixed`, `dice_sym`), paired difference ± s.e. | `comparison_summary.md`, aggregate table |
| Time (deformable-only), per device: MPS, CUDA, CPU | `comparison_summary.md`, time column |
| Jacobian (folding %, min/max J, σ(log J)) | `comparison_summary.md`, section "Jacobien du déplacement direct" |

Template to fill, one table per device:

| Model | Library | Dice fixed | Dice sym | Time (s) | Folding % | min J | σ(log J) |
|---|---|---|---|---|---|---|---|
| gaussian | syntx / ANTsTorch | | | | | | |
| sobolev | syntx / ANTsTorch | | | | | | |
| dsti | syntx / ANTsTorch | | | | | | |
| greedy | syntx / ANTsTorch / FireANTs | | | | | | |

## 7. Reproducing the runs

```
# Required: syntx and ANTsTorch must be the checkouts you mean to test, not site-packages copies
export PYTHONPATH=$HOME/Pkg/syntx/src:$HOME/Pkg/ANTsTorch
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --pairs 0 1 2 --out-dir <dir>          # validation
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --out-dir <dir>                        # 90 pairs, fair mode
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --models greedy greedy_regadam --out-dir <dir>
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --models gaussian sobolev dsti bspline_svf gaussian_svf --out-dir <dir>   # SVF: ANTsTorch only
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --fireants --device cuda:0 --out-dir <dir>
```

---

## Primitive comparisons: N4, denoising, cortical thickness, B-spline fitting

Each primitive benchmark script one runs an ANTsPy routine and its ANTsTorch counterpart on the same input and reports run time and agreement of the outputs. They are launched together by `run_primitive_benchmarks.py` (`--device` defaults to `antstorch.get_default_device()`; each comparison writes its own `log.txt`, and the run writes `run_info.txt` and `summary.txt`):

```bash
python tools/benchmarks/run_primitive_benchmarks.py --out-dir <dir>                   # default device
python tools/benchmarks/run_primitive_benchmarks.py --device cuda:0 --out-dir <dir>
python tools/benchmarks/run_primitive_benchmarks.py --only n4_bias_field_correction --device mps
```

| | MPS | CUDA |
|---|---|---|
| Host | Mac (Apple M4, 24 GB) | `cuda:0` = NVIDIA RTX A6000 (49 GB; a second A6000 was present but unused) |
| Python / ANTsTorch commit | 3.11.15 / `73ee932` | 3.11.9 / `2769e40` |
| Input | `S_template3` (136×176×176, ~1.5 mm) for denoising, N4 and cortical thickness; synthetic data for B-spline fitting | same |

How to read the numbers:

- ANTs always runs on the CPU of the host it ran on, and the two hosts differ (ANTs denoising. Speed ratios are therefore meaningful within a column, not between MPS and CUDA.
- ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2``.
- MPS has no float64. `compare_bspline_scattered_data.py` runs ANTsTorch in float32 on MPS and in float64 elsewhere; N4, denoising and cortical thickness use the library's own dtype handling.
- Ratio = ANTs time / ANTsTorch time.

### Run time

| Comparison | Device | ANTs (s) | ANTsTorch (s) | Ratio |
|---|---|---:|---:|---:|
| Denoising (`denoise_image`, median of 3) | MPS | 5.687 | 7.127 | 0.80× |
| | CUDA | 11.421 | 1.110 | 10.29× |
| N4 bias-field correction (corrected image + bias field) | MPS | 8.086 | 5.089 | 1.59× |
| | CUDA | 25.174 | 1.201 | 20.96× |
| Cortical thickness (KellyKapowski, 45 iterations) | MPS | 283.756 | 211.614 | 1.34× |
| | CUDA | 607.950 | 17.418 | 34.90× |

- Cortical thickness: Deep Atropos is run once and shared by both engines; it is excluded from these times (60.7 s on MPS, 101.0 s on CUDA).
- Peak ANTsTorch CUDA memory: 3332 MiB (denoising), 552.5 MiB (N4).
- B-spline fitting is not in the table: the problems are tiny (ANTs 1–8 ms; ANTsTorch 8–400 ms), so its times measure start-up overhead rather than throughput.

### Agreement with ANTs

| Comparison | Metric | MPS | CUDA |
|---|---|---:|---:|
| Denoising | RMSE (% of intensity range) | 0.01306 (0.18 %) | 0.01306 (0.18 %) |
| | MAE | 0.00209 | 0.00209 |
| | Max \|diff\| | 1.7295 | 1.7295 |
| | Correlation, denoised image | 0.99997 | 0.99997 |
| | Correlation, removed noise | 0.99401 | 0.99401 |
| N4 | Corrected-image RMSE | 0.003338 | 0.000475 |
| | Scale-aligned corrected-image RMSE | 0.001652 | 0.000446 |
| | Normalized log-bias MAE | 0.001320 | 0.000570 |
| | Normalized log-bias correlation | 0.99998254 | 0.99999517 |
| Cortical thickness | MAE (mm) | 0.1537 | 0.1453 |
| | RMSE (mm) | 0.2033 | 0.1931 |
| | 95th percentile \|error\| (mm) | 0.3969 | 0.3733 |
| | Pearson correlation | 0.99634 | 0.99649 |
| | GM voxels compared | 140258 | 140245 |

Observations:

- **Denoising:** the two devices give the same agreement to the printed precision; the run-to-run difference is 0 for both ANTs and ANTsTorch. The maximum difference (1.73, for an intensity range of about 8.7) is far above the RMSE, so a few voxels disagree strongly. This has not been investigated.
- **N4:** the corrected-image RMSE is about 7× lower on CUDA than on MPS (2–4× lower for the scale-aligned RMSE and the log-bias MAE). The launcher passes `--no-stable-accumulation` in both runs.
- **Cortical thickness:** agreement is similar on both devices (MAE ≈ 0.15 mm, r ≈ 0.996). The GM voxel counts differ slightly because each run computes its own Deep Atropos segmentation.

### B-spline fitting on scattered data

Synthetic 2-D data, 4 fitting levels; RMSE against ANTsPy.

| Comparison | MPS (float32) | CUDA (float64) |
|---|---:|---:|
| `fit_bspline_object_to_scattered_data` (1920 points, mesh 4) | 3.45e-6 | 3.22e-6 |
| `fit_bspline_displacement_field`, scattered points, component 0 / 1 | 8.20e-5 / 7.87e-5 | 8.19e-5 / 7.85e-5 |
| `fit_bspline_displacement_field`, dense field, component 0 / 1 | 1.33e-6 / 6.68e-7 | 1.33e-6 / 6.68e-7 |

The correlation is ≥ 0.99999999 in every case. The float32 run on MPS is within about 7 % of the float64 run on CUDA, so the remaining difference from ANTs is not float32 round-off.

---

## Reference

### Model catalogue

This file documents, in one place, every model name accepted by the comparison scripts under
`tools/benchmarks/` and by `antstorch.benchmark.evaluate_mindboggle_pair()`, which they call. Each
entry points to the docstring that is the actual authority (this file is a navigation summary, not a
new source of truth — if they disagree, the code's docstring is always right).

### Scripts and what they compare

| Script | Compares | Models / `--models` |
|---|---|---|
| `compare_syntx_antstorch_mindboggle.py` | syntx vs ANTsTorch, dense SyN family (fair mode by default, see below) | `gaussian`, `sobolev`, `dsti` (default), `bspline`, `gaussian_regadam`, `sobolev_regadam`, `dsti_regadam`, `bspline_regadam`, `greedy`, `greedy_regadam` (syntx vs ANTsTorch), `bspline_svf`, `gaussian_svf` (ANTsTorch only); `--fireants` adds FireANTs |
| `compare_bspline_scattered_data.py` | ANTs vs ANTsTorch, B-spline fit on scattered data (not a registration model) | N/A — primitive comparison, no `--models` |
| `compare_n4_bias_field_correction.py` | ANTs vs ANTsTorch, N4 bias-field correction (not a registration model) | N/A — primitive comparison, no `--models` |
| `compare_denoise_image.py` | ANTs `denoise_image` vs `antstorch.denoise_image` (adaptive non-local means, Rician or Gaussian noise) | N/A — primitive comparison, no `--models`; `--repeats 2+` also reports ANTs run-to-run variability |
| `compare_cortical_thickness.py` | ANTs `kelly_kapowski` vs the ANTsTorch tensor DiReCT engine on a shared Deep Atropos segmentation (`S_template3` by default) | N/A — same GM/WM inputs and shared parameters; `--optimizer` and `--regularizer` select the ANTsTorch variant |
| `run_primitive_benchmarks.py` | Runs `compare_denoise_image.py`, `compare_n4_bias_field_correction.py`, `compare_cortical_thickness.py` and `compare_bspline_scattered_data.py` in sequence on one device (`--device`, default `antstorch.get_default_device()`) with a common `--out-dir`, an optional shared image (positional; B-spline uses synthetic data unless `--bspline-image`), one `log.txt` per comparison, `run_info.txt` and `summary.txt`; `--only`, `--repeats`, `--ants-threads`, `--dry-run` | N/A — launcher for the primitive comparisons |

### Native ANTsTorch models (`antstorch.benchmark.evaluate_mindboggle_pair(model=...)`)

Authoritative documentation: the docstring of `antstorch.benchmark.evaluate.evaluate_mindboggle_pair()`
(updated with every new arm) and, for the registration details themselves, the docstring of
`antstorch.syn.syn_registration()`.

### Dense SyN family (`antstorch.syn.syn_registration(type_of_transform="SyNOnly", regularizer=..., ...)`)

The four `*_syn` variants differ only in the fluid regularizer applied at each iteration; all share
the same per-pair canonical affine (fairness invariant, project §17/§18).

| Model name | Regularizer | Optimizer | Note |
|---|---|---|---|
| `gaussian_syn` | `gaussian` (separable Gaussian smoothing) | `gradient_descent` (default) | — |
| `sobolev_syn` | `sobolev` (spectral Green operator) | `gradient_descent` (default) | — |
| `dsti_syn` | `dsti` (DST-I Green operator, homogeneous Dirichlet) | `gradient_descent` (default) | Persistent gap vs syntx's `dsti` arm, documented in project §34/§36; in the fair mode it runs with `syn_reg_adam` |
| `bspline_syn` | `bspline` (ANTs/ITK `BSplineSyN`, per-level cubic B-spline fit) | `gradient_descent` (default) | Unofficial arm on the syntx side (see below) |
| `gaussian_regadam` | `gaussian` (same regularizer as `gaussian_syn`) | **`reg_adam`** | Tests the effect of the optimizer alone on `gaussian`, without touching `gaussian_syn` |
| `sobolev_regadam` | `sobolev` (same regularizer as `sobolev_syn`) | **`reg_adam`** | Same for `sobolev` |
| **`dsti_regadam`** | `dsti` (same regularizer as `dsti_syn`) | **`reg_adam`** | First `_regadam` arm created — tests whether the optimizer alone explains part of the `dsti` gap vs syntx, without touching `dsti_syn` (project §40) |
| `bspline_regadam` | `bspline` (same regularizer as `bspline_syn`) | **`reg_adam`** | Same for `bspline` (also an unofficial arm on the syntx side, see below) |

The four `*_regadam` arms above use a light port of the Adam-momentum optimizer of `syntx/greedy.py` (`syn_registration(..., optimizer="reg_adam")`, project §40/§41). `optimizer` and `regularizer` are two independent parameters of `syn_registration()` — `reg_adam` works with any of the four regularizers, not only `dsti` (where it was introduced first). No corresponding `*_syn` arm is modified — each `*_regadam` is a fully separate arm, preserving the reproducibility of any earlier run. Not yet run on real Mindboggle-101 data — only validated on the synthetic test set.

Note that this `reg_adam` is the `greedy.py` algorithm, which is **not** the `reg_adam` of `syntx.syn()`; the latter is available as `optimizer="syn_reg_adam"` (see the fair-comparison section below).

`gaussian_sigma_mode` / `conservative_smooth` (optional kwargs) reproduce the default conventions of `syntx.syn` for `gaussian` / `sobolev` / `dsti` (`*_syn` and `*_regadam` alike) instead of ANTsTorch's different ones — see the `syn_registration()` docstring for exactly what changes. `adam_betas` / `adam_eps` (optional kwargs, relevant only for the `*_regadam` arms) control the decay of the Adam moments.

### Stationary velocity field (SVF) family

| Model name | Backend | Note |
|---|---|---|
| `bspline_svf` (alias `svf`) | `antstorch.bspline_flows.bspline_svf_registration()` | B-spline parameterization of the SVF — shares the word "bspline" with `bspline_syn` but is a different transform family (SVF, not SyN) |
| `gaussian_svf` | `antstorch.bspline_flows.gaussian_svf_registration()` | Dense (voxel-wise) SVF with Gaussian update/total-field smoothing |

### Traditional ANTs baselines (non-ANTsTorch)

| Model name | Dispatch |
|---|---|
| `ants_syn_quick` (key of `_ANTS_TRADITIONAL_MODELS`) | Direct `ants.registration(type_of_transform="antsRegistrationSyNQuick[so]")`, on the same shared canonical affine |
| any **deformable-only** `ants.registration()` `type_of_transform` (e.g. `"antsRegistrationSyNQuick[so]"`, `"SyNOnly"`, `"antsRegistrationSyN[bo]"`) | Passed through as is, case-sensitive — a preset that is not deformable-only (e.g. `"...[s]"`) is explicitly rejected, because it would recompute its own affine and break the fairness invariant |

