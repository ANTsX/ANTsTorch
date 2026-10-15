# Model catalogue — `tools/benchmarks/`

This file documents, in one place, every model name accepted by the comparison scripts under
`tools/benchmarks/` and by `antstorch.benchmark.evaluate_mindboggle_pair()`, which they call. Each
entry points to the docstring that is the actual authority (this file is a navigation summary, not a
new source of truth — if they disagree, the code's docstring is always right).

## Scripts and what they compare

| Script | Compares | Models / `--models` |
|---|---|---|
| `compare_syntx_antstorch_mindboggle.py` | syntx vs ANTsTorch, dense SyN family (fair mode by default, see below) | `gaussian`, `sobolev`, `dsti` (default), `bspline`, `gaussian_regadam`, `sobolev_regadam`, `dsti_regadam`, `bspline_regadam` |
| `compare_bspline_scattered_data.py` | ANTs vs ANTsTorch, B-spline fit on scattered data (not a registration model) | N/A — primitive comparison, no `--models` |
| `compare_n4_bias_field_correction.py` | ANTs vs ANTsTorch, N4 bias-field correction (not a registration model) | N/A — primitive comparison, no `--models` |
| `compare_denoise_image.py` | ANTs `denoise_image` vs `antstorch.denoise_image` (adaptive non-local means, Rician or Gaussian noise) | N/A — primitive comparison, no `--models`; `--repeats 2+` also reports ANTs run-to-run variability |
| `compare_cortical_thickness.py` | ANTs `kelly_kapowski` vs the ANTsTorch tensor DiReCT engine on a shared Deep Atropos segmentation (`S_template3` by default) | N/A — same GM/WM inputs and shared parameters; `--optimizer` and `--regularizer` select the ANTsTorch variant |
| `run_ants_comparisons.sh` | Runs `compare_denoise_image.py`, `compare_n4_bias_field_correction.py` and `compare_cortical_thickness.py` in sequence on one device (`DEVICE=cuda:0` by default) and, optionally, one shared image (`$1` or `IMAGE`), with one log per comparison | N/A — variables `IMAGE`, `DEVICE`, `OUT_DIR`, `PYTHON`, `REPEATS`, `ANTS_THREADS` |

## Native ANTsTorch models (`antstorch.benchmark.evaluate_mindboggle_pair(model=...)`)

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

The four `*_regadam` arms above use a light port of the Adam-momentum optimizer of
`syntx/greedy.py` (`syn_registration(..., optimizer="reg_adam")`, project §40/§41). `optimizer` and
`regularizer` are two independent parameters of `syn_registration()` — `reg_adam` works with any of
the four regularizers, not only `dsti` (where it was introduced first). No corresponding `*_syn` arm
is modified — each `*_regadam` is a fully separate arm, preserving the reproducibility of any earlier
run. Not yet run on real Mindboggle-101 data — only validated on the synthetic test set.

Note that this `reg_adam` is the `greedy.py` algorithm, which is **not** the `reg_adam` of
`syntx.syn()`; the latter is available as `optimizer="syn_reg_adam"` (see the fair-comparison section below).

`gaussian_sigma_mode` / `conservative_smooth` (optional kwargs) reproduce the default conventions of
`syntx.syn` for `gaussian` / `sobolev` / `dsti` (`*_syn` and `*_regadam` alike) instead of
ANTsTorch's different ones — see the `syn_registration()` docstring for exactly what changes.
`adam_betas` / `adam_eps` (optional kwargs, relevant only for the `*_regadam` arms) control the decay
of the Adam moments.

### Stationary velocity field family (SVF, not SyN)

| Model name | Backend | Note |
|---|---|---|
| `bspline_svf` (alias `svf`) | `antstorch.bspline_flows.bspline_svf_registration()` | B-spline parameterization of the SVF — shares the word "bspline" with `bspline_syn` but is a different transform family (SVF, not SyN) |
| `gaussian_svf` | `antstorch.bspline_flows.gaussian_svf_registration()` | Dense (voxel-wise) SVF with Gaussian update/total-field smoothing |

### Traditional ANTs baselines (non-ANTsTorch)

| Model name | Dispatch |
|---|---|
| `ants_syn_quick` (key of `_ANTS_TRADITIONAL_MODELS`) | Direct `ants.registration(type_of_transform="antsRegistrationSyNQuick[so]")`, on the same shared canonical affine |
| any **deformable-only** `ants.registration()` `type_of_transform` (e.g. `"antsRegistrationSyNQuick[so]"`, `"SyNOnly"`, `"antsRegistrationSyN[bo]"`) | Passed through as is, case-sensitive — a preset that is not deformable-only (e.g. `"...[s]"`) is explicitly rejected, because it would recompute its own affine and break the fairness invariant |

## Models on the syntx / FireANTs comparison side

Authoritative documentation: the docstring at the top of `compare_syntx_antstorch_mindboggle.py` and
of the same script (its `--fireants*` options).

| `--models` name | syntx arm | ANTsTorch arm | FireANTs arm (`--fireants` only) |
|---|---|---|---|
| `gaussian` | `syntx.benchmark.evaluate_mindboggle_pair(model="gaussian")` | `gaussian_syn` | `fireants.registration.greedy.GreedyRegistration` (the only supported regularizer) |
| `sobolev` | `model="sobolev"` | `sobolev_syn` | n/a |
| `dsti` | `model="syn_dsti1"` (syntx's own `dsti1` / `reg_adam` arm) | `dsti_syn` | n/a |
| `bspline` | **Unofficial** arm: `syntx.syn(regularizer="bspline", ...)` called directly, mimicking the `sobolev` branch of the syntx harness — syntx does not expose this model name in its own `evaluate_mindboggle_pair()` | `bspline_syn` | n/a |
| `gaussian_regadam` | `model="gaussian"` — **same syntx arm as `gaussian`**, no separate counterpart | `gaussian_regadam` | n/a |
| `sobolev_regadam` | `model="sobolev"` — **same syntx arm as `sobolev`** | `sobolev_regadam` | n/a |
| `dsti_regadam` | `model="syn_dsti1"` — **same syntx arm as `dsti`** | `dsti_regadam` | n/a |
| `bspline_regadam` | Same unofficial arm as `bspline` — **same syntx arm as `bspline`** | `bspline_regadam` | n/a |

None of the four `*_regadam` has a syntx counterpart of its own: each is compared with the same syntx
arm as its base counterpart (see `_base_reg()` in the script). Since 2026-10-02 the defaults of
`antstorch.syn.syn_registration()` follow syntx; since 2026-10-14 the bare command
`python compare_syntx_antstorch_mindboggle.py` runs the **fair mode** (models `gaussian sobolev dsti`).
Both are described in the next section. The `*_regadam` arms receive the schedule and formula
alignment of their base counterpart but keep `grad_step=0.5` and the `reg_adam` optimizer;
`--legacy-defaults` restores the previous ANTsTorch defaults (`syntx_defaults=False`) to reproduce
earlier runs. See project §34 for the details of that decision and §40/§41 for the `*_regadam` arms.

## Fair syntx / ANTsTorch / FireANTs comparison (added 2026-10-14)

This section describes the current behavior of `compare_syntx_antstorch_mindboggle.py` (FireANTs is an optional
third arm of that script since 2026-10-15) and supersedes any older statement elsewhere in this file when they disagree.
Hard rule followed throughout: only ANTsTorch is modified, never syntx.

### 1. ANTsTorch defaults now follow syntx

`antstorch.syn.syn_registration()` resolves every argument left at `None` from the table
`_SYNTX_DEFAULTS[(regularizer, optimizer)]` (`antstorch/syn/syn.py`), so the command line needs
almost no arguments. `syntx_defaults=False` (CLI: `--legacy-defaults`) restores the previous
ANTsTorch defaults (lncc, `grad_step` 0.5, `flow_sigma` 3.0, (100,100,50) iterations, physical
sigma, per-iteration inverses). Explicit arguments always win; `levels` and `reg_iterations` must be
given together.

| Model | `grad_step` | `flow_sigma` (as ANTsTorch sigma) | Iterations | Other |
|---|---|---|---|---|
| gaussian | 0.25 | sqrt(3) | (100,100,20) | cc2 (radius 2), sigma in voxels |
| sobolev | 0.4 | sqrt(2.4) | (100,100,20) | Sobolev alpha 2.25, conservative post-filter |
| dsti | 0.5 | sqrt(3) | (100,50,10) | alpha 1.0, conservative post-filter |
| bspline | 0.25 | 3.0 | (100,100,20) | |

Pyramid (4,2,1) everywhere. In syntx `flow_sigma` is an ITK *variance* (sigma = sqrt(flow_sigma));
ANTsTorch uses it as a sigma, hence the square roots. Verified bit-for-bit between the libraries
(relative difference 0.0): the Sobolev / DST-I Green operators for alpha in {none, 1.0, 2.25}, the
conservative DST-I formula (Green operator + sigma/2 Gaussian post-filter), and the Gaussian filter.
`total_sigma` is 0.0 in both.

### 2. Fair mode (default of the bare command)

`python tools/benchmarks/compare_syntx_antstorch_mindboggle.py` runs models `gaussian sobolev dsti`
in **fair mode**:

1. **Same affine for both libraries.** syntx runs first; its cached `.mat` is copied into
   ANTsTorch's canonical-affine cache (a warning is printed if the two affine Dice values differ by
   more than 1e-3). `--skip-syntx` or `--own-defaults` removes the sharing.
2. **Iso-algorithm ANTsTorch configuration:** `inverse_schedule="per_iteration"`,
   `in_loop_inverse_steps=10` (syntx passes 10 to SyNTo in *all* arms), `syntx_parity=True`
   (CFL scale, slope early stop, best-state restore, floor pyramid) and
   `use_analytical_gradients=False` (autograd, as every syntx benchmark arm).
3. **`dsti` uses `optimizer="syn_reg_adam"`** (see section 4).
4. **Times are compared deformable-only** on both sides (affine time reported separately).
5. One throw-away warm-up run per library on the first pair (device/kernel initialization), skipped
   with `--no-warmup`.
6. The Markdown summary starts with a paired aggregate table (mean difference +/- standard error,
   median, win rate, deformable-time ratio, median folding) followed by the per-pair table.

`--own-defaults` disables all of this (ANTsTorch defaults, end-of-level inverses, analytical
gradients, separate affines). `--matched` is a deprecated no-op.

### 3. Timing bias that was found and fixed

syntx's `runtime_seconds` includes its affine time (`syntx_affine_time`, about 6 s even when the
affine is cached), whereas ANTsTorch reported an affine time of 0.0 on a cache hit, so only the first
model of a pair included it. The earlier "1.8x to 2.8x faster" statement was therefore biased.
Corrected deformable-only time ratios (ANTsTorch / syntx, 90 pairs, CUDA): about 0.68 (upper bound)
for gaussian, 0.50 for sobolev and 0.41 for dsti. Fix: the affine time is stored in
`pair_NNN_affine_info.json`; records now carry `affine_seconds`, `deformable_seconds` and
`runtime_seconds` = their sum. Part of the remaining speed difference comes from syntx's two Anderson
inverses per iteration (10 steps each), its best-state `.clone()` calls and its `.item()` GPU
synchronizations.

### 4. `reg_adam` in ANTsTorch is not `reg_adam` in `syntx.syn()`

ANTsTorch's existing `optimizer="reg_adam"` ports the `greedy.py` path (Gaussian pre-smoothing sigma
1.8, Adam, Gaussian sigma 0.8). `syntx.syn()` (SyNTo) does something different: the gradient is first
regularized exactly as in the plain-CFL path, then a bias-corrected Adam quotient is formed (moments
reset every level), then the quotient goes through the Green operator *again* (no post-filter), then
one CFL scale shared by both half-warps (clamped to <= 1) times `optimizer_lr` is applied.
`optimizer="syn_reg_adam"` (with `optimizer_lr=1.0`) reproduces the second behavior; it is valid for
the `sobolev` and `dsti` regularizers only.

### 5. Results to date (older, non-fair runs) and what is still not identical

90 pairs, CUDA, paired difference of Dice sym (ANTsTorch - syntx): gaussian -0.0126 +/- 0.0078,
sobolev -0.0107 +/- 0.0075, dsti -0.0145 +/- 0.0076 (medians +0.005 / +0.004 / +0.002); without pair
63: -0.0058, -0.0038, -0.0077. CUDA, MPS and CPU agree within 0.0005 Dice for ANTsTorch. These runs
used separate affines and non-iso settings, so **they are not comparable with fair-mode runs**.

- **Pair 63:** ANTsTorch's own affine fails (Dice 0.001 vs 0.315 for syntx); cause not investigated.
  Fair mode uses syntx's affine, so it does not affect the SyN comparison there.
- **Pairs 59, 76, 68:** ANTsTorch SyN fails with heavy folding.
- **Folding:** median 0.0101 % (ANTsTorch gaussian) vs 0.0011 % (syntx); 45 pairs above 0.01 % vs 3.
- **Residual difference:** on a CPU harness (48x56x44, one level, 1 to 30 iterations) the final field
  differs from syntx by a relative 0.09 to 0.12 for gaussian/sobolev (cause unknown) and 0.25 to 0.39
  for dsti. The working hypothesis, not verified, is that Adam amplifies the CC-gradient difference
  where the gradient is small.
- **Not validated on real data:** fair mode has been exercised only with stubbed libraries so far.
  bspline and the `*_regadam` models are not part of the default fair-mode model list.

### 6. FireANTs (`--fireants` in `compare_syntx_antstorch_mindboggle.py`)

Since 2026-10-15 there is a single script. Without `--fireants` it behaves exactly as before; with it, FireANTs is added as a
third arm (the former `compare_fireants_mindboggle.py` has been removed):

- FireANTs runs **once per pair, after syntx and ANTsTorch**, and reads the same shared affine (syntx's;
  `_affine_source` is recorded and a warning printed otherwise).
- Time = construction of `GreedyRegistration` + `optimize()`, bracketed by `torch.cuda.synchronize()`;
  affine, NIfTI I/O, warp saving and scoring are outside the timer.
- **CUDA only:** any other device exits with a message.
- Inputs are normalized with `normalize_image('auto')` as in syntx (`--no-normalize` disables it).
- **Presets** (`--fireants-preset`, pyramid [4,2,1]; explicit `--fireants-*` options override single entries):

  | Preset | Iterations | warp σ | grad σ | lr | CC window | Meaning |
  |---|---|---|---|---|---|---|
  | `matched` (default) | [100,100,20] | 1.0 | 1.0 | 0.5 | 5 | same iteration budget as syntx / ANTsTorch: the fair-budget setting |
  | `default` | [200,100,25] | 0.5 | 1.0 | 0.5 | 7 | FireANTs' own defaults with its tutorial schedule |
  | `fine-tuned-syntx` | [100,100,50] | 0.35 | 2.0 | 0.4 | 5 | the "Tuned Greedy Compositive" setting of syntx's `docs/v_fire_ants` (reports 3.7% folding) |

  `fine-tuned-syntx` gives FireANTs 50 iterations at the last level (syntx has 20 here); add
  `--fireants-iterations 100 100 20` for an equal budget. Other overrides: `--fireants-lr`, `--fireants-warp-sigma`,
  `--fireants-grad-sigma`, `--fireants-cc-kernel`. (`--tutorial-schedule` was replaced by `--fireants-preset default`.)
- **Console output:** FireANTs' own progress bars and log warnings are silenced; each pair prints the same `-> SUCCESS dice_sym=… dice_fixed=… dice_moving=… deformable=…s` line as the syntx / ANTsTorch arms (plus iterations run). `--fireants-verbose` shows FireANTs' output again.
- **Early stop:** disabled by default, so every nominal iteration runs (equal budget with syntx and ANTsTorch, comparable times). `--fireants-early-stop` keeps FireANTs' own ConvergenceMonitor (tolerance 1e-6, 10 iterations); `--fireants-no-early-stop` is accepted but now redundant. Iterations actually run and first/last loss per level are recorded.
- FireANTs is a different algorithm (Adam, compositive warp, CC kernel 5): this is a library comparison, not a parity check.
- `dice_fixed` (forward warp only) is the reliable measure: `GreedyRegistration` has no native inverse, so the post-hoc
  inverse is poor and lowers `dice_sym`.
- **Folding caveat:** the settings that give FireANTs its best Dice fold the field (syntx's doc: 3.7% at the tuned
  preset). Compare Dice together with folding; with the default lr 0.5 FireANTs is under-trained.
- **Sweeps:** `--fireants-sweep --sweep-set {1,2,3}` runs a grid on the given pairs (needs syntx's cached affines in
  `<out-dir>/antstorch_canonical_affines` from an earlier run) and writes `fireants_sweep.md/.json`. Selection rule:
  highest `dice_fixed` with median folding below 0.01%; validate the chosen setting on other pairs to avoid tuning bias.
- Outputs: `comparison_results.json`, `comparison_summary.md` with a three-way paired aggregate and a per-pair table.
- Requires FireANTs on the machine that runs it.

### 6b. The `greedy` model: `syntx.greedy` vs `antstorch.greedy`

`antstorch.greedy` (`antstorch/greedy.py`) is a port of `syntx.greedy`: one-directional greedy compositive registration
(a single forward displacement field refined by `u <- v + u(x + v)`, Adam on the Gaussian-smoothed local-CC gradient,
`learning_rate` 0.375, `flow_sigma` 1.8, `total_sigma` 0.28, no inverse unless `return_inverse=True`). `--models greedy` (alone or after other models) runs two
arms per pair: `syntx.greedy` first, then `antstorch.greedy`, from the same shared affine (the one the dense syntx arms
already cached, else the one `syntx.greedy` fits itself and shares), both with `reg_iterations=[100, 100, 20]`, `return_inverse=True`, scored by the same antstorch metrics (`dice_fixed` from the
forward warp; `dice_moving` / `dice_sym` use the Anderson-computed inverse, so `dice_fixed` is the primary measure). The
time is the library's own `provenance['runtime_deformable_sec']` (optimisation only: no affine, export, inverse or scoring).
`greedy_regadam` runs the same two libraries with `optimizer='regadam'` (Adam quotient smoothed, `regadam_sigma` 0.8); unlike the dense models, both libraries have it, so it is paired like `greedy`. Only the Gaussian regularizer exists for greedy; neither is in the default `--models`.

`antstorch.greedy` ports the `syntx.greedy` that uses syntx's central 'deformable' pyramid (`syntx.pyramid.build_pyramid`, syntx >= 6.2: spacing-aware, millimetre-isotropic levels); it does not reproduce the older per-axis shrink of syntx < 6.2.

Parity (CPU, tests in `tests/greedy`, against syntx >= 6.2): without an affine the fields are identical bit for bit, also for anisotropic spacing, a moving image on its own grid, 2-D and a fixed mask; with the same affine file they
agree to 3.5e-5 voxel after 60 iterations and to ~0.03 voxel after 220, the residual coming from a 3e-7 rounding difference in
the two `parse_ants_affine` implementations (amplified by the iterations). On CUDA / MPS, `syntx` samples through its
`DeterministicGridSample` in the backward pass, `antstorch.greedy` through `F.grid_sample` (`antstorch._torch_compat`): expect
round-off-level differences, not algorithmic ones.

```
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --models greedy --pairs 0 24 88 --out-dir <dir>
```

### 7. Suggested runs

```
# Validation, 3 pairs (any device with syntx + ANTsTorch)
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --pairs 0 1 2 --out-dir <dir>
# Full fair comparison on the GPU machine (90 pairs)
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --out-dir <dir>
# Three-way comparison, GPU only
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --fireants --pairs 0 1 2 --out-dir <dir> --device cuda:0
# FireANTs hyper-parameter sweep (reuses syntx affines already in <dir>)
python tools/benchmarks/compare_syntx_antstorch_mindboggle.py --fireants-sweep --sweep-set 3 --pairs 0 1 2 --out-dir <dir>
```

Check the output for `!! affine Dice differs` or `!! FireANTs ... affine source` warnings. Old CPU/MPS/CUDA
runs do not need to be repeated for Dice, but their times are biased (section 3).

## History

The design decisions behind this catalogue (why a given arm exists, why another one stayed out of
scope) are documented in detail in the Claude project document "ANTsTorch registration framework"
(`claude/syntx-antstorch-integration-proposal.md`), in particular:
- §17-19: `_syn` naming convention, shared canonical-affine invariant.
- §34/§36: diagnosis of the `dsti_syn` vs `dsti` (syntx) gap, and its run-to-run variance.
- §37-39: integration of FireANTs as a third arm.
- §40: `RegAdam` / `dsti_regadam` port.
- §41: generalization of `RegAdam` to the three other regularizers (`gaussian_regadam`,
  `sobolev_regadam`, `bspline_regadam`).
- Fair-mode work (2026-10): project documents `claude/parite-syntx-antstorch-constats.md` and
  `claude/parite-syntx-antstorch-mode-juste.md`.
