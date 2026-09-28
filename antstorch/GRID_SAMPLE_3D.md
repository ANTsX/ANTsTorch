# 3-D sampler selection and GPU backends

`_grid_sample_3d.grid_sample_3d` implements trilinear (`mode="bilinear"`) and
nearest sampling using Torch gathers and arithmetic. It supports zero, border,
and reflection padding and both `align_corners` conventions. It differentiates
with respect to image values and coordinates; nearest coordinate gradients
are zero. It implements PyTorch padding semantics, not ITK's half-voxel mask.

On MPS, the default is now `ANTSTORCH_MPS_GRID_SAMPLE=auto`:

1. Try the native sampler if a cached capability test confirms the required
   forward operation and, when gradients are needed, both input/grid derivatives.
2. Otherwise use the fused Metal backend for supported float32 inputs after
   checking its forward/backward capability.
3. If neither supports the call, retain the existing CPU fallback (or CPU
   relocation for registration capability probes).

The capability probe treats PyTorch's own MPS CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`) as unavailable rather than native: that
setting turns an unimplemented op into a `UserWarning` plus a silent CPU
run instead of a raised exception, which would otherwise be misread as
native support. The probe forces the variable off and promotes any
`UserWarning` raised during its own two-op check to an error, independent
of the caller's environment.
Explicit `native`, `metal`, and `torch` overrides remain available. The portable
Torch backend is never selected automatically. Invalid values raise an error.
Set overrides before starting Python because capability probes are cached.
CPU, CUDA and 2-D calls keep native sampling. CPU fallback is not allowed inside
registration capability probes, so a CPU result cannot masquerade as MPS support.

The shared path is connected to DiReCT, B-spline/Gaussian SVF and affine
registration, and SyN sampling/probes. Request `device="mps"` explicitly;
SyN's automatic device selection still conservatively avoids MPS. Other
unsupported operators can still limit a particular algorithm.

Example from the repository root, using existing benchmark inputs:

```bash
ANTSTORCH_MPS_GRID_SAMPLE=torch PYTHONPATH="$PWD" python \
  tools/benchmarks/compare_cortical_thickness.py \
  --reuse-inputs /path/to/saved_inputs --device mps --iterations 5 \
  --output-dir /path/to/portable_sampler_results --verbose
```

Tests compare output values and first derivatives with native CPU grid_sample,
including batches/channels, out-of-domain positions, singleton dimensions,
reflection and small chunks. A numerical gradcheck is included. A hardware
MPS forward/backward test is skipped when MPS is unavailable. Native 3-D MPS
sampling is not required by that test.

This is experimental, not a proven performance improvement. Processing output
points in chunks (default 65,536 per batch item) bounds forward temporaries;
autograd still retains data from all chunks, and output chunks are concatenated.
Large-volume memory, MPS gradient accumulation and nondeterminism, numerical
agreement over long optimizations, and GPU speed require hardware validation.
The intended workload uses finite float32 coordinates on MPS; this is not a
complete replacement for every dtype, exceptional input or native kernel.


## Fused Metal backend

Set `ANTSTORCH_MPS_GRID_SAMPLE=metal` to select the fused implementation in
`_grid_sample_3d_metal.py`. It uses `torch.mps.compile_shader` (available in the
PyTorch 2.7.0 installation tested) and requires float32 image/grid tensors on
MPS. The Metal source is embedded in the Python module, so packaging does not
require a separate shader asset. Shader compilation is cached per process.

A forward kernel computes each interpolated value directly; the first-order
backward kernel computes coordinate derivatives and atomically accumulates
image derivatives. Image-gradient accumulation can be nondeterministic and
raises when deterministic algorithms are enabled. Second-order gradients are
not supported. Coordinates should be finite; tensors use 32-bit indexing and
oversized inputs are rejected. Other dtypes are rejected rather than silently
converted. Scalar interpolation preserves PyTorch, not ITK, boundary rules.

The setting uses the same shared dispatch and capability probes as the portable
backend, including forward/backward checks for registration. The default auto policy can select Metal when native capabilities are missing;
use `native` to restore the previous native/CPU fallback policy.

Hardware validation on this Mac with PyTorch 2.7.0 compared all padding modes,
both align_corners conventions, nearest and trilinear interpolation, values,
first derivatives, singleton dimensions, and noncontiguous tensors with the
CPU reference. A small DiReCT CPU/MPS comparison and backend probe tests pass.
This does not yet establish accuracy or speed for every full registration.

An interpolation microbenchmark at shape D/H/W = 189/233/197, batch 1,
3 channels, border padding and align_corners=True gave these synchronized
medians over three runs, after warm-up:

| Backend | Seconds per interpolation |
|---|---:|
| CPU fallback including transfers | 0.217880 |
| Portable Torch MPS | 1.216559 |
| Fused Metal MPS | 0.002586 |

Metal versus CPU maximum absolute error was 2.38e-7 (RMSE 2.93e-8).
These timings are hardware-specific forward timings, not whole-registration
speedups. Reproduce with:

```bash
PYTHONPATH="$PWD" python tools/benchmarks/benchmark_grid_sample_3d.py \
  --channels 3 --repeats 3 --output /tmp/grid_sample_benchmark.json
```


### Full-volume DiReCT validation and synchronization

On the saved mprage_hippmapp3r inputs (five iterations), ANTs took 64.349 s.
With a completion barrier after each custom Metal dispatch, two repeated
ANTsTorch runs took 14.889 s and 14.444 s. Their thickness arrays were identical
and their iteration energies matched the previous portable-backend run.
Final GM metrics versus ANTs were MAE 0.0138495 mm, RMSE 0.0422008 mm,
and Pearson r 0.995532. These are single-volume, five-iteration observations,
not validation of every registration workflow or a 45-iteration run.

The initial asynchronous implementation showed intermittent energy and map
differences during repeated full-volume runs despite passing small operator
tests. A `torch.mps.synchronize()` after forward and backward dispatch keeps
shader arguments alive through completion and eliminated the discrepancy in
the repeated runs. Retain these barriers until a nonblocking replacement is
validated under allocator reuse; do not remove them solely for microbenchmark
speed. The interpolation-only table above was measured before this barrier
was introduced; the repeated full-volume timings include it.

The native shader API is described in the official PyTorch documentation:
https://docs.pytorch.org/docs/stable/generated/torch.mps.compile_shader.html


## Automatic-selection registration checks

On the tested PyTorch 2.7.0 Mac, CPU/MPS comparisons with **no backend override**
used a translated 16-cubed Gaussian volume, five iterations and MSE:

| Algorithm | Maximum absolute warped-image difference | RMSE |
|---|---:|---:|
| B-spline SVF | 2.38e-7 | 1.38e-8 |
| SyNOnly | 3.58e-7 | 1.66e-8 |

These small tests validate integration, not realistic registration throughput.
MPS was slower at this scale, including first-use overhead. They do not validate
all metrics, regularizers, affine initializations or large-volume memory use.
Continue to request `device="mps"` explicitly; this change selects the sampler,
not the top-level device. SyN's conservative device autodetection is unchanged.
