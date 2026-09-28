"""Diagnostic: report which grid_sample_3d backend antstorch selects on this
machine's installed PyTorch, across the mode/padding/align_corners/backward
combinations used by DiReCT, SyN and the B-spline/Gaussian SVF registration
frameworks.

Run on the machine with MPS + the PyTorch build to check (2.14.0 here):

    PYTHONPATH="$PWD" python tools/benchmarks/check_native_grid_sampler_3d.py

Does not change any runtime behavior; it only calls the same cached
capability probe (`antstorch._torch_compat._available`) that
ANTSTORCH_MPS_GRID_SAMPLE=auto already uses internally.
"""
import torch
import antstorch._torch_compat as compat

print(f"torch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit("No MPS device on this machine; nothing to probe.")

combos = [
    ("bilinear", "zeros", True, False),
    ("bilinear", "zeros", True, True),
    ("bilinear", "border", True, False),
    ("bilinear", "border", True, True),
    ("bilinear", "reflection", False, False),
    ("bilinear", "reflection", False, True),
    ("nearest", "zeros", True, False),
    ("nearest", "zeros", True, True),
]

header = f"{'mode':<10}{'padding':<12}{'align_corners':<15}{'backward':<10}{'native':<8}{'metal':<8}selected"
print(header)
print("-" * len(header))
for mode, padding, align_corners, backward in combos:
    dtype = torch.float32
    native_ok = compat._available("native", dtype, mode, padding, align_corners, backward)
    metal_ok = compat._available("metal", dtype, mode, padding, align_corners, backward)
    selected = "native" if native_ok else ("metal" if metal_ok else "NONE (torch/cpu fallback)")
    print(f"{mode:<10}{padding:<12}{str(align_corners):<15}{str(backward):<10}"
          f"{str(native_ok):<8}{str(metal_ok):<8}{selected}")
