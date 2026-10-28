# MPS primitive run, 2026-10-28

Apple M4, 24 GB, macOS 26.6.2; Python 3.11.15, PyTorch 2.14.0,
ANTsPy 0.6.4, NumPy 2.0.2. Image: S_template3, full-image mask where applicable.
Library commit: 202cd69ef97777e1acd9d93c9773b5ad93f12872.
The benchmark-only working changes add repetitions and timing summaries.
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2.

`run_info.txt` and `summary.txt` describe the N=5 denoising, N4 and B-spline run.
Each subdirectory contains its raw log and timing samples. First samples are
included. N4 times cover two full calls (corrected image and bias field).
Agreement metrics use the last repetition, not an average across repetitions.

Cortical thickness is the initial single full 45-iteration run, before the
repetition changes. Its comparison.json and summary.md are preserved. Shared
Deep Atropos preprocessing took 86.068 s and is excluded from engine times.
The original launcher used the same image, device and two ITK threads.
Its original logs were removed by concurrent repository cleanup; the surviving
JSON/Markdown were copied after completion. Image artifacts remain local under
/private/tmp/antstorch_primitives_20261028_initial/cortical_thickness (paths in
comparison.json reflect their original output directory).

CUDA was not available. CUDA numbers in the parent README are historical.
