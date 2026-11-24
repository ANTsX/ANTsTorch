# ANTsTorch example notebooks

These Jupyter notebooks combine explanations, executable code, output checks and
plots. They adapt workflows from the [ANTsX tutorial](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621)
to ANTsTorch's current APIs.

From the repository root, install the tools and register the Python environment:

```bash
python -m pip install -e '.[notebooks]'
python -m ipykernel install --user --name python3 --display-name "Python 3 (ANTsTorch)"
```

Open a notebook in Jupyter or VS Code, select this kernel and run cells from top
to bottom. For JupyterLab, install it separately with `python -m pip install jupyterlab`
and launch `python -m jupyterlab examples`.

| Notebook | Workflow | Resources |
| --- | --- | --- |
| [Getting started](getting_started.ipynb) | ANTs images, physical geometry, NumPy and tensor axes | Synthetic; no downloads |
| [2-D U-Net](unet_2d.ipynb) | Forward pass and probability maps with random weights | Synthetic; no downloads; no trained segmentation |
| [N4 bias correction](n4_bias_correction.ipynb) | Correct a known bias field and measure nonuniformity | Synthetic; no downloads |
| [SyN registration](syn_registration.ipynb) | Align r64 to r16 and inspect the Jacobian | Downloads ANTsPy sample images |
| [Brain extraction](brain_extraction.ipynb) | Pretrained T1 extraction and mask inspection | Downloads sample T1, templates and weights |
| [Deep Atropos](deep_atropos.ipynb) | Six-tissue segmentation and tissue volumes | Downloads sample T1, templates and weights |
| [Cortical thickness](cortical_thickness.ipynb) | Deep Atropos followed by DiReCT | Downloads sample T1, templates and weights; most expensive workflow |

All notebooks default to CPU. Change `device` in the setup cell to use a supported
accelerator. The pretrained 3-D applications can require substantial memory and
several minutes or longer on CPU. They default to a downloadable T1; set `t1_path`
to an absolute path for your own scalar 3-D T1 MRI. Downloaded ANTsTorch assets are
cached under `~/.antstorch`.

Application outputs are written under `example_outputs/` relative to the kernel
working directory; rerunning overwrites matching filenames. Inspect plots and
results before using them downstream. Short demonstration schedules and synthetic
checks do not establish clinical or anatomical accuracy.

## Validate and execute

```bash
# Validate all notebook schemas without running cells.
python -m pytest tests/examples --test-device=cpu

# Execute the three offline notebooks in isolated kernels.
python -m pytest tests/examples --run-notebooks --test-device=cpu \
    -m "not requires_data and not slow and not cuda and not mps"

# Execute one pretrained application (downloads and potentially long runtime).
python -m pytest tests/examples --run-notebooks --test-device=cpu \
    -k brain_extraction --notebook-timeout=1800
```

The execution tests use temporary working directories and never save outputs
back into the source notebooks. See [testing details](../tests/README.md#example-notebooks).
