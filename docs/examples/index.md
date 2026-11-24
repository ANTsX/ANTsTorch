# Example notebooks

The [notebook collection](../../examples/README.md) provides executable tutorials
adapted to ANTsTorch. Install with `python -m pip install -e '.[notebooks]'`,
register the environment with `python -m ipykernel install --user --name python3`,
and open the notebooks in Jupyter or VS Code using that kernel.

| Notebook | Data and outputs |
| --- | --- |
| [Getting started](../../examples/getting_started.ipynb) | Offline; image geometry and tensor conversion |
| [2-D U-Net](../../examples/unet_2d.ipynb) | Offline; probability maps from random weights |
| [N4 correction](../../examples/n4_bias_correction.ipynb) | Offline; synthetic bias correction and nonuniformity measurement |
| [SyN registration](../../examples/syn_registration.ipynb) | Downloads r16/r64; warped image, Jacobian and transforms |
| [Brain extraction](../../examples/brain_extraction.ipynb) | Downloads T1 and pretrained assets; probability, mask and brain image |
| [Deep Atropos](../../examples/deep_atropos.ipynb) | Downloads T1 and pretrained assets; segmentation and tissue probabilities |
| [Cortical thickness](../../examples/cortical_thickness.ipynb) | Downloads T1 and pretrained assets; tissue maps and DiReCT thickness |

All notebooks default to CPU. Pretrained applications can take several minutes
or longer and substantial memory. Set `t1_path` to an absolute path to use a
local scalar 3-D T1 image. Application outputs go to `example_outputs/` relative
to the kernel working directory and matching filenames are overwritten.

Notebooks are not executed during documentation builds. Pytest validates their
format and supports opt-in execution; see [testing instructions](../../tests/README.md).

```{toctree}
:maxdepth: 1

unet_2d
n4_synthetic
syn_registration
brain_extraction
```
