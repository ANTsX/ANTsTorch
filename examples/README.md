# ANTsTorch examples

Install the package with `python -m pip install -e .` from the repository root.
Run the scripts from that directory:

```bash
python examples/unet_2d.py
python examples/n4_synthetic.py
python examples/syn_registration.py --output-dir example_outputs/syn
python examples/brain_extraction.py /path/to/t1.nii.gz --output-dir example_outputs/brain
```

The first two examples use synthetic data without downloads. SyN downloads
ANTsPy sample images; brain extraction requires a local 3-D T1 image and downloads
pretrained weights/templates on first use. The latter scripts default to CPU,
accept `--device` and `--output-dir`, and overwrite matching output filenames.

See the [example guide](../docs/examples/index.md) for outputs and limitations,
and the [ANTsX tutorial](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621)
for additional workflows. These scripts are independent runnable demonstrations;
they do not run automatically when documentation is built.
