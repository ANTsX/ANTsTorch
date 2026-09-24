# ANTsTorch documentation

ANTsTorch provides PyTorch implementations of ANTsX deep-learning
architectures and applications, together with differentiable image-processing
and registration tools.

## Getting started

- [Installation and project overview](../README.md#overview)
- [Available applications](../README.md#applications)
- [Self-contained ANTsX examples](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#antsxnet)

## Registration and image processing

- [SyN registration tutorial](antsx_tutorial_syn.md)
- [N4 and diffeomorphic B-spline registration tutorial](antsx_tutorial_bspline_flows.md)

## Normalizing flows

- [Hybrid LAMNR flow trainer](lamnr_hybrid_trainer.md)

## Validation and development

- [Model validation and ANTsPyNet parity](model_validation.md)
- [Development tools](../tools/README.md)
- [Application verification scripts](../tools/verify_applications/README.md)

The public Python API is exposed through the `antstorch` package. Application
functions are available from the top-level namespace, while registration and
benchmark functionality is organized under `antstorch.syn`,
`antstorch.bspline_flows`, and `antstorch.benchmark`.
