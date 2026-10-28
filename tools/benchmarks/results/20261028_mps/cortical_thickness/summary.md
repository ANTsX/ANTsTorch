# Cortical-thickness comparison

Input: `/Users/ntustison/.antstorch/S_template3.nii.gz`

Both engines used the same Deep Atropos segmentation and tissue probabilities.

| Measurement | Value |
|---|---:|
| Deep Atropos | 86.068 s |
| ANTs KellyKapowski | 304.300 s |
| ANTsTorch DiReCT | 221.616 s |
| ANTsTorch speed ratio | 1.373x |
| GM voxels compared | 141747 |
| GM MAE | 0.137702 mm |
| GM RMSE | 0.189079 mm |
| GM 95th percentile absolute error | 0.357723 mm |
| GM Pearson correlation | 0.995861 |

The speed ratio is `ANTs time / ANTsTorch time`; values above 1 favor ANTsTorch.
