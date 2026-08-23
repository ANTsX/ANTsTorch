# Verify applications (real data, real weights)

These are standalone scripts, **not** part of `tests/` (Nick asked to
keep them out -- they're too slow/heavy for the automated pytest suite:
real network downloads, real pretrained weights, real `ants`
preprocessing/registration). Run them by hand, individually or all at
once, whenever you want to confirm a ported application actually works
end-to-end -- as opposed to `tests/test_*_architectures.py`, which only
checks that the model builds and runs a forward pass on random data,
without touching real weights or real preprocessing.

## Usage

```bash
cd scripts/verify_applications
python verify_lung_extraction_proton.py        # run one script directly
python run_all.py                               # run all 30 and print a PASS/FAIL summary
python run_all.py "verify_lung_*.py"             # run a subset (shell glob, quoted)
python run_all.py --list                         # list discovered scripts
```

Each script prints what it downloaded/loaded, runs the real
`antstorch.<function>(...)` call with `verbose=True`, and prints a short
summary of the returned image(s). A non-zero exit code means it raised.

## Data sources

- The 5 white-matter-hyperintensity scripts reuse the real T1/FLAIR pair
  Nick supplied (figshare ids `40251796` / `40251793`), downloaded once
  via `_common.get_t1_flair_pair()` and cached under
  `~/.antstorch/verify_data/`.
- Lung and mouse scripts use ANTsTorch's own bundled ANTsXNet template
  data (`antstorch.get_antstorch_data(...)`) -- real images, correct
  modality, but often the *same* template the function aligns to
  internally, so the registration step is closer to identity than it
  would be for a genuinely unseen subject. These are smoke tests of the
  full real pipeline (preprocessing, real weights, inference,
  reconstruction), not validation benchmarks.
- A few modalities have no bundled real sample at all (lung ventilation
  MRI, chest x-ray, mouse ex5/histology acquisitions, mouse histology
  super-resolution RGB). Those scripts use a 2-D slice of the closest
  bundled volume as a **structural** stand-in -- correct shape/dtype so
  the pipeline runs, but not the intended modality. Each such script says
  so explicitly in its docstring.
- The 4 scripts added 2026-08-22 (`verify_hippmapp3r_segmentation.py`,
  `verify_hypothalamus_segmentation.py`, `verify_claustrum_segmentation.py`,
  `verify_quality_assessment.py`) reuse the same real T1/FLAIR pair as the
  white-matter-hyperintensity scripts (only T1 is needed for the first
  three). `verify_quality_assessment.py` was updated 2026-08-23 (see its
  docstring and the "Weights status" note below) to try real converted
  weights first, falling back to an untrained placeholder only if they
  haven't been converted yet.
- `verify_mri_super_resolution.py` (added 2026-08-22, after the SIQ DBPN
  architecture correction and `convert_mri_super_resolution_bespoke.py`)
  also reuses the same real T1. It runs with the function's defaults
  (`expansion_factor=(1,1,2)`, `feature="vgg"`) -- see its docstring for
  why this is the least-verified script in the folder (new architecture,
  new converter, never yet run against a real SIQ `.h5`).

## Weights status (as of 2026-08-23)

12 of the 30 scripts should work out of the box, using weights already
converted and delivered to `~/.antstorch/` earlier this session (11 from
2026-08-22, plus `sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl`
converted and uploaded 2026-08-22 evening). Most of the rest need weights
that are either not yet converted (run the relevant `tools/convert_*_bespoke.py`
locally -- see the project's gap-analysis doc for the exact commands) or
whose source `.h5` was never located in `~/.keras/ANTsXNet/` at all.
`verify_quality_assessment.py` no longer needs a special case: as of
2026-08-23 the real ResNet-50 architecture behind
`tidsQualityAssessment`/`koniqMS`/`koniqMS2`/`koniqMS3` was confirmed
(see `tools/convert_quality_assessment_bespoke.py`), so it now behaves
like any other `⏳` script for `koniqMS3` -- it just also has a documented
fallback (untrained placeholder model) if the weights aren't converted
yet, so it never hard-fails.

| Script | Needs | Status |
|---|---|---|
| `verify_sysu_media_wmh_segmentation.py` | `sysuMediaWmhFlairOnlyModel{0,1,2}_pytorch` | ⏳ run `convert_wmh_bespoke.py` |
| `verify_hypermapp3r_segmentation.py` | `hyperMapp3r_pytorch` | ✅ ready |
| `verify_wmh_segmentation.py` | `antsxnetWmhOr_pytorch` | ⏳ run `convert_wmh_bespoke.py` |
| `verify_shiva_pvs_segmentation.py` | `pvs_shiva_t1_flair_{0..4}_pytorch` | ⏳ run `convert_wmh_bespoke.py` |
| `verify_shiva_wmh_segmentation.py` | `wmh_shiva_t1_flair_{0..4}_pytorch` | ⏳ run `convert_wmh_bespoke.py` |
| `verify_lung_extraction_proton.py` | `protonLungMri_pytorch` | ✅ ready |
| `verify_lung_extraction_proton_lobes.py` | `protonLobes_pytorch` | ⏳ never converted |
| `verify_lung_extraction_mask_lobes.py` | `maskLobes_pytorch` | ⏳ never converted |
| `verify_lung_extraction_ct.py` | `lungCtWithPriorsSegmentationWeights_pytorch` | ⏳ never converted |
| `verify_lung_extraction_ventilation.py` | `wholeLungMaskFromVentilation_pytorch` | ✅ ready |
| `verify_lung_extraction_xray.py` | `xrayLungExtraction_pytorch` | ✅ ready |
| `verify_el_bicho.py` | `elBicho_pytorch` | ⏳ never converted |
| `verify_lung_pulmonary_artery_segmentation.py` | `pulmonaryArteryWeights_pytorch` + CT priors weights | ⏳ never converted |
| `verify_lung_airway_segmentation.py` | `pulmonaryAirwayWeights_pytorch` + CT priors weights | ⏳ never converted |
| `verify_mouse_brain_extraction_t2.py` | `mouseT2wBrainExtraction3D_pytorch` | ✅ ready |
| `verify_mouse_brain_extraction_ex5_coronal.py` | `ex5_coronal_weights_pytorch` | ✅ ready |
| `verify_mouse_brain_extraction_ex5_sagittal.py` | `ex5_sagittal_weights_pytorch` | ✅ ready |
| `verify_mouse_brain_parcellation_nick.py` | `mouseT2wBrainParcellation3DNick_pytorch` | ✅ ready |
| `verify_mouse_brain_parcellation_tct.py` | `mouseT2wBrainParcellation3DTct_pytorch` | ✅ ready |
| `verify_mouse_brain_parcellation_jay.py` | `mouseSTPTBrainParcellation3DJay_pytorch` | ⏳ source `.h5` not found |
| `verify_mouse_cortical_thickness.py` | (same as nick parcellation) | ✅ ready |
| `verify_mouse_histology_brain_mask.py` | `allen_brain_mask_weights_pytorch` | ✅ ready |
| `verify_mouse_histology_hemispherical_coronal_mask.py` | `allen_brain_leftright_coronal_mask_weights_pytorch` | ⏳ source `.h5` not found |
| `verify_mouse_histology_cerebellum_mask.py` | `allen_cerebellum_sagittal_mask_weights_pytorch` | ⏳ source `.h5` not found |
| `verify_mouse_histology_super_resolution.py` | `allen_sr_weights_pytorch` | ⏳ never converted (no DBPN converter written) |
| `verify_hippmapp3r_segmentation.py` | `hippMapp3rInitial_pytorch` + `hippMapp3rRefine_pytorch` | ⏳ never converted |
| `verify_hypothalamus_segmentation.py` | `hypothalamus_pytorch` | ⏳ never converted |
| `verify_claustrum_segmentation.py` | `claustrum_axial_0_pytorch` + `claustrum_coronal_0_pytorch` | ⏳ never converted |
| `verify_quality_assessment.py` | `koniqMS3_pytorch` (falls back to an untrained placeholder if absent) | ⏳ run `convert_quality_assessment_bespoke.py --only koniqMS3` -- **architecture confirmed 2026-08-23, never yet converted against the real `.h5`, see docstring** |
| `verify_mri_super_resolution.py` | `sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl_pytorch` | ⏳ run `convert_mri_super_resolution_bespoke.py` -- **never tested against a real SIQ `.h5`, see docstring** |

A script marked "⏳" will fail immediately at the weight-loading step with
a clear error from `get_pretrained_network` (no cached weights, no
download URL registered) -- that's expected and not a bug in the script;
it's exactly the gap the gap-analysis doc already tracks. Re-run it once
the corresponding weights exist in `~/.antstorch/`. `verify_quality_assessment.py`
is the one exception to "fails immediately": it catches that exact error
and falls back to an untrained placeholder model instead of failing, so a
passing run before conversion confirms only that the code path works --
not that the output means anything (see its docstring). After conversion
it uses the real weights and a passing run means what it says.
