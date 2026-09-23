# Model validation and ANTsPyNet parity

The companion [ANTsPyNet–ANTsTorch Comparison repository](https://github.com/ntustison/ANTsPyNet-ANTsTorch-Comparison)
is the reference harness for numerical agreement between the two implementations.
It owns the comparison scripts, thresholds, test inputs, and saved parity evidence.
ANTsTorch's `tools/verify_applications/` scripts remain useful for application
smoke checks; they do not replace a paired numerical comparison.

In the comparison checkout, run:

```bash
python run_all.py --only brain_extraction
python build_catalog.py
```

`validation_catalog.md` and `validation_catalog.json` inventory saved evidence
for all comparison scripts, including missing reports (`NOT_RUN`). A recorded
execution error takes precedence over a legacy passing verdict. Existing
reports are preserved; generating the catalog does not rerun a model.

A parity `PASS` means agreement on particular inputs under the recorded
thresholds. It does not establish accuracy against annotated ground truth,
coverage of every modality or ensemble member, or robustness across subjects.
Inspect each report's output metrics, execution errors, and notes.

New runs also produce `run_metadata.json` with package versions, available
source commits, platform/device information, explicit call arguments, and image
fingerprints. Explicit weight/data files can be fingerprinted with repeatable
`--artifact ROLE=PATH` arguments. This manifest does not automatically discover
or prove which weights were loaded. Historical reports without provenance
cannot establish the behavior of the current installed packages.

The local evidence reviewed on 2026-09-23 contains 22 comparison scripts and
21 saved reports. After accounting for recorded errors, the inventory has
11 PASS, 8 WARN, 2 ERROR, and 1 NOT_RUN (`lung_extraction`). These are historical
observations, not a fresh validation of this ANTsTorch revision. In particular,
`lung_segmentation`'s stored PASS masks failed artery/airway subcases, and the
`quality_assessment` report records a dimensionality error. Both need targeted
reruns against identified versions before drawing conclusions about current code.

Next validation work should extend the existing harness with multiple subjects
and acquisitions, automatic tracking of loaded weights, and visual QC outputs.
Keep that evidence in the companion repository rather than duplicating its
thresholds or archived results here.
