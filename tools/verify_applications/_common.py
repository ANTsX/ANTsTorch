"""
Shared helpers for the ANTsTorch "verify applications" scripts.

These are NOT pytest tests (deliberately kept out of tests/, per Nick's
request -- they're too slow/heavy for the automated suite: real network
downloads, real pretrained weights, real ants preprocessing/registration).
They are standalone scripts you run by hand (or all at once via
run_all.py) to confirm a given application actually runs end-to-end on
real data with real converted weights.

Data sources used across the verify_*.py scripts (documented here so it's
in one place):

  * White matter hyperintensity functions (5 scripts) reuse the real T1 /
    FLAIR pair Nick supplied (figshare ids 40251796 / 40251793), downloaded
    once and cached locally.
  * Lung and mouse functions use ANTsTorch's own bundled ANTsXNet template
    data (antstorch.get_antstorch_data(...) -- protonLungTemplate,
    ctLungTemplate, DevCCF_P56_MRI-T2_50um, DevCCF_P04_STPT_50um, etc.).
    These are real, correctly-shaped, correctly-typed images for their
    respective modalities -- using a template as "the input image" is a
    smoke test (the registration/warp is closer to identity than it would
    be for a genuinely held-out subject), not a validation benchmark, but
    it exercises the complete real code path: preprocessing, model
    construction, real weight loading, inference, and reconstruction.
  * A handful of modalities have no bundled real sample at all (lung
    ventilation MRI, chest x-ray, mouse ex5/histology acquisitions, mouse
    histology super-resolution RGB). For those, a 2-D slice is extracted
    from the closest bundled volume as a *structural* stand-in -- correct
    shape/dtype/channel-count so the pipeline runs, but NOT the intended
    modality. Each such script says so explicitly in a comment.

Every script exposes a bare `main()` that returns the function's real
result (so you can `python verify_X.py` directly, or `import verify_X;
verify_X.main()` from a REPL). run_all.py drives all of them as
subprocesses and prints a PASS/FAIL summary.
"""

import os
import urllib.request

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".antstorch", "verify_data")
os.makedirs(CACHE_DIR, exist_ok=True)

# Same figshare files Nick's own sysu_media example downloads (via
# tf.keras.utils.get_file there; plain urllib here so these scripts don't
# need tensorflow, which antstorch has no dependency on).
T1_URL = "https://ndownloader.figshare.com/files/40251796"
FLAIR_URL = "https://ndownloader.figshare.com/files/40251793"


def download_file(url, fname):
    path = os.path.join(CACHE_DIR, fname)
    if not os.path.exists(path):
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, path)
    return path


def get_t1_flair_pair():
    """Returns (t1_path, flair_path) for the real T1/FLAIR pair used by
    every white_matter_hyperintensity_segmentation verify script."""
    t1_path = download_file(T1_URL, "t1.nii.gz")
    flair_path = download_file(FLAIR_URL, "flair.nii.gz")
    return t1_path, flair_path


def middle_slice(volume, axis=2):
    """Extract the middle 2-D slice of a 3-D ANTsImage along `axis` --
    used as a fast, real-data-derived stand-in wherever a function needs
    a 2-D image and no bundled 2-D sample exists for that modality."""
    import ants
    idx = volume.shape[axis] // 2
    return ants.slice_image(volume, axis=axis, idx=idx)


def to_fake_rgb(gray_2d):
    """Replicate a grayscale 2-D ANTsImage into a 3-component (RGB) image
    -- used only for mouse_histology_super_resolution, which requires a
    3-channel input and has no bundled color-histology sample."""
    import ants
    return ants.merge_channels([gray_2d, gray_2d, gray_2d])


def summarize(value, indent="  "):
    """Best-effort human-readable summary of an application's return
    value (ANTsImage / dict / list / other), for a quick sanity glance --
    not a correctness check."""
    import ants

    def _one(v):
        if isinstance(v, ants.core.ants_image.ANTsImage):
            comps = getattr(v, "components", 1)
            return f"ANTsImage shape={v.shape} components={comps}"
        return repr(type(v))

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, list):
                print(f"{indent}{k}: list[{len(v)}] = {[_one(x) for x in v]}")
            else:
                print(f"{indent}{k}: {_one(v)}")
    elif isinstance(value, list):
        print(f"{indent}list[{len(value)}] = {[_one(x) for x in value]}")
    else:
        print(f"{indent}{_one(value)}")
