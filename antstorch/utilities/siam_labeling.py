"""ANTsImage interface to the original SIAM v0.3 nnU-Net model.

SIAM: Valabregue et al., https://arxiv.org/abs/2605.02737.
Model source: https://github.com/romainVala/SIAM (Apache-2.0).
Published weights: https://zenodo.org/records/17733431 (CC BY 4.0).
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

import ants
import numpy as np
import torch

from .device_manager import get_default_device
from .get_antstorch_data import get_antstorch_cache_directory, get_antstorch_data


_MODEL_NAME = "pred_DS108_LcsfP_Ano"
# Keep the source-model identity so existing, equivalent Zenodo caches work.
_MODEL_MD5 = "0c3ca6d0730a3dcb924e079af04474c3"
_ARCHIVE_MD5 = "a5845a9dc617346dcc1cb17f520315a6"


def _model_folds(folder):
    """Return the folds in a complete, atomically installed model bundle."""
    marker = folder / "antstorch_model.json"
    if not marker.is_file():
        return None
    try:
        manifest = json.loads(marker.read_text())
        folds = manifest["folds"]
        if (manifest["archive_md5"] != _MODEL_MD5 or not folds
                or not all(isinstance(fold, int) and fold >= 0 for fold in folds)):
            return None
        files = [folder / "plans.json", folder / "dataset.json"]
        files += [folder / f"fold_{fold}" / "checkpoint_final.pth" for fold in folds]
        if all(path.is_file() and path.stat().st_size > 0 for path in files):
            return tuple(folds)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _extract_model(archive_path, destination):
    """Extract only the inference files, never archive paths or links."""
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != _MODEL_NAME:
                continue
            relative = "/".join(parts[1:])
            wanted = relative in ("plans.json", "dataset.json") or re.fullmatch(
                r"fold_\d+/checkpoint_final\.pth", relative)
            if not wanted:
                continue
            if not member.isfile():
                raise ValueError("SIAM archive contains a non-regular inference file.")
            target = destination.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)

    for filename in ("plans.json", "dataset.json"):
        with (destination / filename).open() as stream:
            json.load(stream)
    folds = sorted(int(path.parent.name[5:]) for path in
                   destination.glob("fold_*/checkpoint_final.pth") if path.stat().st_size)
    if not folds:
        raise ValueError("SIAM archive contains no model checkpoints.")
    (destination / "antstorch_model.json").write_text(json.dumps(
        {"archive_md5": _MODEL_MD5, "folds": folds}))


def _get_siam_model(verbose=False):
    from filelock import FileLock

    cache = Path(get_antstorch_cache_directory()) / "siam" / "v0.3"
    cache.mkdir(parents=True, exist_ok=True)
    model_folder = cache / _MODEL_NAME
    with FileLock(str(cache / "download.lock")):
        folds = _model_folds(model_folder)
        if folds is not None:
            return model_folder, folds
        archive_path = Path(get_antstorch_data("siamV03", show_progress=verbose))
        checksum = hashlib.md5()
        with archive_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != _ARCHIVE_MD5:
            raise ValueError("SIAM archive checksum does not match the Figshare package.")
        # Extraction stays private until the model structure passes validation.
        with tempfile.TemporaryDirectory(prefix=".install-", dir=cache) as temporary:
            temporary = Path(temporary)
            extracted = temporary / "model"
            extracted.mkdir()
            _extract_model(archive_path, extracted)
            if model_folder.exists():
                shutil.rmtree(model_folder)
            extracted.rename(model_folder)
        return model_folder, _model_folds(model_folder)


def _load_siam_dependencies():
    try:
        import nibabel
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import filelock
    except ImportError as error:
        raise ImportError(
            "SIAM requires optional dependencies. Install with "
            "`python -m pip install 'antstorch[siam]'` "
            "(or `python -m pip install -e '.[siam]'` from the checkout)."
        ) from error
    return nibabel, nnUNetPredictor


def siam_labeling(image, device=None, verbose=False):
    """Segment a whole-head 3-D image with the original SIAM v0.3 model.

    Uses SIAM's pretrained nnU-Net, all published folds, Gaussian sliding
    windows with a 0.5 step, and no test-time mirroring. The nnU-Net plans
    perform intensity normalization and resampling to 0.75 mm. No brain
    extraction, N4 correction, or template registration is performed.

    Parameters
    ----------
    image : ANTsImage
        Scalar 3-D head image (e.g., T1, T2, FLAIR or CT). Physical geometry
        must be correct. Canonical RAS orientation is applied internally.
    device : str, torch.device or None
        Inference device. None uses the ANTsTorch session default.
    verbose : bool
        Show download and inference progress.

    Returns
    -------
    dict
        ``segmentation_image`` and 18 ``probability_images`` in the input
        geometry, plus ``labels`` mapping names to label values. Labels:
        0 background, 1 GM, 2 WM, 3 CSF, 4 ventricular CSF, 5 cerebellar GM,
        6 thalamus, 7 pallidum, 8 putamen, 9 caudate, 10 accumbens,
        11 amygdala, 12 hippocampus, 13 dura, 14 vessels, 15 skull,
        16 head, 17 anomalies. Probabilities include background in value order.

    Notes
    -----
    Install the ``siam`` extra first. The 2.68 GB inference-only archive is
    downloaded from Figshare using get_antstorch_data("siamV03"), verified,
    cached and unpacked. Weights are CC BY 4.0. Cite Valabregue et al.,
    https://arxiv.org/abs/2605.02737 and https://zenodo.org/records/17733431.
    Inference and the returned probability maps require substantial memory.

    Examples
    --------
    >>> result = siam_labeling(image, device="cuda", verbose=True)
    """
    if image.dimension != 3 or image.has_components:
        raise ValueError("SIAM requires a scalar 3-D image.")
    array = image.numpy().astype(np.float32)
    if not np.isfinite(array).all() or np.ptp(array) == 0:
        raise ValueError("SIAM requires finite, non-constant image intensities.")
    nib, predictor_class = _load_siam_dependencies()
    device = get_default_device() if device is None else torch.device(device)
    model_folder, folds = _get_siam_model(verbose)

    # Match TorchIO ToCanonical, then SimpleITKIO's (C,Z,Y,X) convention.
    lps_to_ras = np.diag([-1., -1., 1., 1.])
    affine = np.eye(4)
    affine[:3, :3] = image.direction @ np.diag(image.spacing)
    affine[:3, 3] = image.origin
    native = nib.Nifti1Image(array, lps_to_ras @ affine)
    canonical = nib.as_closest_canonical(native)
    undo_orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(canonical.affine),
        nib.orientations.io_orientation(native.affine))
    data = np.ascontiguousarray(np.asarray(canonical.dataobj).transpose(2, 1, 0)[None])
    properties = {"spacing": tuple(nib.affines.voxel_sizes(canonical.affine)[::-1])}

    predictor = predictor_class(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=(device.type == "cuda"), device=device,
        verbose=verbose, verbose_preprocessing=verbose, allow_tqdm=verbose)
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=folds, checkpoint_name="checkpoint_final.pth")
    segmentation, probabilities = predictor.predict_single_npy_array(
        data, properties, save_or_return_probabilities=True)

    def to_native(array):
        array = nib.orientations.apply_orientation(array.transpose(2, 1, 0), undo_orientation)
        return ants.from_numpy_like(np.ascontiguousarray(array), image)

    labels = predictor.dataset_json["labels"]
    return {
        "segmentation_image": to_native(segmentation),
        "probability_images": [to_native(probability) for probability in probabilities],
        "labels": dict(sorted(labels.items(), key=lambda item: item[1])),
    }
