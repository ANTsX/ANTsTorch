
import os
from urllib.parse import urlparse
from typing import Optional
from torchvision.datasets.utils import download_url

_antstorch_cache_directory = os.path.join(os.path.expanduser('~'), '.antstorch')


def get_antstorch_cache_directory():
    """Get the cache directory for ANTsTorch data. Data and pre-trained models will be
    downloaded here.

    The default directory is ~/.antstorch.

    Returns
    -------
    antstorch_cache_directory string
        The directory to store ANTsTorch data.

    """
    return(_antstorch_cache_directory)


def set_antstorch_cache_directory(antstorch_cache_dir: str):
    """Set the cache directory for ANTsTorch data. Data and pre-trained models will be
    downloaded here.

    Arguments
    ---------
    antstorch_cache_dir string
        The directory to store ANTsTorch data. It will be created if it does not exist.
    """
    global _antstorch_cache_directory
    _antstorch_cache_directory = os.path.abspath(antstorch_cache_dir)

    if not os.path.exists(_antstorch_cache_directory):
        os.makedirs(_antstorch_cache_directory)


def get_antstorch_data(
    file_id: Optional[str] = None,
    target_file_name: Optional[str] = None,
):
    """
    Download ANTsTorch data (templates, priors, etc.) from the Figshare repo, or from the local cache.

    Parameters
    ----------
    file_id : str
        One of the permitted file ids or pass "show" to list all valid possibilities.
        Note that most require internet access to download.
    target_file_name : str, optional
        Optional target filename. If omitted, a sensible default is inferred from `file_id`.

    Returns
    -------
    str
        Absolute path to the downloaded file (or existing cached file).

    Example
    -------
    >>> template_file = get_antstorch_data('kirby')
    """

    if file_id is None:
        raise ValueError("Missing file id. Pass one of the valid ids or 'show' to list them.")

    # Mirror the ANTsXNet coverage.
    _switcher = {
        "biobank": "https://ndownloader.figshare.com/files/22429242",
        "croppedMni152": "https://ndownloader.figshare.com/files/22933754",
        "croppedMni152Priors": "https://ndownloader.figshare.com/files/27688437",
        "deepFlashPriors": "https://ndownloader.figshare.com/files/31208272",
        "deepFlashTemplateT1": "https://ndownloader.figshare.com/files/31207795",
        "deepFlashTemplateT1SkullStripped": "https://ndownloader.figshare.com/files/31339867",
        "deepFlashTemplateT2": "https://ndownloader.figshare.com/files/31207798",
        "deepFlashTemplateT2SkullStripped": "https://ndownloader.figshare.com/files/31339870",
        "deepFlashTemplate2T1SkullStripped": "https://ndownloader.figshare.com/files/46461451",
        "deepFlashTemplate2Labels": "https://ndownloader.figshare.com/files/46461415",
        "mprage_hippmapp3r": "https://ndownloader.figshare.com/files/24984689",
        "protonLobePriors": "https://ndownloader.figshare.com/files/30678452",
        "protonLungTemplate": "https://ndownloader.figshare.com/files/22707338",
        "ctLungTemplate": "https://ndownloader.figshare.com/files/22707335",
        "luna16LungPriors": "https://ndownloader.figshare.com/files/28253796",
        "xrayLungPriors": "https://ndownloader.figshare.com/files/41965815",
        "priorDktLabels": "https://ndownloader.figshare.com/files/24139802",
        "S_template3": "https://ndownloader.figshare.com/files/22597175",
        "priorDeepFlashLeftLabels": "https://ndownloader.figshare.com/files/25422098",
        "priorDeepFlashRightLabels": "https://ndownloader.figshare.com/files/25422101",
        "adni": "https://ndownloader.figshare.com/files/25516361",
        "ixi": "https://ndownloader.figshare.com/files/25516358",
        "kirby": "https://ndownloader.figshare.com/files/25620107",
        "mni152": "https://ndownloader.figshare.com/files/25516349",
        "nki": "https://ndownloader.figshare.com/files/25516355",
        "nki10": "https://ndownloader.figshare.com/files/25516346",
        "oasis": "https://ndownloader.figshare.com/files/25516352",
        "magetTemplate": "https://ndownloader.figshare.com/files/41052572",
        "magetTemplateBrainMask": "https://ndownloader.figshare.com/files/41052569",
        "magetCerebellumTemplate": "https://ndownloader.figshare.com/files/41052581",
        "magetCerebellumTemplatePriors": "https://ndownloader.figshare.com/files/41052578",
        "magetCerebellumxTemplate0GenericAffine": "https://ndownloader.figshare.com/files/41052575",
        "mraTemplate": "https://ndownloader.figshare.com/files/46406695",
        "mraTemplateBrainMask": "https://ndownloader.figshare.com/files/46406698",
        "mraTemplateVesselPrior": "https://ndownloader.figshare.com/files/46406713",
        "hcpaT1Template": "https://ndownloader.figshare.com/files/54248318",
        "hcpaT2Template": "https://ndownloader.figshare.com/files/54248324",
        "hcpaFATemplate": "https://ndownloader.figshare.com/files/54248321",
        "hcpyaT1Template": "https://ndownloader.figshare.com/files/46746142",
        "hcpyaT2Template": "https://ndownloader.figshare.com/files/46746334",
        "hcpyaFATemplate": "https://ndownloader.figshare.com/files/46746349",
        "hcpyaTemplateBrainMask": "https://ndownloader.figshare.com/files/46746388",
        "hcpyaTemplateBrainSegmentation": "https://ndownloader.figshare.com/files/46746367",
        "hcpinterT1Template": "https://ndownloader.figshare.com/files/49372855",
        "hcpinterT2Template": "https://ndownloader.figshare.com/files/49372849",
        "hcpinterFATemplate": "https://ndownloader.figshare.com/files/49372858",
        "hcpinterTemplateBrainMask": "https://ndownloader.figshare.com/files/49372861",
        "hcpinterTemplateBrainSegmentation": "https://ndownloader.figshare.com/files/49372852",
        "bsplineT2MouseTemplate": "https://ndownloader.figshare.com/files/44706247",
        "bsplineT2MouseTemplateBrainMask": "https://ndownloader.figshare.com/files/44869285",
        "DevCCF_P56_MRI-T2_50um": "https://ndownloader.figshare.com/files/44706244",
        "DevCCF_P56_MRI-T2_50um_BrainParcellationNickMask": "https://ndownloader.figshare.com/files/44706238",
        "DevCCF_P56_MRI-T2_50um_BrainParcellationTctMask": "https://ndownloader.figshare.com/files/47214532",
        "DevCCF_P04_STPT_50um": "https://ndownloader.figshare.com/files/46711546",
        "DevCCF_P04_STPT_50um_BrainParcellationJayMask": "https://ndownloader.figshare.com/files/46712656",
    }

    valid_list = tuple(sorted(list(_switcher.keys()) + ["show"]))

    if file_id == "show":
        return valid_list

    if file_id not in _switcher:
        raise ValueError('No data with the id you passed - try "show" to get list of valid ids.')

    # get the cache directory - create if it does not exist
    antstorch_cache_directory = get_antstorch_cache_directory()

    if not os.path.exists(antstorch_cache_directory):
        os.makedirs(antstorch_cache_directory, exist_ok=True)

    url = _switcher[file_id]

    # Choose a default extension: .mat for the affine, .nii.gz otherwise.
    if target_file_name is None:
        if file_id == "magetCerebellumxTemplate0GenericAffine":
            target_file_name = f"{file_id}.mat"
        else:
            target_file_name = f"{file_id}.nii.gz"

    # Resolve destination path
    target_file_name_path = os.path.join(antstorch_cache_directory, target_file_name)

    # Download only if needed
    if not os.path.exists(target_file_name_path):
        # torchvision's download_url will skip if the file already exists in the directory
        download_url(url, antstorch_cache_directory, target_file_name, md5=None)

    return target_file_name_path
