import torchvision
import os

def get_pretrained_network(file_id=None,
                           target_file_name=None,
                           antstorch_cache_directory=None):

    """
    Download (or resolve cached) pretrained network/weights.


    Arguments
    ---------
    file_id : str
    One of the permitted ids (see `show`) or any custom id ending in
    `_pytorch` for cache‑only use. Pass "show" to list known ids.


    target_file_name : str, optional
    Target filename. If omitted, defaults to `<file_id>.pt` for ids ending
    in `_pytorch`, otherwise `<file_id>.h5`.


    antstorch_cache_directory : str, optional
    Cache directory (defaults to `~/.antstorch/`).


    Returns
    -------
    str : absolute filename to the pretrained weights

    Example
    -------
    >>> model_file = get_pretrained_network('not_yet')
    """

    def switch_networks(argument):
        switcher = {
            "chexnet_repro_pytorch": "https://figshare.com/ndownloader/files/42411897",
            "mriModalityClassification": "https://figshare.com/ndownloader/files/41692998",
            "brainExtractionRobustT1_pytorch": "https://figshare.com/ndownloader/files/58439353",
            "brainExtractionBrainWeb20_pytorch" : "https://figshare.com/ndownloader/files/58438324",
            "brainExtractionRobustT2_pytorch": "https://figshare.com/ndownloader/files/58439389",
            "brainExtractionRobustT2Star_pytorch": "https://figshare.com/ndownloader/files/58439458",
            "brainExtractionRobustFLAIR_pytorch": "https://figshare.com/ndownloader/files/58439521",
            "brainExtractionRobustBOLD_pytorch": "https://figshare.com/ndownloader/files/58436692",
            "brainExtractionMra_pytorch": "https://figshare.com/ndownloader/files/58439560",
            "brainExtractionRobustFA_pytorch": "https://figshare.com/ndownloader/files/58436695",
            "brainExtractionT1Hemi_pytorch" : "https://figshare.com/ndownloader/files/58439605",
            "brainExtractionT1Lobes_pytorch" : "https://figshare.com/ndownloader/files/58439629",
            "DeepAtroposHcpT1Weights_pytorch" : "https://figshare.com/ndownloader/files/58468954",
            "DeepAtroposHcpT1T2Weights_pytorch" : "https://figshare.com/ndownloader/files/58468960",
            "DeepAtroposHcpT1FAWeights_pytorch" : "https://figshare.com/ndownloader/files/58469074",
            "DeepAtroposHcpT1T2FAWeights_pytorch" : "https://figshare.com/ndownloader/files/58469134",
            "deepFlashLeftT1Hierarchical_pytorch" : "https://figshare.com/ndownloader/files/58488766",
            "deepFlashRightT1Hierarchical_pytorch" : "https://figshare.com/ndownloader/files/58488796",
            "deepFlashLeftBothHierarchical_pytorch" : "https://figshare.com/ndownloader/files/58488715",
            "deepFlashRightBothHierarchical_pytorch" : "https://figshare.com/ndownloader/files/58488772",
            "deepFlashLeftT1Hierarchical_ri_pytorch" : "https://figshare.com/ndownloader/files/58488769",
            "deepFlashRightT1Hierarchical_ri_pytorch" : "https://figshare.com/ndownloader/files/58488805",
            "deepFlashLeftBothHierarchical_ri_pytorch" : "https://figshare.com/ndownloader/files/58488760",
            "deepFlashRightBothHierarchical_ri_pytorch" : "https://figshare.com/ndownloader/files/58488778",
            "HarvardOxfordAtlasSubcortical_pytorch": "https://figshare.com/ndownloader/files/58488943"
        }
        return(switcher.get(argument, "Invalid argument."))

    if file_id == None:
        raise ValueError("Missing file id.")

    valid_list = ("chexnet_repro_pytorch",
                  "mriModalityClassification",
                  "brainExtractionRobustT1_pytorch",
                  "brainExtractionBrainWeb20_pytorch",
                  "brainExtractionRobustT1_pytorch",
                  "brainExtractionRobustT2_pytorch",
                  "brainExtractionRobustT2Star_pytorch",
                  "brainExtractionRobustFLAIR_pytorch",
                  "brainExtractionRobustBOLD_pytorch",
                  "brainExtractionMra_pytorch",
                  "brainExtractionRobustFA_pytorch",
                  "brainExtractionT1Hemi_pytorch",
                  "brainExtractionT1Lobes_pytorch",
                  "DeepAtroposHcpT1Weights_pytorch",
                  "DeepAtroposHcpT1T2Weights_pytorch",
                  "DeepAtroposHcpT1FAWeights_pytorch",
                  "DeepAtroposHcpT1T2FAWeights_pytorch",
                  "deepFlashLeftT1Hierarchical_pytorch",
                  "deepFlashRightT1Hierarchical_pytorch",
                  "deepFlashLeftBothHierarchical_pytorch",
                  "deepFlashRightBothHierarchical_pytorch",
                  "deepFlashLeftT1Hierarchical_ri_pytorch",
                  "deepFlashRightT1Hierarchical_ri_pytorch",
                  "deepFlashLeftBothHierarchical_ri_pytorch",
                  "deepFlashRightBothHierarchical_ri_pytorch",
                  "HarvardOxfordAtlasSubcortical_pytorch",
                  "show")

    if not file_id in valid_list:
        raise ValueError(("No data with the id you passed, ", file_id,   
                         ".  Try \"show\" to get list of valid ids."))

    if file_id == "show":
       return(valid_list)

    url = switch_networks(file_id)

    if target_file_name is None:        
        target_file_name = (
            f"{file_id}.pt" if file_id.endswith("_pytorch") else f"{file_id}.h5"
        )

    if antstorch_cache_directory is None:
        antstorch_cache_directory = os.path.join(os.path.expanduser("~"), ".antstorch/")

    os.makedirs(antstorch_cache_directory, exist_ok=True)
    target_file_name_path = os.path.join(antstorch_cache_directory, target_file_name)

    url = switch_networks(file_id)

    if url is not None:
        # We know where to download from
        if not os.path.exists(target_file_name_path):
            torchvision.datasets.utils.download_url(url, antstorch_cache_directory, target_file_name)
        return target_file_name_path

    # No URL mapping: allow cache‑only ids, but be explicit.
    if os.path.exists(target_file_name_path):
        return target_file_name_path

    raise ValueError(
        (f"No URL mapping for file_id='{file_id}', and not found in cache: \n"
        f" {target_file_name_path}\n"
        "Add a mapping in get_pretrained_network(), or place the file in the cache."
        )
    )

