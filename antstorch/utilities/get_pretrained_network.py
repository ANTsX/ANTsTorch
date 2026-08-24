import os

from torch.hub import download_url_to_file
from typing import Optional,Tuple

from .get_antstorch_data import get_antstorch_cache_directory

def get_pretrained_network(file_id: Optional[str] = None,
                           target_file_name: Optional[str] = None,
                           show_progress: bool=True) -> str|Tuple:

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

    show_progress : bool, optional
    Whether to show download progress. Default is True.

    Returns
    -------
    str : absolute filename to the pretrained weights

    Example
    -------
    >>> model_file = get_pretrained_network('not_yet')
    """

    def switch_networks(argument):
        switcher = {
            "chexnet_repro_pytorch": "https://ndownloader.figshare.com/files/42411897",
            "mriModalityClassification": "https://ndownloader.figshare.com/files/41692998",
            "brainExtractionRobustT1_pytorch": "https://ndownloader.figshare.com/files/58439353",
            "brainExtractionBrainWeb20_pytorch" : "https://ndownloader.figshare.com/files/58438324",
            "brainExtractionRobustT2_pytorch": "https://ndownloader.figshare.com/files/58439389",
            "brainExtractionRobustT2Star_pytorch": "https://ndownloader.figshare.com/files/58439458",
            "brainExtractionRobustFLAIR_pytorch": "https://ndownloader.figshare.com/files/58439521",
            "brainExtractionRobustBOLD_pytorch": "https://ndownloader.figshare.com/files/58436692",
            "brainExtractionMra_pytorch": "https://ndownloader.figshare.com/files/58439560",
            "brainExtractionRobustFA_pytorch": "https://ndownloader.figshare.com/files/58436695",
            "brainExtractionT1Hemi_pytorch" : "https://ndownloader.figshare.com/files/58439605",
            "brainExtractionT1Lobes_pytorch" : "https://ndownloader.figshare.com/files/58439629",
            "DeepAtroposHcpT1Weights_pytorch" : "https://ndownloader.figshare.com/files/58468954",
            "DeepAtroposHcpT1T2Weights_pytorch" : "https://ndownloader.figshare.com/files/58468960",
            "DeepAtroposHcpT1FAWeights_pytorch" : "https://ndownloader.figshare.com/files/58469074",
            "DeepAtroposHcpT1T2FAWeights_pytorch" : "https://ndownloader.figshare.com/files/58469134",
            "deepFlashLeftT1Hierarchical_pytorch" : "https://ndownloader.figshare.com/files/58488766",
            "deepFlashRightT1Hierarchical_pytorch" : "https://ndownloader.figshare.com/files/58488796",
            "deepFlashLeftBothHierarchical_pytorch" : "https://ndownloader.figshare.com/files/58488715",
            "deepFlashRightBothHierarchical_pytorch" : "https://ndownloader.figshare.com/files/58488772",
            "deepFlashLeftT1Hierarchical_ri_pytorch" : "https://ndownloader.figshare.com/files/58488769",
            "deepFlashRightT1Hierarchical_ri_pytorch" : "https://ndownloader.figshare.com/files/58488805",
            "deepFlashLeftBothHierarchical_ri_pytorch" : "https://ndownloader.figshare.com/files/58488760",
            "deepFlashRightBothHierarchical_ri_pytorch" : "https://ndownloader.figshare.com/files/58488778",
            "HarvardOxfordAtlasSubcortical_pytorch": "https://ndownloader.figshare.com/files/67802100",
            "DesikanKillianyTourvilleOuter_pytorch": "https://ndownloader.figshare.com/files/58494949",
            "cerebellumWhole_pytorch": "https://ndownloader.figshare.com/files/58495219",
            "cerebellumTissue_pytorch": "https://ndownloader.figshare.com/files/58495210",
            "cerebellumLabels_pytorch": "https://ndownloader.figshare.com/files/58495102",
            "deep_nbm_rank_pytorch": "https://ndownloader.figshare.com/files/64608699",
            "deepCIT168_pytorch": "https://ndownloader.figshare.com/files/64608723",
            "deepCIT168_sn_pytorch": "https://ndownloader.figshare.com/files/64608726",
            "resnet_grader_pytorch": "https://ndownloader.figshare.com/files/64896243",
            "protonLungMri_pytorch": "https://ndownloader.figshare.com/files/67757418",
            "protonLobes_pytorch": "https://ndownloader.figshare.com/files/67757424",
            "maskLobes_pytorch": "https://ndownloader.figshare.com/files/67757427",
            "lungCtWithPriorsSegmentationWeights_pytorch": "https://ndownloader.figshare.com/files/67757430",
            "wholeLungMaskFromVentilation_pytorch": "https://ndownloader.figshare.com/files/67757433",
            "xrayLungExtraction_pytorch": "https://ndownloader.figshare.com/files/67754508",
            "elBicho_pytorch": "https://ndownloader.figshare.com/files/67757439",
            "pulmonaryArteryWeights_pytorch": "https://ndownloader.figshare.com/files/67757448",
            "pulmonaryAirwayWeights_pytorch": "https://ndownloader.figshare.com/files/67757454",
            "mouseT2wBrainExtraction3D_pytorch": "https://ndownloader.figshare.com/files/67757457",
            "ex5_coronal_weights_pytorch": "https://ndownloader.figshare.com/files/67757460",
            "ex5_sagittal_weights_pytorch": "https://ndownloader.figshare.com/files/67757463",
            "mouseT2wBrainParcellation3DNick_pytorch": "https://ndownloader.figshare.com/files/67757466",
            "mouseT2wBrainParcellation3DTct_pytorch": "https://ndownloader.figshare.com/files/67757469",
            "mouseSTPTBrainParcellation3DJay_pytorch": "https://ndownloader.figshare.com/files/67757472",
            # "allen_brain_mask_weights_pytorch": "https://ndownloader.figshare.com/files/67757478",
            # "allen_brain_leftright_coronal_mask_weights_pytorch": "",
            # "allen_cerebellum_sagittal_mask_weights_pytorch": "",
            # "allen_cerebellum_coronal_mask_weights_pytorch": "",
            # "allen_sr_weights_pytorch": "",
            "sysuMediaWmhFlairOnlyModel0_pytorch": "https://ndownloader.figshare.com/files/67757481",
            "sysuMediaWmhFlairOnlyModel1_pytorch": "https://ndownloader.figshare.com/files/67757484",
            "sysuMediaWmhFlairOnlyModel2_pytorch": "https://ndownloader.figshare.com/files/67757493",
            "sysuMediaWmhFlairT1Model0_pytorch": "https://ndownloader.figshare.com/files/67757496",
            "sysuMediaWmhFlairT1Model1_pytorch": "https://ndownloader.figshare.com/files/67757502",
            "sysuMediaWmhFlairT1Model2_pytorch": "https://ndownloader.figshare.com/files/67757508",
            "hyperMapp3r_pytorch": "https://ndownloader.figshare.com/files/67755978",
            "antsxnetWmhOr_pytorch": "https://ndownloader.figshare.com/files/67757520",
            "antsxnetWmh_pytorch": "https://ndownloader.figshare.com/files/67757550",
            "pvs_shiva_t1_0_pytorch": "https://ndownloader.figshare.com/files/67757583",
            "pvs_shiva_t1_1_pytorch": "https://ndownloader.figshare.com/files/67757592",
            "pvs_shiva_t1_2_pytorch": "https://ndownloader.figshare.com/files/67757604",
            "pvs_shiva_t1_3_pytorch": "https://ndownloader.figshare.com/files/67757634",
            "pvs_shiva_t1_4_pytorch": "https://ndownloader.figshare.com/files/67757664",
            "pvs_shiva_t1_5_pytorch": "https://ndownloader.figshare.com/files/67757670",
            "pvs_shiva_t1_flair_0_pytorch": "https://ndownloader.figshare.com/files/67757676",
            "pvs_shiva_t1_flair_1_pytorch": "https://ndownloader.figshare.com/files/67757826",
            "pvs_shiva_t1_flair_2_pytorch": "https://ndownloader.figshare.com/files/67757841",
            "pvs_shiva_t1_flair_3_pytorch": "https://ndownloader.figshare.com/files/67757889",
            "pvs_shiva_t1_flair_4_pytorch": "https://ndownloader.figshare.com/files/67757940",
            "wmh_shiva_flair_0_pytorch": "https://ndownloader.figshare.com/files/67757943",
            "wmh_shiva_flair_1_pytorch": "https://ndownloader.figshare.com/files/67758108",
            "wmh_shiva_flair_2_pytorch": "https://ndownloader.figshare.com/files/67758813",
            "wmh_shiva_flair_3_pytorch": "https://ndownloader.figshare.com/files/67759557",
            "wmh_shiva_flair_4_pytorch": "https://ndownloader.figshare.com/files/67759596",
            "wmh_shiva_t1_flair_0_pytorch": "https://ndownloader.figshare.com/files/67759620",
            "wmh_shiva_t1_flair_1_pytorch": "https://ndownloader.figshare.com/files/67759698",
            "wmh_shiva_t1_flair_2_pytorch": "https://ndownloader.figshare.com/files/67759836",
            "wmh_shiva_t1_flair_3_pytorch": "https://ndownloader.figshare.com/files/67759845",
            "wmh_shiva_t1_flair_4_pytorch": "https://ndownloader.figshare.com/files/67759848",
            "sig_smallshort_train_1x1x2_1chan_featgraderL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761963",
            "sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761855",  
            "sig_smallshort_train_1x1x3_1chan_featgraderL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761966",
            "sig_smallshort_train_1x1x3_1chan_featvggL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761969", 
            "sig_smallshort_train_1x1x4_1chan_featgraderL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761972", 
            "sig_smallshort_train_1x1x4_1chan_featvggL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761975", 
            "sig_smallshort_train_1x1x6_1chan_featvggL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761978", 
            "sig_smallshort_train_2x2x2_1chan_featgraderL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761984", 
            "sig_smallshort_train_2x2x2_1chan_featvggL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761987",  # 
            "sig_smallshort_train_2x2x4_1chan_featgraderL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761990", 
            "sig_smallshort_train_2x2x4_1chan_featvggL6_best_mdl_pytorch": "https://ndownloader.figshare.com/files/67761993", 
            "hippMapp3rInitial_pytorch": "https://ndownloader.figshare.com/files/67761933",
            "hippMapp3rRefine_pytorch": "https://ndownloader.figshare.com/files/67761939",
            "hypothalamus_pytorch": "https://ndownloader.figshare.com/files/67761942",
            "claustrum_axial_0_pytorch": "https://ndownloader.figshare.com/files/67761945",
            "claustrum_axial_1_pytorch": "https://ndownloader.figshare.com/files/67761948",
            "claustrum_axial_2_pytorch": "https://ndownloader.figshare.com/files/67761951",
            "claustrum_coronal_0_pytorch": "https://ndownloader.figshare.com/files/67761954",
            "claustrum_coronal_1_pytorch": "https://ndownloader.figshare.com/files/67761957",
            "claustrum_coronal_2_pytorch": "https://ndownloader.figshare.com/files/67761960",
            "tidsQualityAssessment_pytorch": "https://ndownloader.figshare.com/files/67762350",
            "koniqMS_pytorch": "https://ndownloader.figshare.com/files/67762356",
            "koniqMS2_pytorch": "https://ndownloader.figshare.com/files/67762359",
            "koniqMS3_pytorch": "https://ndownloader.figshare.com/files/67762362",
        }
        return(switcher.get(argument, None))

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
                  "DesikanKillianyTourvilleOuter_pytorch",
                  "cerebellumWhole_pytorch",
                  "cerebellumTissue_pytorch",
                  "cerebellumLabels_pytorch",
                  "deep_nbm_rank_pytorch",
                  "deepCIT168_pytorch",
                  "deepCIT168_sn_pytorch",
                  "resnet_grader_pytorch",
                  "protonLungMri_pytorch",
                  "protonLobes_pytorch",
                  "maskLobes_pytorch",
                  "lungCtWithPriorsSegmentationWeights_pytorch",
                  "wholeLungMaskFromVentilation_pytorch",
                  "xrayLungExtraction_pytorch",
                  "elBicho_pytorch",
                  "pulmonaryArteryWeights_pytorch",
                  "pulmonaryAirwayWeights_pytorch",
                  "mouseT2wBrainExtraction3D_pytorch",
                  "ex5_coronal_weights_pytorch",
                  "ex5_sagittal_weights_pytorch",
                  "mouseT2wBrainParcellation3DNick_pytorch",
                  "mouseT2wBrainParcellation3DTct_pytorch",
                  "mouseSTPTBrainParcellation3DJay_pytorch",
                  # "allen_brain_mask_weights_pytorch",
                  # "allen_brain_leftright_coronal_mask_weights_pytorch",
                  # "allen_cerebellum_sagittal_mask_weights_pytorch",
                  # "allen_cerebellum_coronal_mask_weights_pytorch",
                  # "allen_sr_weights_pytorch",
                  "sysuMediaWmhFlairOnlyModel0_pytorch",
                  "sysuMediaWmhFlairOnlyModel1_pytorch",
                  "sysuMediaWmhFlairOnlyModel2_pytorch",
                  "sysuMediaWmhFlairT1Model0_pytorch",
                  "sysuMediaWmhFlairT1Model1_pytorch",
                  "sysuMediaWmhFlairT1Model2_pytorch",
                  "hyperMapp3r_pytorch",
                  "antsxnetWmhOr_pytorch",
                  "antsxnetWmh_pytorch",
                  "pvs_shiva_t1_0_pytorch",
                  "pvs_shiva_t1_1_pytorch",
                  "pvs_shiva_t1_2_pytorch",
                  "pvs_shiva_t1_3_pytorch",
                  "pvs_shiva_t1_4_pytorch",
                  "pvs_shiva_t1_5_pytorch",
                  "pvs_shiva_t1_flair_0_pytorch",
                  "pvs_shiva_t1_flair_1_pytorch",
                  "pvs_shiva_t1_flair_2_pytorch",
                  "pvs_shiva_t1_flair_3_pytorch",
                  "pvs_shiva_t1_flair_4_pytorch",
                  "wmh_shiva_flair_0_pytorch",
                  "wmh_shiva_flair_1_pytorch",
                  "wmh_shiva_flair_2_pytorch",
                  "wmh_shiva_flair_3_pytorch",
                  "wmh_shiva_flair_4_pytorch",
                  "wmh_shiva_t1_flair_0_pytorch",
                  "wmh_shiva_t1_flair_1_pytorch",
                  "wmh_shiva_t1_flair_2_pytorch",
                  "wmh_shiva_t1_flair_3_pytorch",
                  "wmh_shiva_t1_flair_4_pytorch",
                  "sig_smallshort_train_1x1x2_1chan_featgraderL6_best_mdl_pytorch",
                  "sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl_pytorch",
                  "sig_smallshort_train_1x1x3_1chan_featgraderL6_best_mdl_pytorch",
                  "sig_smallshort_train_1x1x3_1chan_featvggL6_best_mdl_pytorch",
                  "sig_smallshort_train_1x1x4_1chan_featgraderL6_best_mdl_pytorch",
                  "sig_smallshort_train_1x1x4_1chan_featvggL6_best_mdl_pytorch",
                  "sig_smallshort_train_1x1x6_1chan_featvggL6_best_mdl_pytorch",
                  "sig_smallshort_train_2x2x2_1chan_featgraderL6_best_mdl_pytorch",
                  "sig_smallshort_train_2x2x2_1chan_featvggL6_best_mdl_pytorch",
                  "sig_smallshort_train_2x2x4_1chan_featgraderL6_best_mdl_pytorch",
                  "sig_smallshort_train_2x2x4_1chan_featvggL6_best_mdl_pytorch",
                  "hippMapp3rInitial_pytorch",
                  "hippMapp3rRefine_pytorch",
                  "hypothalamus_pytorch",
                  "claustrum_axial_0_pytorch",
                  "claustrum_axial_1_pytorch",
                  "claustrum_axial_2_pytorch",
                  "claustrum_coronal_0_pytorch",
                  "claustrum_coronal_1_pytorch",
                  "claustrum_coronal_2_pytorch",
                  "tidsQualityAssessment_pytorch",
                  "koniqMS_pytorch",
                  "koniqMS2_pytorch",
                  "koniqMS3_pytorch",
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

    antstorch_cache_directory = get_antstorch_cache_directory()

    if not os.path.exists(antstorch_cache_directory):
        os.makedirs(antstorch_cache_directory, exist_ok=True)

    target_file_name_path = os.path.join(antstorch_cache_directory, target_file_name)

    url = switch_networks(file_id)

    if url:
        # Download only if needed
        if not os.path.exists(target_file_name_path):
            download_url_to_file(url, target_file_name_path, hash_prefix=None, progress=show_progress)
        return target_file_name_path
    else:
        # No URL mapping (missing from switcher, or explicitly "" for a known
        # but not-yet-hosted id): allow cache-only ids, but be explicit.
        if os.path.exists(target_file_name_path):
            return target_file_name_path
        # Only get here if no URL mapping and not found in cache
        raise ValueError(
            (
                f"No URL mapping for file_id='{file_id}', and not found in cache: \n"
                f" {target_file_name_path}\n"
                "Add a mapping in get_pretrained_network(), or place the file in the cache."
            )
        )
