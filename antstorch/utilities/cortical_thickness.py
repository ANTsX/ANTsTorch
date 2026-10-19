import ants
from .backend_ops import run_kelly_kapowski

def cortical_thickness(t1, device=None, verbose: bool = False, backend=None):
    """
    Perform KellyKapowski cortical thickness using antstorch.deep_atropos for
    segmentation.

    Parameters
    ----------
    t1 : ANTsImage
        Input 3-D unprocessed T1-weighted brain image.

    verbose : bool
        Print progress to the screen.

    backend : str, dict or None
        "ants" or "antstorch" for KellyKapowski (key "kelly_kapowski"), and for the
        N4 and template registration inside deep_atropos (keys "n4", "registration").
        None uses antstorch.set_default_backend(), then ANTSTORCH_BACKEND, then "antstorch".

    Returns
    -------
    dict
        Dictionary containing the cortical thickness image and segmentation/probability images.
    """

    from ..utilities.deep_atropos import deep_atropos
    from ..utilities.device_manager import get_default_device

    atropos = deep_atropos([t1, None, None], do_preprocessing=True, device=device, verbose=verbose,
                           backend=backend)
            
    # Kelly-Kapowski cortical thickness (ANTs or ANTsTorch DiReCT, see ``backend``)
    kk_segmentation = ants.image_clone(atropos['segmentation_image'])
    kk_segmentation[kk_segmentation == 4] = 3

    gray_matter = atropos['probability_images'][2]
    white_matter = (atropos['probability_images'][3] + atropos['probability_images'][4])

    kk = run_kelly_kapowski(
        kk_segmentation, gray_matter, white_matter,
        iterations=45, gradient_step=0.025, velocity_smoothing_variance=1.5,
        backend=backend, device=device, verbose=verbose
    )

    return {
        'thickness_image': kk,
        'segmentation_image': atropos['segmentation_image'],
        'csf_probability_image': atropos['probability_images'][1],
        'gray_matter_probability_image': atropos['probability_images'][2],
        'white_matter_probability_image': atropos['probability_images'][3],
        'deep_gray_matter_probability_image': atropos['probability_images'][4],
        'brain_stem_probability_image': atropos['probability_images'][5],
        'cerebellum_probability_image': atropos['probability_images'][6],
    }


def cortical_thickness2(
    t1,
    device=None,
    optimizer: str = "direct",
    regularizer: str = "gaussian",
    verbose: bool = False,
):
    """Cortical-thickness workflow using ANTsTorch's tensor DiReCT engine.

    This provisional entry point mirrors :func:`cortical_thickness` while
    leaving its established ``ants.kelly_kapowski`` behavior unchanged.
    """
    from ..direct import kelly_kapowski
    from ..utilities.deep_atropos import deep_atropos

    atropos = deep_atropos(
        [t1, None, None], do_preprocessing=True, device=device, verbose=verbose
    )
    kk_segmentation = ants.image_clone(atropos["segmentation_image"])
    kk_segmentation[kk_segmentation == 4] = 3
    gray_matter = atropos["probability_images"][2]
    white_matter = atropos["probability_images"][3] + atropos["probability_images"][4]
    thickness = kelly_kapowski(
        kk_segmentation,
        gray_matter,
        white_matter,
        iterations=45,
        gradient_step=0.025,
        velocity_smoothing_variance=1.5,
        optimizer=optimizer,
        regularizer=regularizer,
        device=device,
        verbose=verbose,
    )
    return {
        "thickness_image": thickness,
        "segmentation_image": atropos["segmentation_image"],
        "csf_probability_image": atropos["probability_images"][1],
        "gray_matter_probability_image": atropos["probability_images"][2],
        "white_matter_probability_image": atropos["probability_images"][3],
        "deep_gray_matter_probability_image": atropos["probability_images"][4],
        "brain_stem_probability_image": atropos["probability_images"][5],
        "cerebellum_probability_image": atropos["probability_images"][6],
    }


def longitudinal_cortical_thickness(
    t1s,
    initial_template: "str|ants.ANTsImage" = "oasis",
    number_of_iterations: int = 1,
    refinement_transform: str = "antsRegistrationSyNQuick[a]",
    device=None,
    verbose: bool = False,
    backend=None
):
    """
    Longitudinal KellyKapowski cortical thickness using antstorch.deep_atropos for the
    SST priors and traditional Atropos+KK for each timepoint.

    Parameters
    ----------
    t1s : list[ANTsImage]
        List of 3-D unprocessed T1-weighted images from a single subject.
    initial_template : str or ANTsImage
        Template spec or image for the single-subject template (SST) orientation.
    number_of_iterations : int
        Number of SST refinement iterations.
    refinement_transform : str
        ants.registration transform string for SST refinement when s>0.
    verbose : bool
        Print progress.

    backend : str, dict or None
        "ants" or "antstorch" per operation (keys "n4", "denoise", "registration",
        "kelly_kapowski"); Atropos has no ANTsTorch equivalent and stays on ANTs.
        None uses antstorch.set_default_backend(), then ANTSTORCH_BACKEND, then "antstorch".

    Returns
    -------
    list
        Per-timepoint dictionaries (preprocessed image, thickness, segmentation, probability maps, transforms)
        plus the final SST appended at the end of the list.
    """
    # Torch-backed helpers
    from antstorch.utilities import get_antstorch_data as get_antsx_data
    from antstorch.utilities import preprocess_brain_image
    from antstorch.utilities import deep_atropos  # torch-backed

    # --- Initial SST (optionally refined) ---
    if isinstance(initial_template, str):
        template_file = get_antsx_data(initial_template)
        sst = ants.image_read(template_file)
    else:
        sst = initial_template

    for s in range(number_of_iterations):
        if verbose:
            print(f"Refinement iteration {s} (out of {number_of_iterations})")
        sst_tmp = ants.image_clone(sst) * 0
        for i, im in enumerate(t1s):
            if verbose:
                print("***************************")
                print(f"SST processing image {i} (out of {len(t1s)})")
                print("***************************")
            transform_type = "antsRegistrationSyNQuick[r]" if s == 0 else refinement_transform
            t1_pre = preprocess_brain_image(
                im,
                truncate_intensity=(0.01, 0.99),
                brain_extraction_modality=None,
                template=sst,
                template_transform_type=transform_type,
                do_bias_correction=False,
                do_denoising=False,
                intensity_normalization_type="01",
                device=device,
                verbose=verbose,
                backend=backend
            )
            sst_tmp += t1_pre['preprocessed_image']
        sst = sst_tmp / len(t1s)

    # --- Preprocess each image to the final SST ---
    t1s_preprocessed = []
    for i, im in enumerate(t1s):
        if verbose:
            print("***************************")
            print(f"Final processing image {i} (out of {len(t1s)})")
            print("***************************")
        t1_pre = preprocess_brain_image(
            im,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality="t1",
            template=sst,
            template_transform_type="antsRegistrationSyNQuick[a]",
            do_bias_correction=True,
            do_denoising=True,
            intensity_normalization_type="01",
            device=device,
            verbose=verbose,
            backend=backend
        )
        t1s_preprocessed.append(t1_pre)

    # --- Torch Deep Atropos on SST to get priors ---
    sst_atropos = deep_atropos([sst, None, None], do_preprocessing=True, device=device, verbose=verbose,
                               backend=backend)

    # --- Traditional Atropos + KellyKapowski per timepoint ---
    return_list = []
    for i, pre in enumerate(t1s_preprocessed):
        if verbose:
            print(f"Atropos for image {i} (out of {len(t1s)})")
        atropos_output = ants.atropos(
            pre['preprocessed_image'],
            x=pre['brain_mask'],
            i=sst_atropos['probability_images'][1:7],
            m="[0.1,1x1x1]",
            c="[5,0]",
            priorweight=0.5,
            p="Socrates[1]",
            verbose=int(verbose)
        )

        kk_segmentation = ants.image_clone(atropos_output['segmentation'])
        kk_segmentation[kk_segmentation == 4] = 3
        gray_matter = atropos_output['probabilityimages'][1]
        white_matter = atropos_output['probabilityimages'][2] + atropos_output['probabilityimages'][3]

        kk = run_kelly_kapowski(
            kk_segmentation, gray_matter, white_matter,
            iterations=45, gradient_step=0.025, velocity_smoothing_variance=1.5,
            backend=backend, device=device, verbose=verbose
        )

        return_list.append({
            'preprocessed_image': pre['preprocessed_image'],
            'thickness_image': kk,
            'segmentation_image': atropos_output['segmentation'],
            'csf_probability_image': atropos_output['probabilityimages'][0],
            'gray_matter_probability_image': atropos_output['probabilityimages'][1],
            'white_matter_probability_image': atropos_output['probabilityimages'][2],
            'deep_gray_matter_probability_image': atropos_output['probabilityimages'][3],
            'brain_stem_probability_image': atropos_output['probabilityimages'][4],
            'cerebellum_probability_image': atropos_output['probabilityimages'][5],
            'template_transforms': pre['template_transforms']
        })

    # Append final SST
    return_list.append(sst)
    return return_list
