import ants

def cortical_thickness(t1, verbose: bool = False):
    """
    Perform KellyKapowski cortical thickness using antstorch.deep_atropos for
    segmentation.

    Parameters
    ----------
    t1 : ANTsImage
        Input 3-D unprocessed T1-weighted brain image.

    verbose : bool
        Print progress to the screen.

    Returns
    -------
    dict
        Dictionary containing the cortical thickness image and segmentation/probability images.
    """

    from antstorch.utilities import deep_atropos  

    # Run Deep Atropos (torch)
    if isinstance(t1, list):
        atropos = deep_atropos(t1, do_preprocessing=True, verbose=verbose)
    else:
        atropos = deep_atropos([t1, None, None], do_preprocessing=True, verbose=verbose)
            
    # Kelly-Kapowski cortical thickness (unchanged; uses ANTs ops)
    kk_segmentation = ants.image_clone(atropos['segmentation_image'])
    kk_segmentation[kk_segmentation == 4] = 3

    gray_matter = atropos['probability_images'][2]
    white_matter = (atropos['probability_images'][3] + atropos['probability_images'][4])

    kk = ants.kelly_kapowski(
        s=kk_segmentation,
        g=gray_matter,
        w=white_matter,
        its=45,
        r=0.025,
        m=1.5,
        x=0,
        verbose=int(verbose)
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


def longitudinal_cortical_thickness(
    t1s,
    initial_template: "str|ants.ANTsImage" = "oasis",
    number_of_iterations: int = 1,
    refinement_transform: str = "antsRegistrationSyNQuick[a]",
    verbose: bool = False
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
                verbose=verbose
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
            verbose=verbose
        )
        t1s_preprocessed.append(t1_pre)

    # --- Torch Deep Atropos on SST to get priors ---
    sst_atropos = deep_atropos(sst, do_preprocessing=True, verbose=verbose)

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

        kk = ants.kelly_kapowski(
            s=kk_segmentation,
            g=gray_matter,
            w=white_matter,
            its=45,
            r=0.025,
            m=1.5,
            x=0,
            verbose=int(verbose)
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
