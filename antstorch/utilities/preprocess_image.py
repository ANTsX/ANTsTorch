from __future__ import annotations

import ants
from typing import Optional, Tuple, Dict, Any


def preprocess_brain_image(
    image: "ants.ANTsImage",
    truncate_intensity: Optional[Tuple[float, float]] = (0.01, 0.99),
    brain_extraction_modality: Optional[str] = None,
    template_transform_type: Optional[str] = None,
    template: "ants.ANTsImage | str" = "biobank",
    do_bias_correction: bool = True,
    return_bias_field: bool = False,
    do_denoising: bool = True,
    intensity_matching_type: Optional[str] = None,
    reference_image: Optional["ants.ANTsImage"] = None,
    intensity_normalization_type: Optional[str] = None,
    brain_mask: Optional["ants.ANTsImage"] = None,
    device=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Basic preprocessing pipeline for T1-weighted brain MRI

    Standard preprocessing steps that have been previously described
    in various papers including the cortical thickness pipeline:

         https://www.ncbi.nlm.nih.gov/pubmed/24879923

    Arguments
    ---------
    image : ANTsImage
        input image

    truncate_intensity : 2-length tuple
        Defines the quantile threshold for truncating the image intensity

    brain_extraction_modality : string or None
        Perform brain extraction using antstorch tools.  One of "t1", "t1v0",
        "t1nobrainer", "t1combined", "flair", "t2", "bold", "fa", "t1infant",
        "t2infant", "t1threetissue", or None.

    template_transform_type : string
        See details in help for ants.registration.  Typically "Rigid" or
        "Affine".

    template : ANTs image (not skull-stripped)
        Alternatively, one can specify the default "biobank" or "croppedMni152"
        to download and use premade templates.

    do_bias_correction : boolean
        Perform N4 bias field correction.

    return_bias_field : boolean
        If True, return bias field as an additional output *without* bias
        correcting the preprocessed image.

    do_denoising : boolean
        Perform non-local means denoising.

    intensity_matching_type : string
        Either "regression" or "histogram". Only is performed if reference_image
        is not None.

    reference_image : ANTs image
        Reference image for intensity matching.

    intensity_normalization_type : string
        Either rescale the intensities to [0,1] (i.e., "01") or zero-mean, unit variance
        (i.e., "0mean").  If None normalization is not performed.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dictionary with preprocessing information ANTs image (i.e., source_image) matched to the
    (reference_image).

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> preprocessed = preprocess_brain_image(image, brain_extraction_modality=None)
    """

    # Local imports to mirror original structure and keep dependency surface minimal
    import torch
    from ..utilities import brain_extraction
    from ..utilities import get_antstorch_data
    from ..utilities.device_manager import get_default_device
    from ..utilities.n4_bias_field_correction import n4_bias_field_correction
    from ..utilities.denoise_image import denoise_image
    from ..syn.robust_affine import robust_affine
    from ..greedy import greedy_registration

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    preprocessed_image = ants.image_clone(image, pixeltype='float')

    # Truncate intensity
    if truncate_intensity is not None:
        quantiles = (image.quantile(truncate_intensity[0]), image.quantile(truncate_intensity[1]))
        if verbose:
            print("Preprocessing:  truncate intensities ( low =", quantiles[0], ", high =", quantiles[1], ").")

        preprocessed_image[image < quantiles[0]] = quantiles[0]
        preprocessed_image[image > quantiles[1]] = quantiles[1]

    # Brain extraction
    mask = None
    if brain_mask is not None:
        mask = ants.threshold_image(ants.image_clone(brain_mask, pixeltype='float'), 0.5, 1e9, 1, 0)
    elif brain_extraction_modality is not None:
        if verbose:
            print("Preprocessing:  brain extraction.")

        bext = brain_extraction(preprocessed_image, modality=brain_extraction_modality, device=device, verbose=verbose)
        if brain_extraction_modality == "t1threetissue":
            mask = ants.threshold_image(bext['segmentation_image'], 1, 1, 1, 0)
        else:
            mask = ants.threshold_image(bext, 0.5, 1, 1, 0)
            mask = ants.morphology(mask, "close", 6).iMath_fill_holes()

    def _register_to_template(fixed_img, moving_img, tx_type):
        tx_lower = str(tx_type).lower()
        if "syn" in tx_lower and not tx_lower.endswith("[a]") and not tx_lower.endswith("[r]"):
            return greedy_registration(
                fixed=fixed_img, moving=moving_img, return_inverse=True, device=str(device), verbose=verbose
            )
        return robust_affine(
            fixed=fixed_img, moving=moving_img, mode="auto", device=str(device), verbose=verbose
        )

    # Template normalization
    transforms = None
    if template_transform_type is not None:
        if isinstance(template, str):
            template_file_name_path = get_antstorch_data(template)
            template_image = ants.image_read(template_file_name_path)
        else:
            template_image = template

        if mask is None:
            registration = _register_to_template(template_image, preprocessed_image, template_transform_type)
            preprocessed_image = registration['warpedmovout']
            transforms = dict(fwdtransforms=registration['fwdtransforms'],
                              invtransforms=registration['invtransforms'])
        else:
            bext_mod = brain_extraction_modality or "t1"
            template_bext = brain_extraction(template_image, modality=bext_mod, device=device, verbose=verbose)
            if bext_mod == "t1threetissue":
                template_mask = ants.threshold_image(template_bext['segmentation_image'], 1, 1, 1, 0)
            else:
                template_mask = ants.threshold_image(template_bext, 0.5, 1, 1, 0)
            template_brain_image = template_mask * template_image

            preprocessed_brain_image = mask * preprocessed_image

            registration = _register_to_template(template_brain_image, preprocessed_brain_image, template_transform_type)
            transforms = dict(fwdtransforms=registration['fwdtransforms'],
                              invtransforms=registration['invtransforms'])

            preprocessed_image = ants.apply_transforms(
                fixed=template_image, moving=preprocessed_image,
                transformlist=registration['fwdtransforms'], interpolator="linear",
                singleprecision=True, verbose=verbose
            )
            mask = ants.apply_transforms(
                fixed=template_image, moving=mask, transformlist=registration['fwdtransforms'],
                interpolator="genericLabel", singleprecision=True, verbose=verbose
            )

    # Do bias correction
    bias_field = None
    if do_bias_correction:
        if verbose:
            print("Preprocessing:  bias correction.")
        n4_output = n4_bias_field_correction(
            preprocessed_image,
            mask=mask,
            shrink_factor=4,
            return_bias_field=return_bias_field,
            device=device,
            verbose=verbose,
        )
        if return_bias_field:
            bias_field = n4_output
        else:
            preprocessed_image = n4_output

    # Denoising
    if do_denoising:
        if verbose:
            print("Preprocessing:  denoising.")
        preprocessed_image = denoise_image(
            preprocessed_image, mask=mask, shrink_factor=1, device=device, verbose=verbose
        )

    # Image matching
    if reference_image is not None and intensity_matching_type is not None:
        if verbose:
            print("Preprocessing:  intensity matching.")
        if intensity_matching_type == "regression":
            preprocessed_image = ants.regression_match_image(preprocessed_image, reference_image)
        elif intensity_matching_type == "histogram":
            preprocessed_image = ants.histogram_match_image(preprocessed_image, reference_image)
        else:
            raise ValueError("Unrecognized intensity_matching_type.")

    # Intensity normalization
    if intensity_normalization_type is not None:
        if verbose:
            print("Preprocessing:  intensity normalization.")
        if intensity_normalization_type == "01":
            preprocessed_image = (preprocessed_image - preprocessed_image.min()) / (
                preprocessed_image.max() - preprocessed_image.min()
            )
        elif intensity_normalization_type == "0mean":
            preprocessed_image = (preprocessed_image - preprocessed_image.mean()) / preprocessed_image.std()
        else:
            raise ValueError("Unrecognized intensity_normalization_type.")

    return_dict: Dict[str, Any] = {'preprocessed_image': preprocessed_image}
    if mask is not None:
        return_dict['brain_mask'] = mask
    if bias_field is not None:
        return_dict['bias_field'] = bias_field
    if transforms is not None:
        return_dict['template_transforms'] = transforms

    return return_dict
