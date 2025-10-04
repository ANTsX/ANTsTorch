import numpy as np
import ants
import torch

def deep_atropos(
    t1,
    do_preprocessing=True,
    verbose=False
):
    """
    Six-tissue segmentation.

    Perform Atropos-style six tissue segmentation using deep learning.

    The labeling is as follows:
    Label 0 :  background
    Label 1 :  CSF
    Label 2 :  gray matter
    Label 3 :  white matter
    Label 4 :  deep gray matter
    Label 5 :  brain stem
    Label 6 :  cerebellum

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * denoising,
       * brain extraction, and
       * affine registration to MNI.
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted brain image.

    do_preprocessing : boolean
        See description above.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dictionary consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> flash = deep_atropos(image)
    """

    # Local imports to avoid heavy deps at module import time
    from ..architectures import create_unet_model_3d as _create_unet_model_3d
    from ..utilities import get_antstorch_data as _get_antstorch_data
    from ..utilities import get_pretrained_network as _get_pretrained_network
    from ..utilities import brain_extraction as _brain_extraction

    if not isinstance(t1, list):
        raise ValueError("This antstorch implementation only handles the list-input branch ([T1, T2, FA]).")

    if len(t1) != 3:
        raise ValueError("Length of input list must be 3.  Input images are (in order): [T1, T2, FA]. "
                         "Use None as a placeholder for missing modalities.")

    if t1[0] is None:
        raise ValueError("T1 modality must be specified.")

    # Decide which network variant to use based on provided modalities
    which_network = ""
    input_images = [t1[0]]
    if t1[1] is not None and t1[2] is not None:
        which_network = "t1_t2_fa"
        input_images.append(t1[1])
        input_images.append(t1[2])
    elif t1[1] is not None:
        which_network = "t1_t2"
        input_images.append(t1[1])
    elif t1[2] is not None:
        which_network = "t1_fa"
        input_images.append(t1[2])
    else:
        which_network = "t1"

    if verbose:
        print("Prediction using", which_network)

    # ----------------------------
    # Preprocessing
    # ----------------------------
    def truncate_image_intensity(image, truncate_values=(0.01, 0.99)):
        truncated = ants.image_clone(image)
        q = (truncated.quantile(truncate_values[0]), truncated.quantile(truncate_values[1]))
        truncated[image < q[0]] = q[0]
        truncated[image > q[1]] = q[1]
        return truncated

    hcp_t1_template = ants.image_read(_get_antstorch_data("hcpinterT1Template"))
    hcp_template_brain_mask = ants.image_read(_get_antstorch_data("hcpinterTemplateBrainMask"))
    hcp_template_brain_segmentation = ants.image_read(_get_antstorch_data("hcpinterTemplateBrainSegmentation"))
    hcp_t1_template = hcp_t1_template * hcp_template_brain_mask

    reg = None
    t1_mask = None
    preprocessed_images = []
    for i, img in enumerate(input_images):
        n4 = ants.n4_bias_field_correction(
            truncate_image_intensity(img),
            mask=img * 0 + 1,
            convergence={'iters': [50, 50, 50, 50], 'tol': 0.0},
            rescale_intensities=True,
            verbose=verbose
        )
        if i == 0:
            t1_bext = _brain_extraction(input_images[0], modality="t1threetissue", verbose=verbose)
            t1_mask = ants.threshold_image(t1_bext['segmentation_image'], 1, 1, 1, 0)
            n4 = n4 * t1_mask
            reg = ants.registration(hcp_t1_template, n4,
                                    type_of_transform="antsRegistrationSyNQuick[a]",
                                    verbose=verbose)
            preprocessed_images.append(reg['warpedmovout'])
        else:
            n4 = n4 * t1_mask
            n4 = ants.apply_transforms(hcp_t1_template, n4,
                                       transformlist=reg['fwdtransforms'],
                                       verbose=verbose)
            preprocessed_images.append(n4)
        preprocessed_images[i] = ants.iMath_normalize(preprocessed_images[i])

    # ----------------------------
    # Build model and load weights
    # ----------------------------
    patch_size = (192, 224, 192)
    stride_length = (hcp_t1_template.shape[0] - patch_size[0],
                     hcp_t1_template.shape[1] - patch_size[1],
                     hcp_t1_template.shape[2] - patch_size[2])

    # Build template-derived priors (6 tissue priors)
    hcp_template_priors = []
    for i in range(6):
        prior = ants.threshold_image(hcp_template_brain_segmentation, i + 1, i + 1, 1, 0)
        prior_smooth = ants.smooth_image(prior, 1.0)
        hcp_template_priors.append(prior_smooth)

    classes = ("background", "csf", "gray matter", "white matter",
               "deep gray matter", "brain stem", "cerebellum")
    n_classes = len(classes)
    in_channels = len(preprocessed_images) + len(hcp_template_priors)  # modalities + priors

    # ANTsTorch U-Net (generic 3D builder)
    model = _create_unet_model_3d(
        input_channel_size=in_channels,      # channels
        number_of_outputs=n_classes,
        mode="classification",
        number_of_filters=(16, 32, 64, 128),
        dropout_rate=0.0,
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pad_crop="keras"
    )

    if verbose:
        print("DeepAtropos:  retrieving model weights.")

    # Map to converted PyTorch weights (we follow the *_pytorch.pt convention)
    weights_key = {
        "t1": "DeepAtroposHcpT1Weights",
        "t1_t2": "DeepAtroposHcpT1T2Weights",
        "t1_fa": "DeepAtroposHcpT1FAWeights",
        "t1_t2_fa": "DeepAtroposHcpT1T2FAWeights",
    }[which_network]
    weights_path = _get_pretrained_network(weights_key + "_pytorch")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # ----------------------------
    # Prediction (8 octant patches)
    # ----------------------------
    if verbose:
        print("Prediction.")

    # Pre-allocate prediction array: (8, D, H, W, C_out)
    predicted_data = np.zeros((8, *patch_size, n_classes), dtype=np.float32)

    # Working tensor (N=1, C, D, H, W)
    batchX = np.zeros((in_channels, *patch_size), dtype=np.float32)

    with torch.no_grad():
        for h in range(8):
            idx = 0
            # Modalities
            for img in preprocessed_images:
                patches = ants.extract_image_patches(
                    img, patch_size=patch_size, max_number_of_patches="all",
                    stride_length=stride_length, return_as_array=True
                )
                batchX[idx, :, :, :] = patches[h, :, :, :]
                idx += 1
            # Priors
            for prior in hcp_template_priors:
                patches = ants.extract_image_patches(
                    prior, patch_size=patch_size, max_number_of_patches="all",
                    stride_length=stride_length, return_as_array=True
                )
                batchX[idx, :, :, :] = patches[h, :, :, :]
                idx += 1

            xt = torch.from_numpy(batchX[None, ...])  # (1, C, D, H, W)
            xt = xt.to(device=device, dtype=torch.float32)

            yt = model(xt)  # (1, C_out, D, H, W), logits or probs depending on head
            y_np = yt.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (D,H,W,C_out)
            predicted_data[h, ...] = y_np

    # ----------------------------
    # Reconstruct prob maps and warp back if needed
    # ----------------------------
    probability_images = []
    for i, cname in enumerate(classes):
        if verbose:
            print("Reconstructing image", cname)

        recon = ants.reconstruct_image_from_patches(
            predicted_data[..., i],
            domain_image=hcp_t1_template,
            stride_length=stride_length
        )

        if do_preprocessing:
            # Map back to native space of T1
            recon_native = ants.apply_transforms(
                fixed=input_images[0],
                moving=recon,
                transformlist=reg['invtransforms'],
                whichtoinvert=[True],
                interpolator="linear",
                verbose=verbose
            )
            probability_images.append(recon_native)
        else:
            probability_images.append(recon)

    image_matrix = ants.image_list_to_matrix(probability_images, input_images[0] * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), input_images[0] * 0 + 1
    )[0]

    return {
        "segmentation_image": segmentation_image,
        "probability_images": probability_images,
    }
