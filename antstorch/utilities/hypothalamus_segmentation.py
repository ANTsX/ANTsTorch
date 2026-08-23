import numpy as np
import torch
import ants


def _load_state_dict(weights_file_name):
    sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def hypothalamus_segmentation(t1,
                              device=None,
                              verbose=False):

    """
    Hypothalamus and subunits segmentation

    Described here:

        https://pubmed.ncbi.nlm.nih.gov/32853816/

    ported from the original implementation

        https://github.com/BBillot/hypothalamus_seg

    PyTorch port of antspynet.utilities.hypothalamus_segmentation.

    Subunits labeling:

    Label 1:  left anterior-inferior
    Label 2:  left anterior-superior
    Label 3:  left posterior
    Label 4:  left tubular inferior
    Label 5:  left tubular superior
    Label 6:  right anterior-inferior
    Label 7:  right anterior-superior
    Label 8:  right posterior
    Label 9:  right tubular inferior
    Label 10: right tubular superior

    Arguments
    ---------
    t1 : ANTsImage
        input 3-D T1 brain image.

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dict with 'segmentation_image' (argmax label image) and
    'probability_images' (list of 11 per-class probability images, in the
    class order given above, background first).

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> hypo = hypothalamus_segmentation(image)
    """

    from ..architectures import create_hypothalamus_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    classes = ("background",
               "left anterior-inferior",
               "left anterior-superior",
               "left posterior",
               "left tubular inferior",
               "left tubular superior",
               "right anterior-inferior",
               "right anterior-superior",
               "right posterior",
               "right tubular inferior",
               "right tubular superior")

    ################################
    #
    # Rotate to proper orientation
    #
    ################################

    reference_image = ants.make_image((256, 256, 256),
                                      voxval=0,
                                      spacing=(1, 1, 1),
                                      origin=(0, 0, 0),
                                      direction=np.diag((-1.0, -1.0, 1.0)))
    center_of_mass_reference = ants.get_center_of_mass(reference_image + 1)
    center_of_mass_image = ants.get_center_of_mass(t1 * 0 + 1)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_reference), translation=translation)
    xfrm_inv = xfrm.invert()

    crop_image = ants.image_clone(t1) * 0 + 1
    crop_image = ants.apply_ants_transform_to_image(xfrm, crop_image, reference_image)
    crop_image = ants.crop_image(crop_image, label_image=crop_image, label=1)

    t1_warped = ants.apply_ants_transform_to_image(xfrm, t1, crop_image)
    t1_warped = ants.pad_or_crop_image_to_size(t1_warped, (204, 256, 256))

    ################################
    #
    # Normalize intensity
    #
    ################################

    t1_warped = (t1_warped - t1_warped.min()) / (t1_warped.max() - t1_warped.min())

    ################################
    #
    # Build model and load weights
    #
    ################################

    if verbose:
        print("Hypothalamus:  retrieving model weights.")

    model = create_hypothalamus_unet_model_3d(input_channel_size=1, number_of_outputs=len(classes))
    weights_file_name = get_pretrained_network("hypothalamus_pytorch",
        target_file_name="hypothalamus_pytorch.pt")
    model.load_state_dict(_load_state_dict(weights_file_name), strict=True)
    model.eval()
    model = model.to(device)

    ################################
    #
    # Do prediction
    #
    ################################

    if verbose:
        print("Prediction.")

    batch_X = np.expand_dims(t1_warped.numpy(), axis=(0, 1)).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(batch_X).float().to(device)
        predicted_data = model(x).cpu().numpy()

    probability_images = list()
    for i in range(len(classes)):
        if verbose:
            print("Processing image", classes[i])

        probability_image = ants.from_numpy(np.squeeze(predicted_data[0, i, :, :, :]),
            spacing=t1_warped.spacing, origin=t1_warped.origin,
            direction=t1_warped.direction)
        probability_images.append(xfrm_inv.apply_to_image(probability_image, t1))

    image_matrix = ants.image_list_to_matrix(probability_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    return_dict = {"segmentation_image": segmentation_image,
                   "probability_images": probability_images}
    return return_dict
