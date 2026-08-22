"""
PyTorch port of antspynet.utilities.quality_assessment (random_mask,
tid_neural_image_assessment).

IMPORTANT -- confidence note (read before using tid_neural_image_assessment):

Unlike hippmapp3r_segmentation / hypothalamus_segmentation / claustrum_segmentation
(ported the same day, 2026-08-22), the ResNet architecture(s) behind
"tidsQualityAssessment" / "koniqMS" / "koniqMS2" / "koniqMS3" are NOT built via
any explicit architecture-constructor function anywhere in ANTsPyNet -- they
are always loaded whole via tf.keras.models.load_model(...) from a saved
model file. There is therefore no source of truth in the ANTsPyNet source
tree for the exact ResNet hyperparameters (depth, lowest_resolution,
cardinality, squeeze_and_excite, etc.) these particular models were trained
with -- nothing analogous to the isotropic-kernel rule that made the
mri_super_resolution DBPN port at least an informed extrapolation.

Consequently, _default_qa_resnet_model() below is a **best-effort placeholder**
built from antstorch's already-ported create_resnet_model_2d with its plain
defaults (mode="regression", 2 outputs) -- it is not known to match the real
trained models and should be treated as a stand-in, not a verified port. The
recommended way to use tid_neural_image_assessment today is to pass an
already-built-and-weight-loaded torch.nn.Module directly via `which_model`
(exactly mirroring ANTsPyNet's own support for passing a user-defined Keras
model), which sidesteps this uncertainty entirely. If/when a real
tidsQualityAssessment/koniq weights+architecture pair is obtained, use
`architecture_kwargs` to correct the default, or inspect the real model to
fix _default_qa_resnet_model() directly.
"""

import random

import numpy as np
import torch
import ants


def _load_state_dict(weights_file_name):
    sd = torch.load(weights_file_name, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def random_mask(x, n):
    """
    Subsample voxels from the input mask to create a random mask.

    PyTorch/antstorch port of antspynet.utilities.random_mask -- pure
    numpy/ants logic, no neural network involved. Note: the original
    ANTsPyNet implementation has an off-by-one bug (`random.randint(0, xsz)`
    can return the out-of-bounds index `xsz`, since `random.randint` is
    inclusive on both ends) that can raise an IndexError; fixed here to
    `random.randint(0, xsz - 1)`. Sampling is still with replacement (matches
    the original's apparent intent), so requesting n close to xsz may return
    fewer than n nonzero voxels.

    Arguments
    ---------
    x : ANTsImage (2-D or 3-D)
        input mask.

    n : integer
        number of nonzero entries.

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> image = ants.image_read(ants.get_ants_data("r16"))
    >>> mask = ants.get_mask(image)
    >>> mask = random_mask(mask, 5)
    """
    xsz = int((x == 1).sum())
    if n > xsz:
        return x
    binvec = np.zeros(xsz)
    randinds = [random.randint(0, xsz - 1) for _ in range(n)]
    binvec[randinds] = 1
    xnew = x * 0
    xnew[x == 1] = binvec
    return xnew


def _default_qa_resnet_model(number_of_outputs=2, architecture_kwargs=None, device=None):
    from ..architectures import create_resnet_model_2d

    config = dict(input_channel_size=3, number_of_outputs=number_of_outputs,
                  mode="regression")
    config.update(architecture_kwargs or {})
    model = create_resnet_model_2d(**config)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def tid_neural_image_assessment(image,
                                mask=None,
                                patch_size=101,
                                stride_length=None,
                                padding_size=0,
                                dimensions_to_predict=0,
                                which_model="tidsQualityAssessment",
                                image_scaling=(255, 127.5),
                                do_patch_scaling=False,
                                no_reconstruction=False,
                                architecture_kwargs=None,
                                device=None,
                                verbose=False):

    """
    Perform MOS-based assessment of an image.

    Use a ResNet architecture to estimate image quality in 2D or 3D using subjective
    QC image databases described in

    https://www.sciencedirect.com/science/article/pii/S0923596514001490

    or

    https://doi.org/10.1109/TIP.2020.2967829

    where the image assessment is either "global", i.e., a single number or an image
    based on the specified patch size.  In the 3-D case, neighboring slices are used
    for each estimate.  Note that parameters should be kept as consistent as possible
    in order to enable comparison.  Patch size should be roughly 1/12th to 1/4th of
    image size to enable locality. A global estimate can be gained by setting
    patch_size = "global".

    PyTorch port of antspynet.utilities.tid_neural_image_assessment.
    UNVERIFIED / best-effort for the built-in `which_model` string options --
    see the confidence note at the top of this module. Passing an
    already-built torch.nn.Module directly via `which_model` is recommended
    until real weights are available.

    Arguments
    ---------
    image : ANTsImage (2-D or 3-D)
        input image.

    mask : ANTsImage (2-D or 3-D)
        optional mask for designating calculation ROI.

    patch_size : integer or "global"
        prime number of patch_size.  101 is good.  Otherwise, choose "global" for a single
        global estimate of quality.

    stride_length : integer or vector of image dimension length
        optional value to speed up computation (typically less than patch size).

    padding_size : positive or negative integer or vector of image dimension length
        de(padding) to remove edge effects.

    dimensions_to_predict : integer or vector
        if image dimension is 3, this parameter specifies which dimensions should be used for
        prediction.  If more than one dimension is specified, the results are averaged.

    which_model : string or torch.nn.Module
        model type e.g. string "tidsQualityAssessment", "koniqMS", "koniqMS2" or
        "koniqMS3" where the former predicts mean opinion score (MOS) and MOS
        standard deviation and the latter koniq models predict mean opinion
        score (MOS) and sharpness. Passing an already-built (and
        weight-loaded) torch.nn.Module directly is also valid -- recommended,
        see the module-level confidence note.

    image_scaling : a two-tuple where the first value is the multiplier and the
        second value the subtractor so each image will be scaled as
        img = ants.iMath(img, "Normalize") * m - s.

    do_patch_scaling : boolean controlling whether each patch is scaled or
        (if False) only a global scaling of the image is used.

    no_reconstruction : boolean; reconstruction is time consuming -- turn this on
        if you just want the predicted values.

    architecture_kwargs : dict, optional
        Override any of the default ResNet constructor kwargs used when
        `which_model` is a built-in string (ignored when `which_model` is
        already an nn.Module). See the module-level confidence note.

    device : torch.device or string, optional
        Device to run inference on.  Defaults to antstorch's default device.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dict of QC results predicting both human raters' mean and standard
    deviation of the MOS ("mean opinion scores"), or MOS and sharpness
    depending on the selected network.  Both aggregate and spatial scores
    are returned, the latter in the form of an image (patchwise mode only).

    Example
    -------
    >>> image = ants.image_read(ants.get_ants_data("r16"))
    >>> mask = ants.get_mask(image)
    >>> tid = tid_neural_image_assessment(image, mask=mask, patch_size=101, stride_length=7)
    """

    from ..utilities import get_pretrained_network
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if isinstance(which_model, torch.nn.Module):
        tid_model = which_model
        tid_model.eval()
        tid_model = tid_model.to(device)
        which_model = "user_defined"
    else:
        valid_models = ("tidsQualityAssessment", "koniqMS", "koniqMS2", "koniqMS3")
        if which_model not in valid_models:
            raise ValueError("Please pass a valid model (one of " + str(valid_models) +
                             ") or an already-built torch.nn.Module.")

        if verbose:
            print("Neural QA:  retrieving model and weights.")

        number_of_outputs = 2
        weights_file_name = get_pretrained_network(which_model + "_pytorch",
            target_file_name=which_model + "_pytorch.pt")
        tid_model = _default_qa_resnet_model(number_of_outputs=number_of_outputs,
            architecture_kwargs=architecture_kwargs, device=device)
        tid_model.load_state_dict(_load_state_dict(weights_file_name), strict=True)
        tid_model.eval()

    def predict(batch_X):
        with torch.no_grad():
            x = torch.from_numpy(batch_X).float().to(device)
            return tid_model(x).cpu().numpy()

    is_koniq = "koniq" in which_model

    padding_size_vector = padding_size
    if isinstance(padding_size, int):
        padding_size_vector = np.repeat(padding_size, image.dimension)
    elif len(padding_size) == 1:
        padding_size_vector = np.repeat(padding_size[0], image.dimension)

    if isinstance(dimensions_to_predict, int):
        dimensions_to_predict = (dimensions_to_predict,)

    padded_image_size = image.shape + padding_size_vector
    padded_image = ants.pad_or_crop_image_to_size(image, padded_image_size)

    number_of_channels = 3

    if stride_length is None and patch_size != "global":
        stride_length = round(patch_size / 2)
        if image.dimension == 3:
            stride_length = (stride_length, stride_length, 1)

    ###############
    #
    #  Global
    #
    ###############
    if which_model == "tidsQualityAssessment":
        evaluation_image = ants.iMath(padded_image, "Normalize") * 255
    elif is_koniq:
        evaluation_image = ants.iMath(padded_image, "Normalize") * 2.0 - 1.0
    else:
        evaluation_image = ants.iMath(padded_image, "Normalize") * image_scaling[0] - image_scaling[1]

    if patch_size == "global":

        if image.dimension == 2:
            # NOTE: ANTsPyNet's own 2-D global branch has a shape bug
            # (`np.zeros((1, evaluation_image.shape, number_of_channels))`
            # nests a tuple inside the shape tuple, which numpy rejects) --
            # fixed here to the evidently-intended channels-first shape.
            batchX = np.zeros((1, number_of_channels, *evaluation_image.shape), dtype=np.float32)
            for k in range(number_of_channels):
                batchX[0, k, :, :] = evaluation_image.numpy()
            predicted_data = predict(batchX)

            if which_model == "tidsQualityAssessment":
                return {"MOS": None,
                        "MOS.standardDeviation": None,
                        "MOS.mean": predicted_data[0, 0],
                        "MOS.standardDeviationMean": predicted_data[0, 1]}
            else:
                return {"MOS.mean": predicted_data[0, 0],
                        "sharpness.mean": predicted_data[0, 1]}

        elif image.dimension == 3:
            mos_mean = 0
            mos_standard_deviation = 0
            d = 0
            not_padded_image_size = list(padded_image_size)
            del not_padded_image_size[dimensions_to_predict[d]]
            newsize = [number_of_channels, padded_image_size[dimensions_to_predict[d]]] + not_padded_image_size
            batchX = np.zeros(newsize, dtype=np.float32)[np.newaxis, ...]
            for k in range(number_of_channels):
                batchX[0, k, :, :, :] = evaluation_image.numpy()
            predicted_data = predict(batchX)
            mos_mean += predicted_data[0, 0]
            mos_standard_deviation += predicted_data[0, 1]

            mos_mean /= len(dimensions_to_predict)
            mos_standard_deviation /= len(dimensions_to_predict)
            if which_model == "tidsQualityAssessment":
                return {"MOS.mean": mos_mean, "MOS.standardDeviationMean": mos_standard_deviation}
            else:
                return {"MOS.mean": mos_mean, "sharpness.mean": mos_standard_deviation}

    ###############
    #
    #  Patchwise
    #
    ###############

    stride_length_vector = stride_length
    if isinstance(stride_length, int):
        if image.dimension == 2:
            stride_length_vector = (stride_length, stride_length)
    elif len(stride_length) == 1:
        if image.dimension == 2:
            stride_length_vector = (stride_length[0], stride_length[0])

    patch_size_vector = (patch_size, patch_size)

    if image.dimension == 2:
        dimensions_to_predict = (1,)

    permutations = list()

    mos = image * 0
    mos_standard_deviation = image * 0

    for d in range(len(dimensions_to_predict)):
        if image.dimension == 3:
            permutations.append((0, 1, 2))
            permutations.append((0, 2, 1))
            permutations.append((1, 2, 0))

            if dimensions_to_predict[d] == 0:
                patch_size_vector = (patch_size, patch_size, number_of_channels)
                if isinstance(stride_length, int):
                    stride_length_vector = (stride_length, stride_length, 1)
            elif dimensions_to_predict[d] == 1:
                patch_size_vector = (patch_size, number_of_channels, patch_size)
                if isinstance(stride_length, int):
                    stride_length_vector = (stride_length, 1, stride_length)
            elif dimensions_to_predict[d] == 2:
                patch_size_vector = (number_of_channels, patch_size, patch_size)
                if isinstance(stride_length, int):
                    stride_length_vector = (1, stride_length, stride_length)
            else:
                raise ValueError("dimensions_to_predict elements should be 0, 1, and/or 2 for a 3-D image.")

        if mask is None:
            patches = ants.extract_image_patches(evaluation_image, patch_size=patch_size_vector,
                stride_length=stride_length_vector, return_as_array=False)
        else:
            patches = ants.extract_image_patches(evaluation_image, patch_size=patch_size_vector,
                max_number_of_patches=int((mask == 1).sum()),
                return_as_array=False, mask_image=mask, randomize=False)

        # Channels-first (N, C, H, W) batch, unlike ANTsPyNet's channels-last.
        batchX = np.zeros((len(patches), number_of_channels, patch_size, patch_size), dtype=np.float32)

        is_good_patch = np.repeat(False, len(patches))
        patch_image = None
        for i in range(len(patches)):
            if patches[i].var() > 0:
                is_good_patch[i] = True
                patch_image = patches[i]
                patch_image = patch_image - patch_image.min()

                if patch_image.max() > 0:
                    if which_model == "tidsQualityAssessment" and do_patch_scaling:
                        patch_image = patch_image / patch_image.max() * 255
                    elif is_koniq and do_patch_scaling:
                        patch_image = patch_image / patch_image.max() * 2.0 - 1.0
                    elif which_model == "user_defined" and do_patch_scaling:
                        patch_image = patch_image / patch_image.max() * image_scaling[0] - image_scaling[1]

                if image.dimension == 2:
                    for j in range(number_of_channels):
                        batchX[i, j, :, :] = patch_image
                elif image.dimension == 3:
                    batchX[i, :, :, :] = np.transpose(np.squeeze(patch_image), permutations[dimensions_to_predict[d]])

        good_batchX = batchX[is_good_patch, :, :, :]
        predicted_data = predict(good_batchX)

        if no_reconstruction:
            return predicted_data

        patches_mos = list()
        patches_mos_standard_deviation = list()

        zero_patch_image = patch_image * 0

        count = 0
        for i in range(len(patches)):
            if is_good_patch[i]:
                patches_mos.append(zero_patch_image + predicted_data[count, 0])
                patches_mos_standard_deviation.append(zero_patch_image + predicted_data[count, 1])
                count += 1
            else:
                patches_mos.append(zero_patch_image)
                patches_mos_standard_deviation.append(zero_patch_image)

        if mask is None:
            mos = mos + ants.pad_or_crop_image_to_size(ants.reconstruct_image_from_patches(
                patches_mos, evaluation_image, stride_length=stride_length_vector), image.shape)
            mos_standard_deviation = mos_standard_deviation + ants.pad_or_crop_image_to_size(
                ants.reconstruct_image_from_patches(patches_mos_standard_deviation, evaluation_image,
                    stride_length=stride_length_vector), image.shape)
        else:
            mos = mos + ants.pad_or_crop_image_to_size(ants.reconstruct_image_from_patches(
                patches_mos, mask, domain_image_is_mask=True), image.shape)
            mos_standard_deviation = mos_standard_deviation + ants.pad_or_crop_image_to_size(
                ants.reconstruct_image_from_patches(patches_mos_standard_deviation, mask,
                    domain_image_is_mask=True), image.shape)

    mos = mos / len(dimensions_to_predict)
    mos_standard_deviation = mos_standard_deviation / len(dimensions_to_predict)

    if mask is None:
        if which_model == "tidsQualityAssessment":
            return {"MOS": mos, "MOS.standardDeviation": mos_standard_deviation,
                    "MOS.mean": mos.mean(), "MOS.standardDeviationMean": mos_standard_deviation.mean()}
        else:
            return {"MOS": mos, "sharpness": mos_standard_deviation,
                    "MOS.mean": mos.mean(), "sharpness.mean": mos_standard_deviation.mean()}
    else:
        if which_model == "tidsQualityAssessment":
            return {"MOS": mos, "MOS.standardDeviation": mos_standard_deviation,
                    "MOS.mean": (mos[mask >= 0.5]).mean(),
                    "MOS.standardDeviationMean": (mos_standard_deviation[mask >= 0.5]).mean()}
        else:
            return {"MOS": mos, "sharpness": mos_standard_deviation,
                    "MOS.mean": (mos[mask >= 0.5]).mean(),
                    "sharpness.mean": (mos_standard_deviation[mask >= 0.5]).mean()}
