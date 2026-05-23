import ants
import numpy as np
import pandas as pd
import torch

def t1_grader(image, device=None, verbose=False):
    """
    Supervised grader / scoring of T1-weighted brain MRI.

    Arguments
    ---------
    image : ANTsImage
        raw 3-D T1 brain image.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Data frame with letter grade and numeric score predictions.

    Example
    -------
    >>> image = ants.image_read(antspynet.get_antsxnet_data("mprage_hippmapp3r"))
    >>> grade_df = t1_grader(image)
    """

    from ..utilities import get_pretrained_network
    from ..utilities import get_antstorch_data
    from ..architectures import create_resnet_model_3d
    from ..utilities.device_manager import get_default_device

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if image.dimension != 3:
        raise ValueError("Input image needs to be 3D.")

    ################################
    #
    # Prétraitement et Recalage
    #
    ################################

    t1 = ants.iMath(image - image.min(), "Normalize")
    bxt = ants.threshold_image(t1, 0.01, 1.0)
    t1 = ants.rank_intensity(t1, mask=bxt, get_mask=True)

    templateb_file = get_antstorch_data("S_template3_brain")
    templateb = ants.image_read(templateb_file)
    templateb = ants.crop_image(templateb).resample_image((1, 1, 1))
    templateb = ants.pad_image_by_factor(templateb, 8)
    templatebsmall = ants.resample_image(templateb, (2, 2, 2))

    reg = ants.registration(templatebsmall, t1, type_of_transform='Similarity', verbose=verbose)

    ilist = [[templateb]]
    nsim = 16
    uu = ants.randomly_transform_image_data(templateb, ilist,
                                            number_of_simulations=nsim,
                                            transform_type='scaleShear', 
                                            sd_affine=0.075)
    fwdaffgd = ants.read_transform(reg['fwdtransforms'][0])

    score_nums = np.zeros(4)
    score_nums[3] = 0
    score_nums[2] = 1
    score_nums[1] = 2
    score_nums[0] = 3
    score_nums = score_nums.reshape((4, 1))

    ################################
    #
    # Chargement du modèle et des poids
    #
    ################################

    weights_file_name = get_pretrained_network("resnet_grader_pytorch")

    model = create_resnet_model_3d(input_channel_size=1,
                                   lowest_resolution=32,
                                   number_of_outputs=4,
                                   cardinality=1,
                                   squeeze_and_excite=False)
    
    model.load_state_dict(torch.load(weights_file_name, map_location="cpu"))
    model.eval()

    def get_grade(score, probs):
        grade = 'f'
        if score >= 2.25:
            grade = 'a'
        elif score >= 1.5:
            grade = 'b'
        elif score >= 0.75:
            grade = 'c'
        
        probgradeindex = np.argmax(probs)
        probgrade = ['a', 'b', 'c', 'f'][probgradeindex]
        return [grade, probgrade, float(score)]

    gradelist_num = []
    gradelist_prob = []
    grade_score = []

    for k in range(nsim):
        simtx = uu['simulated_transforms'][k]
        cmptx = ants.compose_ants_transforms([simtx, fwdaffgd]) 
        subjectsim = ants.apply_ants_transform_to_image(cmptx, t1, templateb, interpolation='linear')
        subjectsim = ants.add_noise_to_image(subjectsim, 'additivegaussian', (0, 0.01))
        
        xarr = subjectsim.numpy()
        
        # PyTorch : (Batch, Channels, Depth, Height, Width)
        newshape = [1, 1] + list(xarr.shape)
        xarr = np.reshape(xarr, newshape)
        xtensor = torch.from_numpy(xarr).float()
        
        with torch.no_grad():
            preds_tensor = model(xtensor)
            preds = preds_tensor.cpu().numpy()
        
        predsnum = np.dot(preds, score_nums) 
        locgrades = get_grade(predsnum[0][0], preds[0]) 
        
        gradelist_num.append(locgrades[0])
        gradelist_prob.append(locgrades[1])
        grade_score.append(locgrades[2])

    def most_frequent(lst):
        return max(set(lst), key=lst.count)

    mydf = pd.DataFrame({
        "NumericGrade": gradelist_num,
        "ProbGrade": gradelist_prob,
        "NumericScore": grade_score,
        'grade': most_frequent(gradelist_prob)
    })

    smalldf = pd.DataFrame({
        'gradeLetter':  [mydf['grade'].iloc[0]],
        'gradeNum': [mydf['NumericScore'].mean()]
    }, index=[0])

    return smalldf