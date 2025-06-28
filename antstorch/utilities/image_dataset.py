
from torch.utils.data import Dataset

import ants
import torch
import numpy as np
import random

class ImageDataset(Dataset):

    """
    Class for defining an image dataset with data augmentation.

    Arguments
    ---------
    images : list
        List of ANTs images or image files.  Or list of list of 
        ANTs images or image files in the case of multiple 
        modalities (i.e., channels).

    template : ANTs image
        ANts image defining the reference space for normalization.

    outputs: list
        List of output variables, possibly ANTs segmentation images.

    do_data_augmentation: boolean
        Use random spatial transformations, added noise, added bias fields,
        and histogram warping for image data augmentation.        

    data_augmentation_transform_type : string
        One of the following options: "translation", "rigid", "scaleShear", "affine",
        "deformation", "affineAndDeformation".

    data_augmentation_sd_affine : float
        Determines the amount of affine transformation.

    data_augmentation_sd_deformation : float
        Determines the amount of deformable transformation.
        
    data_augmentation_noise_model : string
        'additivegaussian', 'saltandpepper', 'shot', and 'speckle'. Alternatively, one
        can specify a tuple or list of one or more of the options and one is selected
        at random with reasonable, randomized parameters.  Note that the "speckle" model
        takes much longer than the others.

    data_augmentation_sd_simulated_bias_field : float
        Characterize the standard deviation of the amplitude.

    data_augmentation_sd_histogram_warping : float
        Determines the strength of the bias field.

    is_output_segmentation: boolean
        Is the specified output (if not None) segmentation images.

    number_of_samples : integer
        Standard DataSet parameter.    

    Returns
    -------

    Example
    -------
    """    

    def __init__(self,
                 images,
                 template,
                 outputs=None,
                 do_data_augmentation=True,
                 data_augmentation_transform_type="affineAndDeformation",
                 data_augmentation_sd_affine=0.05,
                 data_augmentation_sd_deformation=0.2,
                 data_augmentation_noise_model="additivegaussian",
                 data_augmentation_sd_simulated_bias_field=1.0,
                 data_augmentation_sd_histogram_warping=0.05,
                 is_output_segmentation=False,
                 duplicate_channels=None,
                 number_of_samples=1):

        self.images = images
        self.number_of_modalities = 1
        if isinstance(self.images[0], list):
            self.number_of_modalities = len(self.images[0])
        self.outputs = outputs
        self.template = template
        self.do_data_augmentation = do_data_augmentation
        self.data_augmentation_transform_type = data_augmentation_transform_type
        self.data_augmentation_sd_affine = data_augmentation_sd_affine
        self.data_augmentation_sd_deformation = data_augmentation_sd_deformation
        self.data_augmentation_noise_model = data_augmentation_noise_model
        self.data_augmentation_sd_simulated_bias_field = data_augmentation_sd_simulated_bias_field
        self.data_augmentation_sd_histogram_warping = data_augmentation_sd_histogram_warping
        self.is_output_segmentation = is_output_segmentation
        self.number_of_samples = number_of_samples
        self.duplicate_channels = duplicate_channels

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        random_index = random.sample(list(range(len(self.images))), 1)[0]
        image = list()
        if isinstance(self.images[random_index], list):
            if ants.is_image(self.images[random_index][0]):     
                for i in range(self.number_of_modalities):
                    image.append(ants.image_clone(self.images[random_index][i]))
            else: 
                for i in range(self.number_of_modalities):
                    image.append(ants.image_read(self.images[random_index][i]))
        else:            
            if ants.is_image(self.images[random_index]): 
                image.append(ants.image_clone(self.images[random_index]))
            else:
                image.append(ants.image_read(self.images[random_index]))

        output = None
        if self.is_output_segmentation: 
            if ants.is_image(self.outputs[random_index]):
                output = ants.image_clone(self.outputs[random_index])
            else:  
                output = ants.image_read(self.outputs[random_index])
            
        if self.do_data_augmentation:
            data_aug = ants.data_augmentation(input_image_list=[image],
                                                segmentation_image_list=output,
                                                pointset_list=None,
                                                number_of_simulations=1,
                                                reference_image=self.template,
                                                transform_type=self.data_augmentation_transform_type,
                                                noise_model=self.data_augmentation_noise_model,
                                                sd_simulated_bias_field=self.data_augmentation_sd_simulated_bias_field,
                                                sd_histogram_warping=self.data_augmentation_sd_histogram_warping,
                                                sd_affine=self.data_augmentation_sd_affine,
                                                sd_deformation=self.data_augmentation_sd_deformation,
                                                output_numpy_file_prefix=None,
                                                verbose=False)
            image = data_aug['simulated_images'][0]
            if output is not None: 
                output = data_aug['simulated_segmentation_images'][0]
        else:    
            center_of_mass_template = ants.get_center_of_mass(self.template*0 + 1)
            center_of_mass_image = ants.get_center_of_mass(image[0]*0 + 1)
            translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
            xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
                center=np.asarray(center_of_mass_template), translation=translation)
            for i in range(self.number_of_modalities):
                image[i] = ants.apply_ants_transform_to_image(xfrm, image[i], self.template)
            output = ants.apply_ants_transform_to_image(xfrm, output, self.template, interpolation="nearestneighbor")

        for i in range(self.number_of_modalities):
            image[i] = ants.iMath_normalize(image[i])
        image_array = np.zeros((*image[0].shape, self.number_of_modalities))
        for i in range(self.number_of_modalities):
            if image[i].dimension == 2:
                image_array[:,:,i] = image[i].numpy()
            elif image[i].dimension == 3:    
                image_array[:,:,:,i] = image[i].numpy()

        if self.duplicate_channels is not None:
            if image.dimension == 2:
                image_array = np.tile(image_array, (1, 1, self.duplicate_channels))
            elif image.dimension == 3:    
                image_array = np.tile(image_array, (1, 1, 1, self.duplicate_channels))
            else:
                raise ValueError("Unrecognized dimension.")    

        # swap color axis because
        # numpy image: H x W x D x C
        # torch image: C x H x W x D
        
        image_tensor = None
        output_tensor = None

        if image[0].dimension == 2:
            image_array = image_array.transpose((2, 0, 1))
            image_tensor = torch.from_numpy(image_array)
            if output is not None:
                output_array = np.expand_dims(output.numpy(), axis=-1)
                output_array = output_array.transpose((2, 0, 1))
                output_tensor = torch.from_numpy(output_array)
        elif image[0].dimension == 3:
            image_array = image_array.transpose((3, 0, 1, 2))
            image_tensor = torch.from_numpy(image_array)
            if output is not None:
                output_array = np.expand_dims(output.numpy(), axis=-1)
                output_array = output_array.transpose((3, 0, 1, 2))
                output_tensor = torch.from_numpy(output_array)
        else:
            raise ValueError("Unrecognized dimension.")    

        if output_tensor is not None:
            return image_tensor, output_tensor
        elif self.outputs is None:
            return image_tensor
        else:
            return image_tensor, output
