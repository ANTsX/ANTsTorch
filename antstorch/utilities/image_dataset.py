
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
        List of ANTs images or image files.

    template : ANTs image
        ANts image defining the reference space for normalization.

    outputs: list
        List of output variables, possibly ANTs segmentation images.

    do_data_augmentation: boolean
        Use random spatial transformations, added noise, added bias fields,
        and histogram warping for image data augmentation.        

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
                 is_output_segmentation=False,
                 number_of_samples=1):

        self.images = images
        self.outputs = outputs
        self.template = template
        self.do_data_augmentation = do_data_augmentation
        self.is_output_segmentation = is_output_segmentation
        self.number_of_samples = number_of_samples

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        random_index = random.sample(list(range(len(self.images))), 1)[0]
        if ants.is_image(self.image[random_index]):
            image = ants.image_clone(self.image[random_index])
        else:  
            image = ants.image_read(self.image[random_index])

        if self.is_output_segmentation: 
            if ants.is_image(self.outputs[random_index]):
                output = ants.image_clone(self.outputs[random_index])
            else:  
                output = ants.image_read(self.outputs[random_index])
            
        if self.do_data_augmentation:
            noise_model = None
            if random.uniform(0.0, 1.0) > 0.33:
                noise_model = ("additivegaussian", "shot", "saltandpepper")
            data_aug = ants.data_augmentation(input_image_list=[[image]],
                                                segmentation_image_list=output,
                                                pointset_list=None,
                                                number_of_simulations=1,
                                                reference_image=self.template,
                                                transform_type='affineAndDeformation',
                                                noise_model=noise_model,
                                                sd_simulated_bias_field=1.0,
                                                sd_histogram_warping=0.05,
                                                sd_affine=0.05,
                                                output_numpy_file_prefix=None,
                                                verbose=False)
            image = data_aug['simulated_images'][0][0]
            output = data_aug['simulated_segmentation_images'][0]
        else:    
            center_of_mass_template = ants.get_center_of_mass(self.template*0 + 1)
            center_of_mass_image = ants.get_center_of_mass(image*0 + 1)
            translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
            xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
                center=np.asarray(center_of_mass_template), translation=translation)
            image = ants.apply_ants_transform_to_image(xfrm, image, self.template)
            if is
            output = ants.apply_ants_transform_to_image(xfrm, output, self.template, interpolation="nearestneighbor")

        image = (image - image.min()) / (image.max() - image.min())
        image_array = np.expand_dims(image.numpy(), axis=-1)

        # swap color axis because
        # numpy image: H x W x D x C
        # torch image: C x H x W x D

        image_array = image_array.transpose((3, 0, 1, 2))
        image_tensor = torch.from_numpy(image_array)

        if self.is_output_segmentation:
            output_array = np.expand_dims(output.numpy(), axis=-1)
            output_array = output_array.transpose((3, 0, 1, 2))
            output_tensor = torch.from_numpy(output_array)
            return image_tensor, output_tensor
        elif self.outputs is None:
            return image_tensor
        else:
            return image_tensor, output
