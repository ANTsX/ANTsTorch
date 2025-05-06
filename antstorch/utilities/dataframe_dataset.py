
from torch.utils.data import Dataset

import torch
import numpy as np
import random



class DataFrame(Dataset):

    """
    Class for defining a data frame (i.e., pandas) dataset.  

    Arguments
    ---------
    dataframe : pandas dataframe
        Pandas dataframe with column names.

    alpha : scalar
        Parameter for defining the standard deviation in random 
        data augmentation.

    do_normalize : boolean
        Normalize values (column-wise) to [0, 1].

    do_data_augmentation :  boolean    
        Allow for generation of random values.

    number_of_samples : integer
        Standard DataSet parameter.    

    Returns
    -------

    Singleton PyTorch tensor

    Data frame with probability values for each disease category.

    Example
    -------
    >>> x = torch.rand((5, 2))
    >>> y = torch.rand((5, 2))
    >>> mi = antstorch.mutual_information_kde(x, y)
    """    

    def __init__(self,
                 dataframe,
                 alpha=0.01,
                 do_normalize=True,
                 do_data_augmentation=True,
                 number_of_samples=1):
        self.dataframe = dataframe
        self.number_of_samples = number_of_samples
        self.alpha = alpha
        self.do_normalize = do_normalize
        self.do_data_augmentation = do_data_augmentation
        self.dataframe_numpy = self.dataframe.to_numpy()
        self.number_of_measurements = self.dataframe_numpy.shape[0]
        self.data_std = np.std(self.dataframe_numpy, axis=0)
        self.data_mean = np.mean(self.dataframe_numpy, axis=0)
        self.data_min = np.min(self.dataframe_numpy, axis=0)
        self.data_max = np.max(self.dataframe_numpy, axis=0)

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        random_index = random.sample(range(self.number_of_measurements), k=1)[0]
        random_measurement = self.dataframe_numpy[random_index,:]
        if self.do_data_augmentation:
            random_measurement += np.random.normal(np.zeros(random_measurement.shape),
                                                   self.alpha * self.data_std)
        if self.do_normalize:
            random_measurement = self.normalize_data(random_measurement)
        return random_measurement 

    def normalize_data(self, data):
        if len(data.shape) == 2:
            min = np.tile(self.data_min, (self.number_of_measurements, 1))
            max = np.tile(self.data_max, (self.number_of_measurements, 1))
            normalized_data = (data - min) / (max - min)
        else:
            normalized_data = (data - self.data_min) / (self.data_max - self.data_min) 
        return normalized_data       

    def denormalize_data(self, data):
        if len(data.shape) == 2:
            min = np.tile(self.data_min, (self.number_of_measurements, 1))
            max = np.tile(self.data_max, (self.number_of_measurements, 1))
            denormalized_data = data * (max - min) + min
        else:
            denormalized_data = data * (self.data_max - self.data_min) + self.data_min
        return denormalized_data       
