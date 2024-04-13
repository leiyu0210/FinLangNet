"""Module docstring

This script performs data processing and tokenization for machine learning models.
"""
import io
import itertools
import math
import pickle
import random
from random import shuffle
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from Data_process.data_augnment import delete_data, modify_data, subset_and_pad
from Data_process.data_token import month_income_tokenizer, inquiry_type_tokenizer
from Data_process.data_binning import bin_edges_dict_dz_numeric, bin_edges_dict_person_numeric



def map_values_to_bins(values_str, bin_edges):
    """
    Maps a comma-separated string of values to their corresponding bins.

    Args:
        values_str (str): A comma-separated string of values.
        bin_edges (list): A list of bin edges for binning the values.

    Returns:
        list: A list of binned value indices.
    """
    values = np.array(list(map(float, values_str.split(','))))
    binned_values = np.digitize(values, bins=bin_edges, right=True).tolist()
    binned_values = [int(x) if not (x == float('inf') or x == float('-inf')) else 0 for x in binned_values]
    return binned_values


def map_values_to_bins_single(values_str, bin_edges):
    """
    Maps a comma-separated string of values to their first corresponding bin.

    Args:
        values_str (str): A comma-separated string of values.
        bin_edges (list): A list of bin edges.

    Returns:
        int: The bin index of the first value in the string.
    """
    values = np.array(list(map(float, values_str.split(','))))
    binned_value = np.digitize(values, bins=bin_edges, right=True).tolist()[0]
    return binned_value

class MyData(IterableDataset):
    """Custom dataset that is iterable and capable of data augmentation.

    This class is designed for loading textual data from files, processing each line of text, and optionally augmenting
    the data if it is used for training purposes.

    Attributes:
        file_paths (list of str): List of file paths from which to load the data.
        is_train (bool): Specifies whether the dataset is used for training. Influences data augmentation.
        augmentation_prob (float): Probability with which data augmentation should be applied to a sample.
        shuffle_files (bool): Specifies whether to shuffle the file paths before processing.
    """
    def __init__(self, file_paths, is_train=True, augmentation_prob=0.8,shuffle_files=True):
        """
        Initializes the MyData object.

        Args:
            file_paths (list of str): Paths to the files from which to load the data.
            is_train (bool, optional): Indicates whether the dataset is used for training purposes.
                                       Defaults to True.
            augmentation_prob (float, optional): Probability of augmenting a data sample if is_train is True.
                                                 Defaults to 0.8.
            shuffle_files (bool, optional): Whether to shuffle the order of files from which data is loaded.
                                            Defaults to True.
        """
        if shuffle_files:
            shuffle(file_paths)
        self.file_paths = file_paths
        self.is_train = is_train
        self.augmentation_prob = augmentation_prob
        
    def __iter__(self):
        """Iterates over the dataset, yielding processed (and possibly augmented) data samples.

        This method supports distributed data loading with multiple workers by dividing the dataset accordingly.

        Yields:
            Processed data tuples from the dataset. If is_train is True and the augmentation condition is met,
            augmented samples are yielded instead.
        """
        worker_info = torch.utils.data.get_worker_info()
        world_size = 1
        rank = 0
        worker_id = 0

        if worker_info:  # Adjusting for PyTorch's DataLoader parallelism
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            mod = world_size * num_workers
            shift = worker_id
        else:
            mod = world_size
            shift = rank

        line_iter = itertools.chain(*[io.open(f, encoding="utf-8") for f in self.file_paths])
        for i, line in enumerate(line_iter):
            if (i + shift) % mod == 0:
                return_line = self.process(line)
                if return_line:
                    if self.is_train and torch.rand(1).item() < self.augmentation_prob:
                        augmented_samples = self.augment_data(return_line)
                        yield from augmented_samples
                    else:
                        yield return_line

    def process(self, line):
        """Processes a single line from the dataset.

        The processing steps should ideally be replaced with actual data processing logic.

        Args:
            line (str): A line of text from the dataset file.

        Returns:
            tuple: A tuple containing processed features and label from the line of text.
        """
        text_data = line
        dz_categorica_feature, dz_numeric_feature, person_feature , label, len_dz = process_one_sample(text_data)
        dz_categorica_feature = torch.tensor(dz_categorica_feature)
        dz_numeric_feature = torch.tensor(dz_numeric_feature)
        person_feature = torch.tensor(person_feature)
        label = torch.tensor(label)
        
        tuple_ = (dz_categorica_feature, dz_numeric_feature, person_feature,label, len_dz)
        return tuple_
    
    def augment_data(self, sample):
        """Augments a given data sample based on predefined rules.

        Args:
            sample (tuple): A data sample to augment.

        Returns:
            list: A list containing the original sample and its augmented versions.
        """
        dz_categorica_feature, dz_numeric_feature, person_feature,label, len_dz =  sample
        augmented_data_res = []
        augmented_data_res.append(sample)
        
        add_cnt = 0
        if int(label) == 1:
            add_cnt = 3
        else:
            add_cnt = 0
            
        for i in range(add_cnt):
            augmentation_type = random.choice(['delete', 'subseries', 'modify'])
            if augmentation_type == 'delete' and len_dz > 1 :
                augmented_delete = delete_data(sample)
                augmented_data_res.append(augmented_delete)
            elif augmentation_type == 'subseries' and len_dz > 1 :
                augmented_sub = subset_and_pad(sample)
                augmented_data_res.append(augmented_sub)

            elif augmentation_type == 'modify' and len_dz > 1 :
                augmented_modify = modify_data(sample)
                augmented_data_res.append(augmented_modify)

        return augmented_data_res

def process_2d_list(lst, k, pad_token=120):
    """
    Adjusts each 2D list's sub-list to a specified length by padding with a token or truncating.

    Args:
        lst (list of list of int): The 2D list to be processed.
        k (int): The desired length for each sub-list.
        pad_token (int, optional): The token used for padding shorter sub-lists. Defaults to 120.

    Returns:
        list of list of int: The processed 2D list with sub-lists adjusted to the desired length.
    """
    result = []
    for sub_lst in lst:
        if len(sub_lst) < k:
            # Pad shorter sub-lists with the pad_token
            sub_lst += [pad_token] * (k - len(sub_lst))
        else:
            # Truncate longer sub-lists to length k
            sub_lst = sub_lst[-k:]
        result.append(sub_lst)
    return result


def normalize_features(features, mean_values, std_dev):
    """
    Normalizes a list of features by subtracting the mean and dividing by the standard deviation.

    Args:
        features (list): The features to be normalized.
        mean_values (list): The mean values for normalization.
        std_dev (list): The standard deviations for normalization.

    Returns:
        list: The list of normalized features.
    """
    features = np.array(features)
    mean_values = np.array(mean_values).reshape((-1, 1))
    std_dev = np.array(std_dev).reshape((-1, 1))
    normalized_features = (features - mean_values) / std_dev
    return normalized_features.tolist()


def normalize_features_1d(features, mean_values, std_dev):
    """
    Normalizes a 1D array of features using provided mean values and standard deviations.

    This function is similar to `normalize_features` but designed for 1D arrays.

    Args:
        features (list or np.ndarray): The features to be normalized.
        mean_values (list or np.ndarray): Mean values for normalization.
        std_dev (list or np.ndarray): Standard deviations for normalization.

    Returns:
        list: Normalized features as a 1D list.
    """
    features = np.array(features)
    mean_values = np.array(mean_values)
    std_dev = np.array(std_dev)
    normalized_features = (features - mean_values) / std_dev
    return normalized_features.tolist()

def process_one_sample(features, max_length=120):
    """
    Processes a single sample's features from a tab-separated string into various feature formats.

    Args:
        features (str): A tab-separated string representing features of a sample.
        max_length (int, optional): Maximum length for categorica and numeric feature sequences. Defaults to 120.

    Returns:
        tuple: A tuple containing processed feature lists and label.
    """
    feature1, feature2, len1, label1, label2, label3, feature3, feature4, feature5, feature6, feature7, feature8 = features.split('\t')
    feature2 = feature2.strip('\n')  
    label = [int(label1),int(label2),int(label3)]  
    feature1 = float(feature1)    
    cls_token = 121 # Define class sequence feature token to represent one sentence
    
    # Processed numeric and categorica features followed by discretization
    # Example
    feature1 = month_income_tokenizer.transform([feature1])[0]
    feature2 = inquiry_type_tokenizer.transform([feature2])[0]
    person_categorica_feature = [feature1]
    person_numeric_feature = [feature2]
    person_numeric_feature_discretized = [map_values_to_bins_single(data_str, bin_edges_dict_person_numeric[feature_name]) for feature_name, data_str in zip(list(bin_edges_dict_person_numeric.keys()), person_numeric_feature)]
    person_feature = person_categorica_feature  + person_numeric_feature_discretized
    dz_categorica_feature = [feature3]
    dz_categorica_feature = [fe.split(',') for fe in dz_categorica_feature]
    dz_numeric_feature = [feature4]
    dz_numeric_feature = [map_values_to_bins(data_str, bin_edges_dict_dz_numeric[feature_name]) for feature_name, data_str in zip(list(bin_edges_dict_dz_numeric.keys()), dz_numeric_feature)]
 
    max_length = 25
    len_dz= len(dz_categorica_feature[0])
    len_dz = len_dz if len_dz <=max_length else max_length
    dz_categorica_feature = [[int(value) for value in row] for row in dz_categorica_feature]
    dz_numeric_feature = [[float(value) if value != '' else 0.0 for value in row] for row in dz_numeric_feature]
    dz_categorica_feature = process_2d_list(dz_categorica_feature,max_length)
    # Adjust features to max_length and prepend class token
    dz_categorica_feature = [[cls_token] + sublist for sublist in dz_categorica_feature]
    dz_numeric_feature = process_2d_list(dz_numeric_feature,max_length)
    dz_numeric_feature = [[cls_token] + sublist for sublist in dz_numeric_feature]

    return dz_categorica_feature, dz_numeric_feature, person_feature, label, len_dz
   
if __name__ == "__main__":
    
    train_dataloader = DataLoader(MyData(['train_path'], is_train=False), batch_size=256, shuffle=False,drop_last = False,num_workers = 6)
    val_dataloader = DataLoader(MyData(['val_path'], is_train=False), batch_size=256, shuffle=False,drop_last = False,num_workers = 6)

    