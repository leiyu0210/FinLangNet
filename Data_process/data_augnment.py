import math
import random
import torch
import torch.nn.functional as F


def delete_data(sample):
    """Randomly deletes a certain fraction of data points and pads to maintain original size.

    Args:
        sample (tuple): A tuple containing various features and related information.

    Returns:
        tuple: The modified sample with certain data points deleted and padded.
    """
    augmented_sample = tuple(sample)
    (feature1, feature2, len1, feature3, feature4, len2, len3) = augmented_sample

    # Process fintech records.
    if random.random() < 0.75:
        feature1, feature2, len1 = delete_and_pad(
            feature1, feature2, len1, 120)

    # Process inquiry data.
    if random.random() < 0.75:
        feature3, len2 = delete_and_pad_feature_only(
            feature3, len2, 0.1, 60)

    # Process credit data.
    if random.random() < 0.75:
        feature4, len3 = delete_and_pad_feature_only(
            feature4, len3, 0.1, 120)

    return (feature1, feature2, len1, feature3, feature4, len2, len3)


def modify_data(sample):
    """Randomly replaces a certain fraction of data points.

    Args:
        sample (tuple): The original data sample tuple.

    Returns:
        tuple: The modified sample with certain data points replaced.
    """
    augmented_sample = tuple(sample)
    # Tuple unpacking remains the same; focus on function structure and naming.
    (feature1, feature2, len1, feature3, feature4, len2, len3) = augmented_sample

    if random.random() < 0.75:
       feature1, feature2, len1 = replace_data(feature1, feature2, len1)

    if random.random() < 0.75:
        feature3, len2 = replace_data(feature3, None, len2)

    if random.random() < 0.75:
        feature4, len3 = replace_data(feature4, None, len3)

    return (feature1, feature2, len1, feature3, feature4, len2, len3)


def subset_and_pad(sample):
    """Creates a subset and pads the data based on time series.

    Args:
        sample (tuple): Data sample to be modified.

    Returns:
        tuple: The modified data sample.
    """
    # Function structure follows the similar modification principles.
    augmented_sample = tuple(sample)
    (feature1, feature2, len1, feature3, feature4, len2, len3) = augmented_sample

    if random.random() < 0.75:
        feature1, feature2, len1 = sub_and_pad(feature1, feature2, len1, 120)

    if random.random() < 0.75:
        feature3, len2 = sub_and_pad(feature3, None, len2, 60)

    if random.random() < 0.75:
        feature4, len3 = sub_and_pad(feature4, None, len3, 120)

    return (feature1, feature2, len1, feature3, feature4, len2, len3)


def delete_and_pad(categorical_feature, numeric_feature, length, pad_length):
    """Deletes a fraction of feature data and pads it.
    
    This function randomly removes a portion of the feature data and pads the remanining
    data up to a specified `pad_length` with zeros for numeric features and a special padding
    ID for categorical features.
    
    Args:
        categorical_feature (torch.Tensor): The categorical feature tensor.
        numeric_feature (torch.Tensor): The numeric feature tensor.
        length (int): The original length of the data.
        pad_length (int): The length to pad the data to.

    Returns:
        tuple: Modified features and length.
    """
    # Calculate the number of entries to delete
    delete_count = int(length * random.uniform(0.1, 0.3))  # For example, delete 10-30% of entries
    remaining_length = max(0, length - delete_count)
    
    if remaining_length > 0:
        # Randomly choose indices to keep
        indices = sorted(random.sample(range(length), remaining_length))
        categorical_feature = categorical_feature[indices]
        numeric_feature = numeric_feature[indices]
    else:
        # If no data remains, create empty tensors of the right shape
        categorical_feature = categorical_feature[:0]
        numeric_feature = numeric_feature[:0]
    
    # Pad the remaining data to reach the pad_length
    pad_size = max(0, pad_length - remaining_length)
    if pad_size > 0:
        categorical_pad = torch.full((pad_size, *categorical_feature.shape[1:]), fill_value=-1)  # Assuming -1 is the padding ID for categorical features
        numeric_pad = torch.zeros((pad_size, *numeric_feature.shape[1:]))
        
        categorical_feature = torch.cat([categorical_feature, categorical_pad], dim=0)
        numeric_feature = torch.cat([numeric_feature, numeric_pad], dim=0)
    
    return categorical_feature, numeric_feature, min(length, pad_length)
    


def delete_and_pad_feature_only(feature, feature_length, delete_ratio, pad_length):
    """Specialized deletion and padding for single-feature cases.

    Args:
        feature (Tensor): The feature to be modified.
        feature_length (int): Original length of the feature.
        delete_ratio (float): Fraction of the feature to delete.
        pad_length (int): The length to pad the feature to.

    Returns:
        Tuple[Tensor, int]: The modified feature and its new length.
    """
    # Specific for cases where only one feature type (either categorical or numerical) is dealt with.
    delete_count = int(math.ceil(feature_length * delete_ratio))
    delete_indices = torch.randint(low=0, high=feature_length-1, size=(delete_count,))
    for idx in delete_indices.sort(descending=True)[0]:
        feature = torch.cat((feature[:, :idx], feature[:, idx+1:]), dim=1)
    padding_length = pad_length - feature.shape[1]
    feature = F.pad(feature, (0, padding_length), "constant", 0)
    feature_length -= delete_count
    return feature, feature_length


def replace_data(categorical_feature, numeric_feature, length):
    """Randomly replaces data points in the features.

    Args:
        categorical_feature (Tensor): The categorical feature tensor.
        numeric_feature (Tensor): The numeric feature tensor.
        length (int): Length of the features.

    """
    # Implementation remains mostly unchanged, focus on clearer naming and docstrings.
    num_replace = int(math.ceil(length * 0.1))
    replace_indices = random.sample(range(length-1), num_replace)
    for idx in replace_indices:
        if categorical_feature is not None:
            replacement_value = categorical_feature[:, idx + 1].clone()
            categorical_feature[:, idx] = replacement_value
        if numeric_feature is not None:
            replacement_value = numeric_feature[:, idx + 1].clone()
            numeric_feature[:, idx] = replacement_value
    return categorical_feature, numeric_feature, length



def sub_and_pad(categorical_feature, numeric_feature, length, pad_length):
    """Subsets and pads the data based on given lengths.

    Args:
        categorical_feature (Tensor): The categorical feature tensor.
        numeric_feature (Tensor): The numeric feature tensor.
        length (int): Original length of data features.
        pad_length (int): Length to pad the data to.

    Returns:
        tuple: The modified categorical feature, numeric feature, and the new length.
    """
    # The actual implementation will adjust according to clear specification of arguments.
    min_len = int(length * 0.5)
    max_len = int(length * 0.9)
    new_len = torch.randint(min_len, max_len + 1, (1,)).item()
    start_pos = torch.randint(0, length - new_len + 1, (1,)).item()
    end_pos = start_pos + new_len
    if categorical_feature is not None:
        categorical_feature = categorical_feature[:, start_pos:end_pos]
        categorical_feature = F.pad(categorical_feature, (0, pad_length - new_len), "constant", 0)
    if numeric_feature is not None:
        numeric_feature = numeric_feature[:, start_pos:end_pos]
        numeric_feature = F.pad(numeric_feature, (0, pad_length - new_len), "constant", 0)
    length = new_len
    return categorical_feature, numeric_feature, length