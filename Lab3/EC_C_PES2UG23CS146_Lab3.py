#lab.py
import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray):
    """
    Calculate the entropy of the entire dataset
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    if data.shape[0] == 0:
        return 0.0

    target_column = data[:, -1]
    unique_classes, counts = np.unique(target_column, return_counts=True)
    total_samples = data.shape[0]
    probabilities = counts / total_samples

    entropy = 0.0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    return entropy

def get_avg_info_of_attribute(data: np.ndarray, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v
    """
    if data.shape[0] == 0 or attribute < 0 or attribute >= data.shape[1] - 1:
        return 0.0

    attribute_column = data[:, attribute]
    total_samples = data.shape[0]
    unique_values = np.unique(attribute_column)

    avg_info = 0.0

    for value in unique_values:
        mask = attribute_column == value
        subset = data[mask]
        weight = subset.shape[0] / total_samples

        if subset.shape[0] > 0:
            subset_entropy = get_entropy_of_dataset(subset)
            avg_info += weight * subset_entropy

    return avg_info

def get_information_gain(data: np.ndarray, attribute: int):
    """
    Calculate Information Gain for an attribute
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)
    """
    if data.shape[0] == 0:
        return 0.0

    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    information_gain = dataset_entropy - avg_info

    return round(information_gain, 4)

def get_selected_attribute(data: np.ndarray):
    """
    Select the best attribute based on highest information gain
    Return a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)
    """
    if data.shape[0] == 0 or data.shape[1] <= 1:
        return ({}, -1)

    num_attributes = data.shape[1] - 1
    gain_dictionary = {}

    for i in range(num_attributes):
        gain_dictionary[i] = get_information_gain(data, i)

    if not gain_dictionary:
        return ({}, -1)

    selected_attribute_index = max(gain_dictionary, key=gain_dictionary.get)

    return (gain_dictionary, selected_attribute_index)
