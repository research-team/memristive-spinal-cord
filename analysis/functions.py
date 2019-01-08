import h5py as hdf5
import numpy as np
import csv


def normalization(list_of_data_to_normalize, max_value, min_value):
    """

    Args:
        list_of_data_to_normalize: list
            data that is needd to be normalized
        max_value: max value of normalized data
        min_value: min value of normalized data

    Returns: list
        normalized data

    """
    fact_max = max(list_of_data_to_normalize)
    fact_min = min(list_of_data_to_normalize)
    x_max = fact_max / max_value
    x_min = fact_min / min_value
    scale = (x_max + x_min) / 2
    normal_data = []
    for i in range(len(list_of_data_to_normalize)):
        normal_data.append(list_of_data_to_normalize[i] / scale)
    return normal_data


def normalization_in_bounds(list_of_data_to_normalize, min_value):
    fact_min = min(list_of_data_to_normalize)
    scale = min_value / fact_min
    normal_data = []
    for i in range(len(list_of_data_to_normalize)):
        normal_data.append(list_of_data_to_normalize[i] * scale)
    return normal_data


def normalization_zero_one(list_of_data_to_normalize):
    """

        Args:
            list_of_data_to_normalize: list
                data that is needd to be normalized
            max_value: max value of normalized data
            min_value: min value of normalized data

        Returns: list
            normalized data

        """
    fact_max = max(list_of_data_to_normalize)
    fact_min = min(list_of_data_to_normalize)
    normal_data = []
    for i in range(len(list_of_data_to_normalize)):
        normal_data.append((list_of_data_to_normalize[i] - fact_min) / (fact_max - fact_min))
    return normal_data


def find_mins(array, matching_criteria):
    """

    Args:
        array:
            list
                data what is needed to find mins in
        matching_criteria:
            int or float
                number less than which min peak should be to be considered as the start of new slice

    Returns:
        min_elems:
            list
                values of the starts of new slice
        indexes:
            list
                indexes of the starts of new slice

    """
    min_elems = []
    indexes = []
    for index_elem in range(1, len(array) - 1):
        if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]) and array[index_elem] < \
                matching_criteria:
            min_elems.append(array[index_elem])
            indexes.append(index_elem)
    return min_elems, indexes


neuron_dict = {}


def find_mins_without_criteria(array):
    recording_step = 0.25
    min_elems = []
    indexes = []
    for index_elem in range(1, len(array) - 1):
        if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]) and array[index_elem] < -0.5:
            min_elems.append(index_elem * recording_step)
            indexes.append(index_elem)
    return min_elems, indexes


def read_NEURON_data(path):
    """

    Args:
        path: string
            path to file

    Returns:
        dict
            data from file

    """
    with hdf5.File(path, 'r') as f:
        for test_name, test_values in f.items():
            neuron_dict[test_name] = test_values[:]
    return neuron_dict


nest_means_dict = {}


def read_NEST_data(path):
    with hdf5.File(path) as f:
        for test_name, test_values in f.items():
            nest_means_dict[test_name] = -test_values[:]
    return nest_means_dict


def list_to_dict(inputing_dict):
    returning_list = []
    list_from_dict = list(inputing_dict.values())
    # print("list_from_dict = ", list_from_dict)
    for i in range(len(list_from_dict)):
        list_tmp = []
        for j in range(len(list_from_dict[i])):
            list_tmp.append(list_from_dict[i][j])
        returning_list.append(list_tmp)
    return returning_list


def read_bio_data(path):
    with open(path) as file:
        for i in range(6):
            file.readline()
        reader = csv.reader(file, delimiter='\t')
        grouped_elements_by_column = list(zip(*reader))

        raw_data_RMG = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[2]]
        data_stim = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[7]]
    stimulations = find_mins_without_criteria(data_stim)[:-1][0]
    indexes = find_mins_without_criteria(data_stim)[1]
    data_RMG = []
    for i in range(indexes[0], indexes[-1]):
        data_RMG.append(raw_data_RMG[i])
    return data_RMG, indexes