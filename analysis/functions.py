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


def normalization_between(data, a, b):
    """
    x` = (b - a) * (xi - min(x)) / (max(x) - min(x)) + a
    Args:
        data:
        a:
        b:

    Returns:

    """
    min_x = min(data)
    max_x = max(data)
    const = (b - a) / (max_x - min_x)
    return [(x - min_x) * const + a for x in data]


def find_latencies(datas, step, with_afferent=False):
    """

    Args:
        datas (list of lis):
            0 slices_max_time
            1 slices_max_value
            2 slices_min_time
            3 slices_min_value
        step:
        with_afferent:

    Returns:

    """
    latencies = []
    slice_numbers = len(datas[2])

    if not with_afferent:
        slice_indexes = range(slice_numbers)
        for slice_index in slice_indexes:
            slice_times = datas[2][slice_index]
            slice_values = datas[3][slice_index]
            # set latencies borders
            # ToDo test this algorithm
            if slice_index in slice_indexes[:int(slice_numbers / 6 * 2)]:
                # in the first two (4/8) slices the poly-answer everytime starts before middle of 25ms
                border_left = 0
                border_right = 25 / 2
            elif slice_index in slice_indexes[int(slice_numbers / 6 * 2):int(slice_numbers / 6 * 3)]:
                border_left = 20 / 3
                border_right = 20 / (3 / 2)
            elif slice_index == slice_indexes[-1]:
                # in the last slice the poly-answer everytime starts after middle of 25ms
                border_left = 25 / 2
                border_right = 20
            else:
                border_left = 5
                border_right = 20

            minimal_val_in_border = min([v for i, v in enumerate(slice_values) if (border_left / step) < slice_times[i] < (border_right / step)])
            index_of_val = slice_values.index(minimal_val_in_border)
            latencies.append(slice_times[index_of_val])
            '''
            elif slice_index == slice_indexes[-1]:
                # in the last slice the poly-answer everytime starts after middle of 25ms
                latencies.append([t for t in slice_times if t > (25 / 2 / step)][0])
            else:
                latencies.append(slice_times[0])
            '''

    else:
        # Neuron simulator variant
        latencies = list(map(lambda tup: tup[0][tup[1].index(min(tup[1]))], zip(datas[2], datas[3])))

    # errors checking
    if len(latencies) != slice_numbers:
        raise Exception("Latency list length is not equal to number of slices!")

    return latencies

def alex_latency(ees_indexes, mins):
    latencies = []
    for ees, minimal in zip(ees_indexes, mins):
        latencies.append(abs(ees - minimal[0]))
    return latencies

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




def find_mins_without_criteria(array):
    indexes = []
    min_elems = []
    recording_step = 0.25
    for index_elem in range(1, len(array) - 1):
        if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]) and array[index_elem] < -0.5:
            min_elems.append(index_elem * recording_step)
            indexes.append(index_elem)
    return min_elems, indexes


def read_neuron_data(path):
    """

    Args:
        path (str):
            path to the file
    Returns:
        list : data from file
    """
    with hdf5.File(path) as file:
        neuron_means = [data[:] for data in file.values()]
    return neuron_means


def read_nest_data(path):
    """
    # todo write description
    Args:
        path (str):
            path to the file
    Returns:
        list : data from file
    """
    with hdf5.File(path) as file:
        nest_means = [-data[:] for data in file.values()]
    return nest_means


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

    indexes = find_mins_without_criteria(data_stim)[1]
    data_RMG = raw_data_RMG[indexes[0]:indexes[-1]]
    # shift indexes to be normalized with data RMG (because a data was sliced)
    shift_by = indexes[0]
    shifted_indexes = [d - shift_by for d in indexes]
    return data_RMG, shifted_indexes
