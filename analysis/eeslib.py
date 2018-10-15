import logging
import numpy


def slice_ees(data_array, slicing_index = 'Stim', data_index = 'RMG ', epsilon = .001) :
    """
    Returns sliced array of voltages from raw data of myograms from matlab file
    Parameters
    ----------
    data_array: dict
        the dict of the raw voltages of myograms and EESes
    slicing_index: str
        the string that indicates the stimulation dictionary (EESes)
    data_index: str
        the sting that indicates the data dictionary (voltages)
    epsilon: float
        the float to identify the maximums of EESes

    Returns
    -------
    list[array]
        the list of slices (array of voltages)
    """
    tick_rate = int(data_array['tickrate'][0][0])
    logging.debug('Slicing')
    max_stim = max(data_array[slicing_index])
    list_of_maxs = [i for i, x in enumerate(data_array[slicing_index]) if x > max_stim-epsilon]
    logging.debug('number of maxs ' + str(len(list_of_maxs)))
    res = numpy.split(data_array[data_index], list_of_maxs)
    logging.debug('number of slices ' + str(len(res)))
    return res

