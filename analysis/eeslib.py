import logging
import numpy

def slice_ees(data_array, slicing_index = 'Stim', data_index = 'RMG ', epsilon = .001) :
    logging.debug('Slicing')
    max_stim = max(data_array[slicing_index])
    list_of_maxs = [i for i, x in enumerate(data_array[slicing_index]) if x > max_stim-epsilon]
    logging.debug('number of maxs ' + str(len(list_of_maxs)))
    res = numpy.split(data_array[data_index], list_of_maxs)
    logging.debug('number of slices ' + str(len(res)))
    return res