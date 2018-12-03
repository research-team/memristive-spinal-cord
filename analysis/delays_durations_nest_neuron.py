from analysis.dispersion import read_NEST_data
import numpy as np
import h5py as hdf5
from matplotlib import pylab as plt
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.peaks_of_real_data_without_EES import delays, calc_durations
from mpl_toolkits.mplot3d import Axes3D
neuron_dict = {}


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


nest_dict = read_NEST_data\
    ('../../nest-data/6cms/40 Hz/sim_healthy_nest_extensor_eesF40_i100_s6cms_T.hdf5')
neuron_dict = read_NEURON_data\
    ('../../neuron-data/res3110/sim_healthy_neuron_extensor_eesF40_i100_s6cms_T.hdf5')
nest_list = []
neuron_list = []
nest_list = list(nest_dict.values())
neuron_list = list(neuron_dict.values())
nest_means = list(map(lambda x: np.mean(x), zip(*nest_list)))
neuron_means = list(map(lambda x: np.mean(x), zip(*neuron_list)))
slices_begin_time_nest_from_ees = find_mins(nest_means, 12)[1]
slices_begin_time_neuron_from_ees = find_mins(neuron_means, -15)[1]   # -14 * 10 ** (-10) for 6 cm/s
print("slices_begin_time_nest_from_ees = ", slices_begin_time_nest_from_ees)
slices_begin_time_nest_from_ees[0] = 37
step_nest = slices_begin_time_nest_from_ees[1] - slices_begin_time_nest_from_ees[0]
slices_begin_time_nest_from_ees.append(slices_begin_time_nest_from_ees[-1] + step_nest)
print("slices_begin_time_nest_from_ees = ", slices_begin_time_nest_from_ees)
step_neuron = slices_begin_time_neuron_from_ees[1] - slices_begin_time_neuron_from_ees[0]
slices_begin_time_nest = []
slices_begin_time_neuron = []
offset = 0
for i in range(len(slices_begin_time_nest_from_ees) + 1):
    slices_begin_time_nest.append(offset)
    offset += step_nest
offset = 0
for i in range(len(slices_begin_time_neuron_from_ees) + 1):
    slices_begin_time_neuron.append(offset)
    offset += step_neuron
nestFD = 0.1
neuronFD = 0.025
data_nest = calc_max_min(slices_begin_time_nest_from_ees, nest_means, nestFD)
data_neuron = calc_max_min(slices_begin_time_neuron, neuron_means, neuronFD)
print("slices_max_time = ", data_nest[0])
print("slices_max_value = ", data_nest[1])
print("slices_min_time = ", data_nest[2])
print("slices_min_value = ", data_nest[3])
max_min_delays_nest = delays(data_nest[0], data_nest[2], data_nest[3], 12, 57, 'nest')
max_min_delays_neuron = delays(data_neuron[0], data_neuron[2], data_neuron[3], -15, -8, 'neuron')
print(len(max_min_delays_nest[1]), "min_delays_nest = ", max_min_delays_nest[1])
max_min_durations_nest = calc_durations(data_nest[0], data_nest[2], data_nest[3], 12, 57, 'nest')
max_min_durations_neuron = calc_durations(data_neuron[0], data_neuron[2], data_neuron[3], -15, -8, 'neuron')
print( len(max_min_durations_nest[1]), "min_durations_nest = ", max_min_durations_nest[1])
ticks = []
labels = []
max_delays_delta = []
min_delays_delta = []
max_durations_delta = []
min_durations_delta = []

ticks = []
labels = []
for i in range(0, len(neuron_means), 300):
    ticks.append(i)
    labels.append(i * 0.1)
max_delays_nest = max_min_delays_nest[0]
max_durations_nest = max_min_durations_nest[0]
min_delays_nest = max_min_delays_nest[1]
min_durations_nest = max_min_durations_nest[1]
max_delays_neuron = max_min_delays_neuron[0]
max_durations_neuron = max_min_durations_neuron[0]
min_delays_neuron = max_min_delays_neuron[1]
min_durations_neuron = max_min_durations_neuron[1]
time = []
for i in range(len(min_delays_nest)):
    time.append(i)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, min_delays_nest, min_durations_nest, label='nest')
ax.plot(time, min_delays_nest, min_durations_nest, '.', lw=0.5, color='r', markersize=5)
ax.plot(time, min_delays_neuron, min_durations_neuron, label='neuron')
ax.plot(time, min_delays_neuron, min_durations_neuron, '.', lw=0.5, color='r', markersize=5)
ax.set_xlabel("Slice number")
ax.set_ylabel("Delays ms")
ax.set_zlabel("Durations ms")
ax.set_title("Slice - Delay - Duration")
plt.legend()
plt.show()