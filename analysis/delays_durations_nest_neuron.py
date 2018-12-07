from analysis.dispersion import read_NEST_data
import numpy as np
import h5py as hdf5
from numpy import *
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pylab as plt
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.peaks_of_real_data_without_EES import delays, calc_durations
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


def find_mins_without_criteria(array):
    """

    Args:
        array:
            list
                data what is needed to find mins in
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
        if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]):
            min_elems.append(array[index_elem])
            indexes.append(index_elem)
    return min_elems, indexes


nest_dict = read_NEST_data\
    ('../../nest-data/21cms/40Hz 100inh/21cms_40Hz_100inh/sim_healthy_nest_extensor_eesF40_i100_s21cms_T.hdf5')
neuron_dict = read_NEURON_data\
    ('../../neuron-data/res3010/sim_healthy_neuron_extensor_eesF40_i100_s21cms_T.hdf5')
nest_list = []
neuron_list = []
nest_list_from_dict = list(nest_dict.values())
for i in range(len(nest_list_from_dict)):
    nest_list_tmp = []
    for j in range(len(nest_list_from_dict[i])):
        nest_list_tmp.append(nest_list_from_dict[i][j])
    nest_list.append(nest_list_tmp)
neuron_list_from_dict = list(neuron_dict.values())
for i in range(len(neuron_list_from_dict)):
    neuron_list_tmp = []
    for j in range(len(neuron_list_from_dict[i])):
        neuron_list_tmp.append(neuron_list_from_dict[i][j])
    neuron_list.append(neuron_list_tmp)
nest_means = list(map(lambda x: np.mean(x), zip(*nest_list)))
neuron_means = list(map(lambda x: np.mean(x), zip(*neuron_list)))
slices_begin_time_nest_from_ees = find_mins(nest_means, 12)[1]
slices_begin_time_neuron_from_ees = find_mins(neuron_means, -15)[1]   # -14 * 10 ** (-10) for 6 cm/s
num_of_dots = len(slices_begin_time_nest_from_ees)
slices_begin_time_nest_from_ees[0] = 37
step_nest = slices_begin_time_nest_from_ees[1] - slices_begin_time_nest_from_ees[0]
slices_begin_time_nest_from_ees.append(slices_begin_time_nest_from_ees[-1] + step_nest)
step_neuron = slices_begin_time_neuron_from_ees[1] - slices_begin_time_neuron_from_ees[0]
nest_list_all_runs = []
neuron_list_all_runs = []
offset = 0
slices_begin_time_nest_from_ees_for_clouds = slices_begin_time_nest_from_ees
slices_begin_time_nest_from_ees_for_clouds[-1] = len(nest_list[0])
while offset < num_of_dots:
    nest_list_one_run = []
    for run_index in range(len(nest_list)):
        nest_list_one_run_tmp = []
        for slice_index in range(slices_begin_time_nest_from_ees_for_clouds[offset],
                                 slices_begin_time_nest_from_ees_for_clouds[offset + 1]):
            nest_list_one_run_tmp.append(nest_list[run_index][slice_index])
        nest_list_one_run.append(nest_list_one_run_tmp)
    offset += 1
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
data_nest_all_runs = []
slices_begin_time_neuron_from_ees.append(len(neuron_list[0]))
for run in range(len(nest_list)):
    nest_list_all_runs.append(calc_max_min(slices_begin_time_nest_from_ees_for_clouds, nest_list[run], nestFD))
    neuron_list_all_runs.append(calc_max_min(slices_begin_time_neuron_from_ees, neuron_list[run], neuronFD))
nest_delays_all_runs = []
nest_durations_all_runs = []
for run in range(len(nest_list_all_runs)):
    nest_delays_all_runs.append(delays(nest_list_all_runs[run][0], nest_list_all_runs[run][2],
                                       nest_list_all_runs[run][3], 8, 62,
                                       'nest'))
    nest_durations_all_runs.append(calc_durations(nest_list_all_runs[run][0], nest_list_all_runs[run][2],
                                       nest_list_all_runs[run][3], 8, 62,
                                       'nest'))
neuron_delays_all_runs = []
neuron_durations_all_runs = []
for run in range(len(neuron_list_all_runs)):
    neuron_delays_all_runs.append(delays(neuron_list_all_runs[run][0], neuron_list_all_runs[run][2],
                                         neuron_list_all_runs[run][3], -15, -3.6, 'neuron'))
    neuron_durations_all_runs.append(calc_durations(neuron_list_all_runs[run][0], neuron_list_all_runs[run][2],
                                         neuron_list_all_runs[run][3], -15, -3.6, 'neuron'))
delays_mins_nest_all_runs = []
for sl in range(len(nest_delays_all_runs[0][1])):
    delays_mins_nest_all_runs_tmp = []
    for run in range(len(nest_delays_all_runs)):
        delays_mins_nest_all_runs_tmp.append(nest_delays_all_runs[run][1][sl])
    delays_mins_nest_all_runs.append(delays_mins_nest_all_runs_tmp)
durations_mins_nest_all_runs = []
for sl in range(len(nest_durations_all_runs[0][1])):
    durations_mins_nest_all_runs_tmp = []
    for run in range(len(nest_durations_all_runs)):
        durations_mins_nest_all_runs_tmp.append(nest_durations_all_runs[run][1][sl])
    durations_mins_nest_all_runs.append(durations_mins_nest_all_runs_tmp)
delays_mins_neuron_all_runs = []
for sl in range(len(neuron_delays_all_runs[0][1])):
    delays_mins_neuron_all_runs_tmp = []
    for run in range(len(neuron_delays_all_runs)):
        delays_mins_neuron_all_runs_tmp.append(neuron_delays_all_runs[run][1][sl])
    delays_mins_neuron_all_runs.append(delays_mins_neuron_all_runs_tmp)
durations_mins_neuron_all_runs = []
for sl in range(len(neuron_durations_all_runs[0][1])):
    durations_mins_neuron_all_runs_tmp = []
    for run in range(len(neuron_durations_all_runs)):
        durations_mins_neuron_all_runs_tmp.append(neuron_durations_all_runs[run][1][sl])
    durations_mins_neuron_all_runs.append(durations_mins_neuron_all_runs_tmp)
max_min_delays_nest = delays(data_nest[0], data_nest[2], data_nest[3], 12, 57, 'nest')
max_min_delays_neuron = delays(data_neuron[0], data_neuron[2], data_neuron[3], -15, -3.6, 'neuron')
for i in range(len(nest_list_all_runs)):
    delays_tmp = []
max_min_durations_nest = calc_durations(data_nest[0], data_nest[2], data_nest[3], 12, 57, 'nest')
max_min_durations_neuron = calc_durations(data_neuron[0], data_neuron[2], data_neuron[3], -15, -3.6, 'neuron')
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
times_nest = []
times_neuron = []
durations = []
for dot in range(len(delays_mins_nest_all_runs)):
    times_tmp = []
    durations_tmp = []
    for l in range(len(delays_mins_nest_all_runs[dot])):
        times_tmp.append(dot)
        durations_tmp.append(min_durations_nest[dot])
    times_nest.append(times_tmp)
    durations.append(durations_tmp)
for dot in range(len(delays_mins_nest_all_runs)):
    times_tmp = []
    for l in range(len(delays_mins_nest_all_runs[dot])):
        times_tmp.append(dot + 0.5)
    times_neuron.append(times_tmp)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, min_delays_nest, min_durations_nest, color='orange')
ax.plot(time, min_delays_nest, min_durations_nest, '.', lw=0.5, color='r', markersize=5)
ax.plot(time, min_delays_neuron, min_durations_neuron, color='blue')
ax.plot(time, min_delays_neuron, min_durations_neuron, '.', lw=0.5, color='r', markersize=5)
nest_y = max(delays_mins_nest_all_runs)
nest_z = max(durations_mins_nest_all_runs)
verts_nest = []
verts_times = []
verts_delays = []
verts_durations = []
for run in range(len(times_nest[0])):
    verts_times_tmp = []
    verts_delays_tmp = []
    verts_durations_tmp = []
    for dot in range(len(times_nest)):
        verts_times_tmp.append(times_nest[dot][run])
        verts_delays_tmp.append(delays_mins_nest_all_runs[dot][run])
        verts_durations_tmp.append(durations_mins_nest_all_runs[dot][run])
    verts_times.append(verts_times_tmp)
    verts_delays.append(verts_delays_tmp)
    verts_durations.append(verts_durations_tmp)
for run in range(len(verts_times)):
    for i in range(1):
        verts_tmp = []
        verts_tmp.append(verts_times[run])
        verts_tmp.append(verts_delays[run])
        verts_tmp.append(verts_durations[run])
    verts_nest.append(verts_tmp)
nest = []
for sl in range(len(delays_mins_nest_all_runs)):
    nest_sl = []
    for dot in range(len(delays_mins_nest_all_runs[0])):
        one_dot = []
        one_dot.append(delays_mins_nest_all_runs[sl][dot])
        one_dot.append(durations_mins_nest_all_runs[sl][dot])
        nest_sl.append(one_dot)
    nest.append(nest_sl)
neuron = []
for sl in range(len(delays_mins_neuron_all_runs)):
    neuron_sl = []
    for dot in range(len(delays_mins_neuron_all_runs[0])):
        one_dot = []
        one_dot.append(delays_mins_neuron_all_runs[sl][dot])
        one_dot.append(durations_mins_neuron_all_runs[sl][dot])
        neuron_sl.append(one_dot)
    neuron.append(neuron_sl)


def rotate(A,B,C):
  return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])


def grahamscan(A):
  n = len(A)
  P = []
  for i in range(n):
    P.append(i)
  for i in range(1,n):
    if A[P[i]][0]<A[P[0]][0]:
      P[i], P[0] = P[0], P[i]
  for i in range(2, n):
    j = i
    while j>1 and (rotate(A[P[0]],A[P[j-1]],A[P[j]])<0):
      P[j], P[j-1] = P[j-1], P[j]
      j -= 1
  S = [P[0],P[1]]
  for i in range(2,n):
    while rotate(A[S[-2]],A[S[-1]],A[P[i]])<0:
      del S[-1]
    S.append(P[i])
  return S


convex_nests = []
convex_neurons = []
for sl in range(len(nest)):
    convex_nest = grahamscan(nest[sl])
    convex_nests.append(convex_nest)
for sl in range(len(neuron)):
    convex_neuron = grahamscan(neuron[sl])
    convex_neurons.append(convex_neuron)
delays_convex_nest = []
durations_convex_nest = []
for sl in range(len(convex_nests)):
    delays_convex_nest_tmp = []
    durations_convex_nest_tmp = []
    for i in convex_nests[sl]:
        delays_convex_nest_tmp.append(delays_mins_nest_all_runs[sl][i])
        durations_convex_nest_tmp.append(durations_mins_nest_all_runs[sl][i])
    delays_convex_nest.append(delays_convex_nest_tmp)
    durations_convex_nest.append(durations_convex_nest_tmp)
delays_convex_neuron = []
durations_convex_neuron = []
for sl in range(len(convex_neurons)):
    delays_convex_neuron_tmp = []
    durations_convex_neuron_tmp = []
    for i in convex_neurons[sl]:
        delays_convex_neuron_tmp.append(delays_mins_neuron_all_runs[sl][i])
        durations_convex_neuron_tmp.append(durations_mins_neuron_all_runs[sl][i])
    delays_convex_neuron.append(delays_convex_neuron_tmp)
    durations_convex_neuron.append(durations_convex_neuron_tmp)
lens_nest = []
for dot in range(len(delays_mins_nest_all_runs)):
    lens_nest.append(len(delays_convex_nest[dot]))
times_convex_nest = []
for i in range(len(lens_nest)):
    times_convex_nest_tmp = []
    for j in range(lens_nest[i]):
        times_convex_nest_tmp.append(i)
    times_convex_nest.append(times_convex_nest_tmp)
lens_neuron = []
for dot in range(len(delays_mins_neuron_all_runs)):
    lens_neuron.append(len(delays_convex_neuron[dot]))
times_convex_nest = []
for i in range(len(lens_nest)):
    times_convex_nest_tmp = []
    for j in range(lens_nest[i]):
        times_convex_nest_tmp.append(i)
    times_convex_nest.append(times_convex_nest_tmp)
times_convex_neuron = []
for i in range(len(lens_neuron)):
    times_convex_neuron_tmp = []
    for j in range(lens_neuron[i]):
        times_convex_neuron_tmp.append(i)
    times_convex_neuron.append(times_convex_neuron_tmp)
for dot in range(len(delays_mins_nest_all_runs)):
    x_nest = times_convex_nest[dot] + [times_convex_nest[dot][0]]
    y_nest = delays_convex_nest[dot] + [delays_convex_nest[dot][0]]
    z_nest = durations_convex_nest[dot] + [durations_convex_nest[dot][0]]

    x_neuron = times_convex_neuron[dot] + [times_convex_neuron[dot][0]]
    y_neuron = delays_convex_neuron[dot] + [delays_convex_neuron[dot][0]]
    z_neuron = durations_convex_neuron[dot] + [durations_convex_neuron[dot][0]]

    ax.add_collection3d(plt.fill_between(y_nest, z_nest, min(z_nest), color='green', alpha=0.3, label="filled plot"),
                        x_nest[dot], zdir='x')
    ax.add_collection3d(plt.fill_between(y_neuron, z_neuron, min(z_neuron), color='purple', alpha=0.3,
                                         label="filled plot"),
                        x_neuron[dot], zdir='x')

    ax.plot(times_convex_nest[dot],
            delays_convex_nest[dot],
            durations_convex_nest[dot],
            color='green',
            alpha=0.3, label = 'nest')
for dot in range(len(delays_mins_neuron_all_runs)):
    ax.plot(times_convex_neuron[dot], delays_convex_neuron[dot], durations_convex_neuron[dot], color='purple',
            alpha=0.3)
nest_clouds_patch = mpatches.Patch(color='green', label='nest clouds')
neuron_clouds_patch = mpatches.Patch(color='purple', label='neuron clouds')
neuron_patches = mpatches.Patch(color='blue', label='neuron')
nest_patches = mpatches.Patch(color='orange', label='nest')
ax.set_xlabel("Slice number")
ax.set_ylabel("Delays ms")
ax.set_zlabel("Durations ms")
ax.set_title("Slice - Delay - Duration")
plt.show()