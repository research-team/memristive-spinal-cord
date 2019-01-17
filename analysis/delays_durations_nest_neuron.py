from analysis.functions import read_neuron_data, read_nest_data, read_bio_data, find_latencies, \
	normalization, read_bio_hdf5, find_fliers
import numpy as np
# import h5py as hdf5
# from numpy import *
# import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt
from analysis.functions import calc_max_min
# from analysis.peaks_of_real_data_without_EES import delays, calc_durations
import matplotlib.patches as mpatches
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from analysis.histogram_lat_amp import bio_process, sim_process, calc_amplitudes, debug
import statistics
import copy
neuron_dict = {}


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


k_min_time = 2
k_min_val = 3


def find_ees_indexes(stim_indexes, datas):
	"""
	Function for finding the indexes of the EES mono-answer in borders formed by stimulations time
	Args:
		stim_indexes (list):
			indexes of the EES stimulations
		datas (list of list):
			includes min/max times and min/max values
	Returns:
		list: global indexes of the EES mono-answers
	"""
	ees_indexes = []
	for slice_index in range(len(stim_indexes)):
		min_values = datas[k_min_val][slice_index]
		min_times = datas[k_min_time][slice_index]
		# EES peak is the minimal one
		ees_value_index = min_values.index(min(min_values))
		ees_indexes.append(stim_indexes[slice_index] + min_times[ees_value_index])
	return ees_indexes


"""def process(data):
	sim_stim_indexes = list(range(0, len(data), int(25 / sim_step)))
	# the steps are the same as above
	sim_datas = calc_max_min(sim_stim_indexes, data, sim_step)
	sim_ees_indexes = find_ees_indexes(sim_stim_indexes, sim_datas)
	norm_nest_means = normalization(data, zero_relative=True)
	sim_datas = calc_max_min(sim_ees_indexes, norm_nest_means,
	                         sim_step, remove_micropeaks=True)
	sim_lat = find_latencies(sim_datas, sim_step, norm_to_ms=True)
	sim_amp = calc_amplitudes(sim_datas, sim_lat)
	return sim_lat, sim_amp
"""

k_bio_volt = 0
k_bio_stim = 1
bio_step = 0.25
sim_step = 0.025
gpu_step = 0.1
neuron_list = read_neuron_data('../../neuron-data/sim_healthy_neuron_extensor_eesF40_i100_s21cms_T_100runs.hdf5')
# print("neuron_tests = ", type(neuron_tests))
nest_list = read_nest_data('../../nest-data/21cms/extensor_21cms_40Hz_100inh.hdf5')
gpu = read_nest_data('../../GPU_extensor_eesF40_inh100_s21cms_T.hdf5')
bio = read_bio_data('../bio-data/3_0.91 volts-Rat-16_5-09-2017_RMG_13m-min_one_step.txt')
d1 = read_bio_hdf5('1_new_bio_1.hdf5')
d2 = read_bio_hdf5('1_new_bio_2.hdf5')
d3 = read_bio_hdf5('1_new_bio_3.hdf5')
d4 = read_bio_hdf5('1_new_bio_4.hdf5')
d5 = read_bio_hdf5('1_new_bio_5.hdf5')
bio_data = [d1[0], d2[0], d3[0], d4[0], d5[0]]

slice_numbers = int(len(neuron_list[0]) * sim_step // 25)
neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
print("neuron_means = ", neuron_means)
nest_means = list(map(lambda voltages: np.mean(voltages), zip(*nest_list)))
print("nest_means = ", nest_means)
gpu_means = list(map(lambda voltages: np.mean(voltages), zip(*gpu)))
print("gpu = ", gpu[0])
print("gpu_means = ", gpu_means)

neuron_means_lat = sim_process(neuron_means, sim_step)[0]
neuron_means_amp = sim_process(neuron_means, sim_step)[1]
nest_means_lat = sim_process(nest_means, sim_step)[0]
nest_means_amp = sim_process(nest_means, sim_step)[1]

gpu_means_lat = sim_process(gpu_means, gpu_step)[0]
gpu_means_amp = sim_process(gpu_means, gpu_step)[1]
print("gpu_means_amp = ", gpu_means_amp)

bio_lat = bio_process(bio, 6)[0]
bio_amp = bio_process(bio, 6)[1]
print("bio_lat = ", bio_lat)
print("bio_amp = ", bio_amp)
print(len(neuron_means_lat), neuron_means_lat)
print(len(neuron_means_amp), neuron_means_amp)
sim_stim_indexes = list(range(0, len(nest_means), int(25 / 0.025)))

# debug(nest_means, nest_means_datas, sim_stim_indexes, nest_ees_indexes, nest_means_lat, sim_step)
# debug(neuron_means, neuron_means_datas, sim_stim_indexes, neuron_ees_indexes, neuron_means_lat, sim_step)
nest_lat = []
nest_amp = []
for test_data in nest_list:
	nest_lat_tmp, nest_amp_tmp = sim_process(test_data, sim_step)
	nest_lat.append(nest_lat_tmp)
	nest_amp.append(nest_amp_tmp)

neuron_lat = []
neuron_amp = []
for test_data in neuron_list:
	neuron_lat_tmp, neuron_amp_tmp = sim_process(test_data, sim_step)
	neuron_lat.append(neuron_lat_tmp)
	neuron_amp.append(neuron_amp_tmp)
gpu_lat = []
gpu_amp = []
for test_data in gpu:
	gpu_lat_tmp, gpu_amp_tmp = sim_process(test_data, gpu_step)
	gpu_lat.append(gpu_lat_tmp)
	gpu_amp.append(gpu_amp_tmp)

latencies_all_runs_neuron = []
latencies_all_runs_nest = []
latencies_all_runs_gpu = []
for sl in range(len(neuron_lat[0])):
	# print("sl = ", sl)
	latencies_all_runs_neuron_tmp = []
	for dot in range(len(neuron_lat)):
		# print("dot = ", dot)
		latencies_all_runs_neuron_tmp.append(neuron_lat[dot][sl])
	latencies_all_runs_neuron.append(latencies_all_runs_neuron_tmp)
print("latencies_all_runs_neuron = ", len(latencies_all_runs_neuron))

for sl in range(len(nest_lat[0])):
	# print("sl = ", sl)
	latencies_all_runs_nest_tmp = []
	for dot in range(len(nest_lat)):
		# print("dot = ", dot)
		latencies_all_runs_nest_tmp.append(nest_lat[dot][sl])
	latencies_all_runs_nest.append(latencies_all_runs_nest_tmp)
for sl in range(len(gpu_lat[0])):
	# print("sl = ", sl)
	latencies_all_runs_gpu_tmp = []
	for dot in range(len(gpu_lat)):
		# print("dot = ", dot)
		latencies_all_runs_gpu_tmp.append(gpu_lat[dot][sl])
	latencies_all_runs_gpu.append(latencies_all_runs_gpu_tmp)
# print("latencies_all_runs_nest = ", len(latencies_all_runs_nest[0]))

amplitudes_all_runs_nest = []
amplitudes_all_runs_neuron = []
amplitudes_all_runs_gpu = []

for sl in range(len(neuron_amp[0])):
	# print("sl = ", sl)
	amplitudes_all_runs_neuron_tmp = []
	for dot in range(len(neuron_amp)):
		# print("dot = ", dot)
		amplitudes_all_runs_neuron_tmp.append(neuron_amp[dot][sl])
	amplitudes_all_runs_neuron.append(amplitudes_all_runs_neuron_tmp)
print("amplitudes_all_runs_neuron = ", amplitudes_all_runs_neuron)

for sl in range(len(nest_amp[0])):
	# print("sl = ", sl)
	amplitudes_all_runs_nest_tmp = []
	for dot in range(len(nest_amp)):
		# print("dot = ", dot)
		amplitudes_all_runs_nest_tmp.append(nest_amp[dot][sl])
	amplitudes_all_runs_nest.append(amplitudes_all_runs_nest_tmp)

for sl in range(len(gpu_amp[0])):
	# print("sl = ", sl)
	amplitudes_all_runs_gpu_tmp = []
	for dot in range(len(gpu_amp)):
		# print("dot = ", dot)
		amplitudes_all_runs_gpu_tmp.append(gpu_amp[dot][sl])
	amplitudes_all_runs_gpu.append(amplitudes_all_runs_gpu_tmp)
# print("amplitudes_all_runs_nest = ", amplitudes_all_runs_nest)

proceed_neuron = find_fliers(amplitudes_all_runs_neuron, latencies_all_runs_neuron)
corr_latencies_all_runs_neuron = proceed_neuron[0]
corr_amplitudes_all_runs_neuron = proceed_neuron[1]
fliers_neuron = proceed_neuron[2]
fliers_latencies_neuron_values = proceed_neuron[3]
fliers_amplitudes_neuron_values = proceed_neuron[4]

proceed_nest = find_fliers(amplitudes_all_runs_nest, latencies_all_runs_nest)
corr_latencies_all_runs_nest = proceed_nest[0]
corr_amplitudes_all_runs_nest = proceed_nest[1]
fliers_nest = proceed_nest[2]
fliers_latencies_nest_values = proceed_nest[3]
fliers_amplitudes_nest_values = proceed_nest[4]
print("corr_latencies_all_runs_nest = ", len(corr_latencies_all_runs_nest[5]))
print("fliers_latencies_nest_values = ", fliers_latencies_nest_values)
print("fliers_amplitudes_nest_values = ", fliers_amplitudes_nest_values)

proceed_gpu = find_fliers(amplitudes_all_runs_gpu, latencies_all_runs_gpu)
corr_latencies_all_runs_gpu = proceed_gpu[0]
corr_amplitudes_all_runs_gpu = proceed_gpu[1]
fliers_gpu = proceed_gpu[2]
fliers_latencies_gpu_values = proceed_gpu[3]
fliers_amplitudes_gpu_values = proceed_gpu[4]

time = []
time_neuron = []
time_gpu = []
for i in range(len(nest_means_amp)):
	time.append(i)
	time_neuron.append(i + 0.5)
	time_gpu.append(i + 0.25)
print("time = ", time)
print("time_neuron = ", time_neuron)

times_nest = []
times_neuron = []
times_gpu = []

old_times_neuron = []
old_times_nest = []
old_times_gpu = []

for dot in range(len(corr_latencies_all_runs_nest)):
	times_tmp = []
	for l in range(len(corr_latencies_all_runs_nest[dot])):
		times_tmp.append(dot)
	times_nest.append(times_tmp)
print("times_nest = ", len(times_nest[0]))

for dot in range(len(corr_latencies_all_runs_neuron)):
	times_tmp = []
	for l in range(len(corr_latencies_all_runs_neuron[dot])):
		times_tmp.append(dot + 0.5)
	times_neuron.append(times_tmp)
print("times_neuron = ", len(times_neuron[0]))

for dot in range(len(corr_latencies_all_runs_gpu)):
	times_tmp = []
	for l in range(len(corr_latencies_all_runs_gpu[dot])):
		times_tmp.append(dot + 0.25)
	times_gpu.append(times_tmp)
print("times_gpu = ", len(times_gpu[0]))

for dot in range(len(corr_latencies_all_runs_neuron)):
	times_tmp = []
	for l in range(len(fliers_neuron[dot])):
		times_tmp.append(dot + 0.5)
	old_times_neuron.append(times_tmp)

for dot in range(len(corr_latencies_all_runs_nest)):
	times_tmp = []
	for l in range(len(fliers_nest[dot])):
		times_tmp.append(dot)
	old_times_nest.append(times_tmp)

for dot in range(len(corr_latencies_all_runs_gpu)):
	times_tmp = []
	for l in range(len(fliers_gpu[dot])):
		times_tmp.append(dot + 0.25)
	old_times_gpu.append(times_tmp)
print("old_times_gpu = ", old_times_gpu)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, nest_means_lat, nest_means_amp, color='green')
ax.plot(time, nest_means_lat, nest_means_amp, '.', lw=0.5, color='r', markersize=5)
ax.plot(time_neuron, neuron_means_lat, neuron_means_amp, color='purple')
ax.plot(time_neuron, neuron_means_lat, neuron_means_amp, '.', lw=0.5, color='r', markersize=5)

ax.plot(time_gpu, gpu_means_lat, gpu_means_amp, color='orange')
ax.plot(time_gpu, gpu_means_lat, gpu_means_amp, '.', lw=0.5, color='r', markersize=5)
ax.plot(time, bio_lat, bio_amp, color='black')
ax.plot(time, bio_lat, bio_amp, '.', lw=0.5, color='r', markersize=5)
nest_y = max(corr_latencies_all_runs_nest)
nest_z = max(corr_amplitudes_all_runs_nest)
nest = []
for sl in range(len(corr_latencies_all_runs_nest)):
	nest_sl = []
	for dot in range(len(corr_latencies_all_runs_nest[sl])):
		one_dot = []
		one_dot.append(corr_latencies_all_runs_nest[sl][dot])
		one_dot.append(corr_amplitudes_all_runs_nest[sl][dot])
		nest_sl.append(one_dot)
	nest.append(nest_sl)
neuron = []
for sl in range(len(corr_latencies_all_runs_neuron)):
	# print("sl = ", sl)
	neuron_sl = []
	for dot in range(len(corr_latencies_all_runs_neuron[sl])):
		one_dot = []
		one_dot.append(corr_latencies_all_runs_neuron[sl][dot])
		one_dot.append(corr_amplitudes_all_runs_neuron[sl][dot])
		neuron_sl.append(one_dot)
	neuron.append(neuron_sl)

gpu = []
for sl in range(len(corr_latencies_all_runs_gpu)):
	gpu_sl = []
	for dot in range(len(corr_latencies_all_runs_gpu[sl])):
		one_dot = []
		one_dot.append(corr_latencies_all_runs_gpu[sl][dot])
		one_dot.append(corr_amplitudes_all_runs_gpu[sl][dot])
		gpu_sl.append(one_dot)
	gpu.append(gpu_sl)


def rotate(A, B, C):
	return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])


def grahamscan(A):
	n = len(A)
	P = []
	for i in range(n):
		P.append(i)
	for i in range(1, n):
		if A[P[i]][0] < A[P[0]][0]:
			P[i], P[0] = P[0], P[i]
	for i in range(2, n):
		j = i
		while j > 1 and (rotate(A[P[0]], A[P[j - 1]], A[P[j]]) < 0):
			P[j], P[j - 1] = P[j - 1], P[j]
			j -= 1
	S = [P[0], P[1]]
	for i in range(2, n):
		while rotate(A[S[-2]], A[S[-1]], A[P[i]]) < 0:
			del S[-1]
		S.append(P[i])
	return S


convex_nests = []
convex_neurons = []
convex_gpus = []

for sl in range(len(nest)):
	convex_nest = grahamscan(nest[sl])
	convex_nests.append(convex_nest)

for sl in range(len(neuron)):
	convex_neuron = grahamscan(neuron[sl])
	convex_neurons.append(convex_neuron)

for sl in range(len(gpu)):
	convex_gpu = grahamscan(gpu[sl])
	convex_gpus.append(convex_gpu)

latencies_convex_nest = []
amplitudes_convex_nest = []
for sl in range(len(convex_nests)):
	latencies_convex_nest_tmp = []
	amplitudes_convex_nest_tmp = []
	for i in convex_nests[sl]:
		latencies_convex_nest_tmp.append(corr_latencies_all_runs_nest[sl][i])
		amplitudes_convex_nest_tmp.append(corr_amplitudes_all_runs_nest[sl][i])
	latencies_convex_nest.append(latencies_convex_nest_tmp)
	amplitudes_convex_nest.append(amplitudes_convex_nest_tmp)

latencies_convex_neuron = []
amplitudes_convex_neuron = []
for sl in range(len(convex_neurons)):
	latencies_convex_neuron_tmp = []
	amplitudes_convex_neuron_tmp = []
	for i in convex_neurons[sl]:
		latencies_convex_neuron_tmp.append(corr_latencies_all_runs_neuron[sl][i])
		amplitudes_convex_neuron_tmp.append(corr_amplitudes_all_runs_neuron[sl][i])
	latencies_convex_neuron.append(latencies_convex_neuron_tmp)
	amplitudes_convex_neuron.append(amplitudes_convex_neuron_tmp)

latencies_convex_gpu = []
amplitudes_convex_gpu= []
for sl in range(len(convex_gpus)):
	latencies_convex_gpu_tmp = []
	amplitudes_convex_gpu_tmp = []
	for i in convex_gpus[sl]:
		latencies_convex_gpu_tmp.append(corr_latencies_all_runs_gpu[sl][i])
		amplitudes_convex_gpu_tmp.append(corr_amplitudes_all_runs_gpu[sl][i])
	latencies_convex_gpu.append(latencies_convex_gpu_tmp)
	amplitudes_convex_gpu.append(amplitudes_convex_gpu_tmp)
lens_nest = []
for dot in range(len(corr_latencies_all_runs_nest)):
	lens_nest.append(len(latencies_convex_nest[dot]))
times_convex_nest = []
for i in range(len(lens_nest)):
	times_convex_nest_tmp = []
	for j in range(lens_nest[i]):
		times_convex_nest_tmp.append(i)
	times_convex_nest.append(times_convex_nest_tmp)

lens_neuron = []
for dot in range(len(corr_latencies_all_runs_neuron)):
	lens_neuron.append(len(latencies_convex_neuron[dot]))
times_convex_neuron = []
for i in range(len(lens_neuron)):
	times_convex_neuron_tmp = []
	for j in range(lens_neuron[i]):
		times_convex_neuron_tmp.append(i + 0.5)
	times_convex_neuron.append(times_convex_neuron_tmp)

lens_gpu = []
for dot in range(len(corr_latencies_all_runs_gpu)):
	lens_gpu.append(len(latencies_convex_gpu[dot]))
times_convex_gpu = []
for i in range(len(lens_gpu)):
	times_convex_gpu_tmp = []
	for j in range(lens_gpu[i]):
		times_convex_gpu_tmp.append(i + 0.25)
	times_convex_gpu.append(times_convex_gpu_tmp)
for dot in range(len(corr_latencies_all_runs_nest)):
	x_nest = times_convex_nest[dot] + [times_convex_nest[dot][0]]
	y_nest = latencies_convex_nest[dot] + [latencies_convex_nest[dot][0]]
	z_nest = amplitudes_convex_nest[dot] + [amplitudes_convex_nest[dot][0]]

	x_neuron = times_convex_neuron[dot] + [times_convex_neuron[dot][0]]
	y_neuron = latencies_convex_neuron[dot] + [latencies_convex_neuron[dot][0]]
	z_neuron = amplitudes_convex_neuron[dot] + [amplitudes_convex_neuron[dot][0]]

	x_gpu = times_convex_gpu[dot] + [times_convex_gpu[dot][0]]
	y_gpu = latencies_convex_gpu[dot] + [latencies_convex_gpu[dot][0]]
	z_gpu = amplitudes_convex_gpu[dot] + [amplitudes_convex_gpu[dot][0]]

	ax.add_collection3d(plt.fill_between(y_nest, z_nest, min(z_nest), color='green', alpha=0.3, label="filled plot"),
	                    x_nest[dot], zdir='x')
	ax.add_collection3d(plt.fill_between(y_neuron, z_neuron, min(z_neuron), color='purple', alpha=0.3,
	                                     label="filled plot"), x_neuron[dot], zdir='x')
	ax.add_collection3d(plt.fill_between(y_gpu, z_gpu, min(z_gpu), color='orange', alpha=0.3,
	                                     label="filled plot"), x_gpu[dot], zdir='x')

	ax.plot(times_convex_nest[dot], latencies_convex_nest[dot], amplitudes_convex_nest[dot], color='green', alpha=0.3,
	        label='nest')
	print("old_times_nest = ", old_times_nest)
	print()
	ax.plot(old_times_nest[dot], fliers_latencies_nest_values[dot], fliers_amplitudes_nest_values[dot], '.',
	        color='green', alpha=0.7)

	ax.plot(times_convex_gpu[dot], latencies_convex_gpu[dot], amplitudes_convex_gpu[dot], color='orange', alpha=0.3,
	        label='gpu')
	ax.plot(old_times_gpu[dot], fliers_latencies_gpu_values[dot], fliers_amplitudes_gpu_values[dot], '.',
	        color='orange', alpha=0.7)
for dot in range(len(corr_latencies_all_runs_neuron)):
	ax.plot(times_convex_neuron[dot], latencies_convex_neuron[dot], amplitudes_convex_neuron[dot], color='purple',
	        alpha=0.3)
for dot in range(len(fliers_neuron)):
	ax.plot(old_times_neuron[dot], fliers_latencies_neuron_values[dot], fliers_amplitudes_neuron_values[dot], '.',
	        color='purple', alpha=0.7)
nest_clouds_patch = mpatches.Patch(color='green', label='nest clouds')
neuron_clouds_patch = mpatches.Patch(color='purple', label='neuron clouds')
neuron_patches = mpatches.Patch(color='blue', label='neuron')
nest_patches = mpatches.Patch(color='orange', label='nest')
ax.set_xlabel("Slice number")
ax.set_ylabel("Latencies ms")
ax.set_zlabel("Amplitudes ms")
ax.set_title("Slice - Latency - Amplitude")
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.0, hspace=0.09)
plt.show()
