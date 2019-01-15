from analysis.functions import read_neuron_data, read_nest_data, read_bio_data, find_latencies, \
	normalization, read_bio_hdf5
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

neuron_list = read_neuron_data('../../neuron-data/sim_healthy_neuron_extensor_eesF40_i100_s21cms_T_100runs.hdf5')
# print("neuron_tests = ", type(neuron_tests))
nest_list = read_nest_data('../../nest-data/21cms/extensor_21cms_40Hz_100inh.hdf5')
# bio = read_bio_data('../bio-data/3_0.91 volts-Rat-16_5-09-2017_RMG_13m-min_one_step.txt')
d1 = read_bio_hdf5('1_new_bio_1.hdf5')
d2 = read_bio_hdf5('1_new_bio_2.hdf5')
d3 = read_bio_hdf5('1_new_bio_3.hdf5')
d4 = read_bio_hdf5('1_new_bio_4.hdf5')
d5 = read_bio_hdf5('1_new_bio_5.hdf5')
bio_data = [d1[0], d2[0], d3[0], d4[0], d5[0]]
# bio_voltages = bio[k_bio_volt]
# bio_stim_indexes = bio[k_bio_stim][:-1]
# bio_datas = calc_max_min(bio_stim_indexes, bio_voltages)
# find EES answers basing on min/max extrema
# bio_ees_indexes = find_ees_indexes(bio_stim_indexes, bio_datas)
# remove unnesesary bio data (after the last EES answer)
# normalize data
# bio_voltages = normalization(bio_voltages, zero_relative=True)
# get the min/max extrema based on EES answers indexes (because we need the data after 25ms of the slice)
# bio_datas = calc_max_min(bio_ees_indexes, bio_voltages)
# get the latencies based on min/max extrema
# bio_lat = find_latencies(bio_datas, bio_step, norm_to_ms=True)

slice_numbers = int(len(neuron_list[0]) * sim_step // 25)
# bio_voltages = bio_voltages[:bio_ees_indexes[slice_numbers]]
# bio_ees_indexes = bio_ees_indexes[:slice_numbers]
# bio_voltages = bio[0]
# bio_stim_indexes = bio[1][:-1]
neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
nest_means = list(map(lambda voltages: np.mean(voltages), zip(*nest_list)))

neuron_means_lat = sim_process(neuron_means, sim_step)[0]
neuron_means_amp = sim_process(neuron_means, sim_step)[1]

nest_means_lat = sim_process(nest_means, sim_step)[0]
nest_means_amp = sim_process(nest_means, sim_step)[1]

print(len(neuron_means_lat), neuron_means_lat)
print(len(neuron_means_amp), neuron_means_amp)
print(len(nest_means_amp), nest_means_amp)
print(len(nest_means_lat), nest_means_lat)
sim_stim_indexes = list(range(0, len(nest_means), int(25 / 0.025)))
# bio_datas = calc_max_min(bio_stim_indexes, bio_voltages)
# bio_ees_indexes = find_ees_indexes(bio_stim_indexes, bio_datas)
# bio_voltages = bio_voltages[:bio_ees_indexes[slice_numbers]]
# bio_ees_indexes = bio_ees_indexes[:slice_numbers]
# bio_voltages = normalization(bio_voltages, zero_relative=True)
# bio_datas = calc_max_min(bio_ees_indexes, bio_voltages)
# bio_lat = find_latencies(bio_datas, bio_step, norm_to_ms=True)

# debug(nest_means, nest_means_datas, sim_stim_indexes, nest_ees_indexes, nest_means_lat, sim_step)
# debug(neuron_means, neuron_means_datas, sim_stim_indexes, neuron_ees_indexes, neuron_means_lat, sim_step)
nest_lat = []
nest_amp = []
for test_data in nest_list:
	nest_lat_tmp, nest_amp_tmp = sim_process(test_data, sim_step)
	nest_lat.append(nest_lat_tmp)
	nest_amp.append(nest_amp_tmp)
print(len(nest_lat))
print(len(nest_amp))
print(max(sum(nest_amp, [])))

neuron_lat = []
neuron_amp = []
for test_data in neuron_list:
	neuron_lat_tmp, neuron_amp_tmp = sim_process(test_data, sim_step)
	neuron_lat.append(neuron_lat_tmp)
	neuron_amp.append(neuron_amp_tmp)
# print("neuron_lat = ", neuron_lat)
# print("neuron_amp = ", neuron_amp)
print(max(sum(neuron_amp, [])))

# bio_amp = calc_amplitudes(bio_datas, bio_lat)

# print("bio_lat = ", bio_lat)
# print("nest_lat = ", len(nest_lat[0]))
# print("neuron_lat = ", len(neuron_lat))
# print("bio_amp = ", bio_amp)

latencies_all_runs_neuron = []
latencies_all_runs_nest = []
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
# print("latencies_all_runs_nest = ", len(latencies_all_runs_nest[0]))

amplitudes_all_runs_nest = []
amplitudes_all_runs_neuron = []

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
# print("amplitudes_all_runs_nest = ", amplitudes_all_runs_nest)
expected_value_amp_neuron = []
std_amp_neuron = []
for dot in amplitudes_all_runs_neuron:
	expected_value_tmp = statistics.mean(dot)
	std_tmp = statistics.stdev(dot)
	expected_value_amp_neuron.append(expected_value_tmp)
	std_amp_neuron.append(std_tmp)
expected_value_lat_neuron = []
std_lat_neuron = []
for dot in latencies_all_runs_neuron:
	expected_value_tmp = statistics.mean(dot)
	std_tmp = statistics.stdev(dot)
	expected_value_lat_neuron.append(expected_value_tmp)
	std_lat_neuron.append(std_tmp)
print("expected_value = ", expected_value_lat_neuron)
print("std = ", std_lat_neuron)
amplitudes_all_runs_neuron_3sigma = []
latencies_all_runs_neuron_3sigma = []
fliers_amplitudes_neuron = []
fliers_latencies_neuron = []
for dot in range(len(amplitudes_all_runs_neuron)):
	amplitudes_all_runs_neuron_dot_3sigma_amp = []
	fliers_amplitudes_neuron_tmp = []
	print("len(dot) = ", len(amplitudes_all_runs_neuron[dot]))
	for i in range(len(amplitudes_all_runs_neuron[dot])):
		if (expected_value_amp_neuron[dot] - 3 * std_amp_neuron[dot]) < amplitudes_all_runs_neuron[dot][i] <\
				(expected_value_amp_neuron[dot] + 3 * std_amp_neuron[dot]):
			amplitudes_all_runs_neuron_dot_3sigma_amp.append(amplitudes_all_runs_neuron[dot][i])
		else:
			fliers_amplitudes_neuron_tmp.append(i)
	fliers_amplitudes_neuron.append(fliers_amplitudes_neuron_tmp)
	print(len(amplitudes_all_runs_neuron_dot_3sigma_amp))
	amplitudes_all_runs_neuron_3sigma.append(amplitudes_all_runs_neuron_dot_3sigma_amp)
print("*" * 10)
for dot in range(len(latencies_all_runs_neuron)):
	latencies_all_runs_neuron_dot_3sigma = []
	fliers_latencies_neuron_tmp = []
	print("len(dot) = ", len(latencies_all_runs_neuron[dot]))
	for i in range(len(latencies_all_runs_neuron[dot])):
		if (expected_value_lat_neuron[dot] - 3 * std_lat_neuron[dot]) < latencies_all_runs_neuron[dot][i] <\
				(expected_value_lat_neuron[dot] + 3 * std_lat_neuron[dot]):
			latencies_all_runs_neuron_dot_3sigma.append(latencies_all_runs_neuron[dot][i])
		else:
			fliers_latencies_neuron_tmp.append(i)
	fliers_latencies_neuron.append(fliers_latencies_neuron_tmp)
	print(len(latencies_all_runs_neuron_dot_3sigma))
	latencies_all_runs_neuron_3sigma.append(latencies_all_runs_neuron_dot_3sigma)
print("amplitudes_all_runs_neuron_3sigma = ", latencies_all_runs_neuron_3sigma)
print("fliers_amplitudes_neuron = ", fliers_amplitudes_neuron)
print("fliers_latencies_neuron = ", fliers_latencies_neuron)
print("latencies_all_runs_neuron = ", len(latencies_all_runs_neuron[1]))
for sl in range(len(fliers_latencies_neuron)):
	print("sl = ", sl)
	for fl in reversed(fliers_latencies_neuron[sl]):
		print(fl)
		if fl:
			del latencies_all_runs_neuron[sl][fl]
			del amplitudes_all_runs_neuron[sl][fl]
	print('---')
	for fl in reversed(fliers_amplitudes_neuron[sl]):
		if fl:
			del latencies_all_runs_neuron[sl][fl]
			del amplitudes_all_runs_neuron[sl][fl]
			print("sl", sl)
			print(fl)
	print('***')
print(len(latencies_all_runs_neuron[5]))
print(len(amplitudes_all_runs_neuron[5]))
time = []
for i in range(len(nest_means_amp)):
	time.append(i)
times_nest = []
times_neuron = []
for dot in range(len(latencies_all_runs_nest)):
	times_tmp = []
	for l in range(len(latencies_all_runs_nest[dot])):
		times_tmp.append(dot)
	times_nest.append(times_tmp)
# print("len(times_nest) = ", len(times_nest[0]))
for dot in range(len(latencies_all_runs_neuron)):
	times_tmp = []
	for l in range(len(latencies_all_runs_neuron[dot])):
		times_tmp.append(dot + 0.5)
	times_neuron.append(times_tmp)
print("times_neuron = ", len(times_neuron[0]))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, nest_means_lat, nest_means_amp, color='green')
ax.plot(time, nest_means_lat, nest_means_amp, '.', lw=0.5, color='r', markersize=5)
ax.plot(time, neuron_means_lat, neuron_means_amp, color='purple')
ax.plot(time, neuron_means_lat, neuron_means_amp, '.', lw=0.5, color='r', markersize=5)
# ax.plot(time, bio_lat, bio_amp, color='black')
# ax.plot(time, bio_lat, bio_amp, '.', lw=0.5, color='r', markersize=5)
nest_y = max(latencies_all_runs_nest)
nest_z = max(amplitudes_all_runs_nest)
verts_nest = []
verts_times = []
verts_latencies = []
verts_amplitudes = []

for run in range(len(times_nest[0])):
	verts_times_tmp = []
	verts_latencies_tmp = []
	verts_amplitudes_tmp = []
	for dot in range(len(times_nest)):
		verts_times_tmp.append(times_nest[dot][run])
		verts_latencies_tmp.append(latencies_all_runs_nest[dot][run])
		verts_amplitudes_tmp.append(amplitudes_all_runs_nest[dot][run])
	verts_times.append(verts_times_tmp)
	verts_latencies.append(verts_latencies_tmp)
	verts_amplitudes.append(verts_amplitudes_tmp)

for run in range(len(verts_times)):
	for i in range(1):
		verts_tmp = []
		verts_tmp.append(verts_times[run])
		verts_tmp.append(verts_latencies[run])
		verts_tmp.append(verts_amplitudes[run])
	verts_nest.append(verts_tmp)
nest = []
for sl in range(len(latencies_all_runs_nest)):
	nest_sl = []
	for dot in range(len(latencies_all_runs_nest[0])):
		one_dot = []
		one_dot.append(latencies_all_runs_nest[sl][dot])
		one_dot.append(amplitudes_all_runs_nest[sl][dot])
		nest_sl.append(one_dot)
	nest.append(nest_sl)
neuron = []
for sl in range(len(latencies_all_runs_neuron)):
	neuron_sl = []
	for dot in range(len(latencies_all_runs_neuron[0])):
		one_dot = []
		one_dot.append(latencies_all_runs_neuron[sl][dot])
		one_dot.append(amplitudes_all_runs_neuron[sl][dot])
		neuron_sl.append(one_dot)
	neuron.append(neuron_sl)


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
for sl in range(len(nest)):
	convex_nest = grahamscan(nest[sl])
	convex_nests.append(convex_nest)
for sl in range(len(neuron)):
	convex_neuron = grahamscan(neuron[sl])
	convex_neurons.append(convex_neuron)
latencies_convex_nest = []
amplitudes_convex_nest = []
for sl in range(len(convex_nests)):
	latencies_convex_nest_tmp = []
	amplitudes_convex_nest_tmp = []
	for i in convex_nests[sl]:
		latencies_convex_nest_tmp.append(latencies_all_runs_nest[sl][i])
		amplitudes_convex_nest_tmp.append(amplitudes_all_runs_nest[sl][i])
	latencies_convex_nest.append(latencies_convex_nest_tmp)
	amplitudes_convex_nest.append(amplitudes_convex_nest_tmp)
latencies_convex_neuron = []
amplitudes_convex_neuron = []
for sl in range(len(convex_neurons)):
	latencies_convex_neuron_tmp = []
	amplitudes_convex_neuron_tmp = []
	for i in convex_neurons[sl]:
		latencies_convex_neuron_tmp.append(latencies_all_runs_neuron[sl][i])
		amplitudes_convex_neuron_tmp.append(amplitudes_all_runs_neuron[sl][i])
	latencies_convex_neuron.append(latencies_convex_neuron_tmp)
	amplitudes_convex_neuron.append(amplitudes_convex_neuron_tmp)
lens_nest = []
for dot in range(len(latencies_all_runs_nest)):
	lens_nest.append(len(latencies_convex_nest[dot]))
times_convex_nest = []
for i in range(len(lens_nest)):
	times_convex_nest_tmp = []
	for j in range(lens_nest[i]):
		times_convex_nest_tmp.append(i)
	times_convex_nest.append(times_convex_nest_tmp)
lens_neuron = []
for dot in range(len(latencies_all_runs_neuron)):
	lens_neuron.append(len(latencies_convex_neuron[dot]))
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
		times_convex_neuron_tmp.append(i + 0.5)
	times_convex_neuron.append(times_convex_neuron_tmp)
for dot in range(len(latencies_all_runs_nest)):
	x_nest = times_convex_nest[dot] + [times_convex_nest[dot][0]]
	y_nest = latencies_convex_nest[dot] + [latencies_convex_nest[dot][0]]
	z_nest = amplitudes_convex_nest[dot] + [amplitudes_convex_nest[dot][0]]

	x_neuron = times_convex_neuron[dot] + [times_convex_neuron[dot][0]]
	y_neuron = latencies_convex_neuron[dot] + [latencies_convex_neuron[dot][0]]
	z_neuron = amplitudes_convex_neuron[dot] + [amplitudes_convex_neuron[dot][0]]

	ax.add_collection3d(plt.fill_between(y_nest, z_nest, min(z_nest), color='green', alpha=0.3, label="filled plot"),
	                    x_nest[dot], zdir='x')
	ax.add_collection3d(plt.fill_between(y_neuron, z_neuron, min(z_neuron), color='purple', alpha=0.3,
	                                     label="filled plot"),
	                    x_neuron[dot], zdir='x')
	ax.plot(times_convex_nest[dot],
	        latencies_convex_nest[dot],
	        amplitudes_convex_nest[dot],
	        color='green',
	        alpha=0.3, label='nest')
	ax.plot(times_nest[dot], latencies_all_runs_nest[dot], amplitudes_all_runs_nest[dot], '.', color='green',
	        alpha=0.7)
for dot in range(len(latencies_all_runs_neuron)):
	ax.plot(times_convex_neuron[dot], latencies_convex_neuron[dot], amplitudes_convex_neuron[dot], color='purple',
	        alpha=0.3)
	ax.plot(times_neuron[dot], latencies_all_runs_neuron[dot], amplitudes_all_runs_neuron[dot], '.', color='purple',
	        alpha=0.7)
nest_clouds_patch = mpatches.Patch(color='green', label='nest clouds')
neuron_clouds_patch = mpatches.Patch(color='purple', label='neuron clouds')
neuron_patches = mpatches.Patch(color='blue', label='neuron')
nest_patches = mpatches.Patch(color='orange', label='nest')
ax.set_xlabel("Slice number")
ax.set_ylabel("Latencies ms")
ax.set_zlabel("Amplitudes ms")
ax.set_title("Slice - Latency - Amplitude")
plt.show()
