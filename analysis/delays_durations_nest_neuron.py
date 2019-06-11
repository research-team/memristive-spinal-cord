from analysis.functions import read_neuron_data, read_nest_data, find_fliers
import numpy as np
from matplotlib import pylab as plt
from analysis.histogram_lat_amp import bio_process, sim_process
from analysis.bio_data_6runs import bio_several_runs
from mpl_toolkits.mplot3d import Axes3D

neuron_dict = {}


k_min_time = 2
k_min_val = 3


# common parameters
k_bio_volt = 0
k_bio_stim = 1
bio_step = 0.25
sim_step = 0.025
gpu_step = 0.1
# readind data
neuron_list = read_neuron_data('../../neuron-data/6ST.hdf5')
nest_list = read_nest_data('../../nest-data/sim_extensor_eesF40_i100_s15cms_T.hdf5')
gpu = read_nest_data('../../GPU_extensor_eesF40_inh100_s15cms_T.hdf5')
# bio = read_bio_data('../bio-data/3_0.91 volts-Rat-16_5-09-2017_RMG_13m-min_one_step.txt')
bio = bio_several_runs()
# print("bio = ", len(bio))
# print(bio[0])
# print(bio[1])
bio_data = bio[0]
# print("bio_data = ", bio_data)
# print("bio_data = ", bio_data)
bio_indexes = bio[1]
# print(len(bio_data[2]))
# print("bio_indexes = ", bio_indexes)
# d1 = read_bio_hdf5('1_new_bio_1.hdf5')
# d2 = read_bio_hdf5('1_new_bio_2.hdf5')
# d3 = read_bio_hdf5('1_new_bio_3.hdf5')
# d4 = read_bio_hdf5('1_new_bio_4.hdf5')
# d5 = read_bio_hdf5('1_new_bio_5.hdf5')
# bio_data = [d1[0], d2[0], d3[0], d4[0], d5[0]]

# calculating the number of slices
slice_numbers = int(len(neuron_list[0]) * sim_step // 25)
# print("slice_numbers = ", slice_numbers)
# mean values of all runs
neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
# print("neuron_means = ", neuron_means)
nest_means = list(map(lambda voltages: np.mean(voltages), zip(*nest_list)))
# print("nest_means = ", nest_means)
gpu_means = list(map(lambda voltages: np.mean(voltages), zip(*gpu)))
# print("gpu = ", gpu[0])
# print("gpu_means = ", gpu_means)
bio_means = list(map(lambda voltages: np.mean(voltages), zip(*bio_data)))
# print(len(bio_means))

# creating the list for bio_process()
voltages_and_stim = []
voltages_and_stim.append(bio_means)
voltages_and_stim.append(bio_indexes)
# print("voltages_and_stim = ", voltages_and_stim[0])
# print("voltages_and_stim = ", voltages_and_stim[1])

# calculating latencies and amplitudes of mean values
neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=False)[0]
neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=False)[1]

nest_means_lat = sim_process(nest_means, sim_step)[0]
nest_means_amp = sim_process(nest_means, sim_step)[1]

gpu_means_lat = sim_process(gpu_means, gpu_step)[0]
gpu_means_amp = sim_process(gpu_means, gpu_step)[1]
# print("gpu_means_amp = ", gpu_means_amp)

# bio_means_lat = bio_process(voltages_and_stim, slice_numbers, reversed_data=True)[0]
# bio_means_amp = bio_process(voltages_and_stim, slice_numbers, reversed_data=True)[1]
# print("bio_lat = ", bio_means_lat)
# print("bio_amp = ", bio_means_amp)
# print(len(neuron_means_lat), neuron_means_lat)
# print(len(neuron_means_amp), neuron_means_amp)

# indexes of stimulations in simulated data
sim_stim_indexes = list(range(0, len(nest_means), int(25 / 0.025)))

# calculating the latencies and amplitudes for all runs
bio_lat = []
bio_amp = []
# creating the lists for bio_process()
for test_data in bio_data:
	to_process = []
	to_process.append(test_data)
	to_process.append(bio_indexes)
	# print("to_process = ", to_process[0])
	# print("to_process = ", to_process[1])
	# bio_lat_tmp, bio_amp_tmp = bio_process(to_process, slice_numbers, reversed_data=True)
	# checking for the elements less than zero
	# for i in range(len(bio_lat_tmp)):
	# 	if bio_lat_tmp[i] < 0:
	# 		bio_lat_tmp[i] = bio_lat_tmp[i - 1]
	# 	if bio_amp_tmp[i] < 0:
	# 		bio_amp_tmp[i] = bio_amp_tmp[i - 1]
	# bio_lat.append(bio_lat_tmp)
	# bio_amp.append(bio_amp_tmp)
	# print("bio_lat = ", bio_lat)
	# print("bio_amp = ", bio_amp)

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
	neuron_lat_tmp, neuron_amp_tmp = sim_process(test_data, sim_step, inhibition_zero=False)
	neuron_lat.append(neuron_lat_tmp)
	neuron_amp.append(neuron_amp_tmp)

gpu_lat = []
gpu_amp = []
for test_data in gpu:
	gpu_lat_tmp, gpu_amp_tmp = sim_process(test_data, gpu_step)
	for i in range(len(gpu_lat_tmp)):
		if gpu_lat_tmp[i] < 0:
			gpu_lat_tmp[i] = gpu_lat_tmp[i - 1]
		if gpu_amp_tmp[i] < 0:
			gpu_amp_tmp[i] = gpu_amp_tmp[i - 1]
	gpu_lat.append(gpu_lat_tmp)
	gpu_amp.append(gpu_amp_tmp)
# converting [runs number] list of [slice number] list to [slice number] list of [runs number] list
latencies_all_runs_neuron = []
latencies_all_runs_nest = []
latencies_all_runs_gpu = []
latencies_all_runs_bio = []

for sl in range(len(neuron_lat[0])):
	# print("sl = ", sl)
	latencies_all_runs_neuron_tmp = []
	for dot in range(len(neuron_lat)):
		# print("dot = ", dot)
		latencies_all_runs_neuron_tmp.append(neuron_lat[dot][sl])
	latencies_all_runs_neuron.append(latencies_all_runs_neuron_tmp)
# print("latencies_all_runs_neuron = ", len(latencies_all_runs_neuron))

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
		latencies_all_runs_gpu_tmp.append(gpu_lat[dot][sl])
	latencies_all_runs_gpu.append(latencies_all_runs_gpu_tmp)

# for sl in range(len(bio_lat[0])):
	latencies_all_runs_bio_tmp = []
	for dot in range(len(bio_lat)):
		latencies_all_runs_bio_tmp.append(bio_lat[dot][sl])
	latencies_all_runs_bio.append(latencies_all_runs_bio_tmp)
# print("latencies_all_runs_bio = ", len(latencies_all_runs_bio))

amplitudes_all_runs_nest = []
amplitudes_all_runs_neuron = []
amplitudes_all_runs_gpu = []
amplitudes_all_runs_bio = []

for sl in range(len(neuron_amp[0])):
	# print("sl = ", sl)
	amplitudes_all_runs_neuron_tmp = []
	for dot in range(len(neuron_amp)):
		amplitudes_all_runs_neuron_tmp.append(neuron_amp[dot][sl])
	amplitudes_all_runs_neuron.append(amplitudes_all_runs_neuron_tmp)
# print("amplitudes_all_runs_neuron = ", amplitudes_all_runs_neuron)

for sl in range(len(nest_amp[0])):
	amplitudes_all_runs_nest_tmp = []
	for dot in range(len(nest_amp)):
		amplitudes_all_runs_nest_tmp.append(nest_amp[dot][sl])
	amplitudes_all_runs_nest.append(amplitudes_all_runs_nest_tmp)

for sl in range(len(gpu_amp[0])):
	amplitudes_all_runs_gpu_tmp = []
	for dot in range(len(gpu_amp)):
		# print("dot = ", dot)
		amplitudes_all_runs_gpu_tmp.append(gpu_amp[dot][sl])
	amplitudes_all_runs_gpu.append(amplitudes_all_runs_gpu_tmp)

# for sl in range(len(bio_amp[0])):
	amplitudes_all_runs_bio_tmp = []
	for dot in range(len(bio_amp)):
		amplitudes_all_runs_bio_tmp.append(bio_amp[dot][sl])
	amplitudes_all_runs_bio.append(amplitudes_all_runs_bio_tmp)
# print("amplitudes_all_runs_bio = ", amplitudes_all_runs_bio)

# finding the fliers of all the data and recreating the lists of latencies and aplitudes by deleting these fliers
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
for i in corr_latencies_all_runs_nest:
	print("corr_latencies_all_runs_nest = ", i)
# print("fliers_latencies_nest_values = ", fliers_latencies_nest_values)
# print("fliers_amplitudes_nest_values = ", fliers_amplitudes_nest_values)

proceed_gpu = find_fliers(amplitudes_all_runs_gpu, latencies_all_runs_gpu)
corr_latencies_all_runs_gpu = proceed_gpu[0]
corr_amplitudes_all_runs_gpu = proceed_gpu[1]
fliers_gpu = proceed_gpu[2]
fliers_latencies_gpu_values = proceed_gpu[3]
fliers_amplitudes_gpu_values = proceed_gpu[4]

# proceed_bio = find_fliers(amplitudes_all_runs_bio, latencies_all_runs_bio)
# corr_latencies_all_runs_bio = proceed_bio[0]
# corr_amplitudes_all_runs_bio = proceed_bio[1]
# fliers_bio = proceed_bio[2]
# fliers_latencies_bio_values = proceed_bio[3]
# fliers_amplitudes_bio_values = proceed_bio[4]
# print("corr_latencies_all_runs_bio = ", corr_latencies_all_runs_bio)
# print("corr_amplitudes_all_runs_bio = ", corr_amplitudes_all_runs_bio)
# print("fliers_bio = ", fliers_bio)
# print("fliers_latencies_bio_values = ", fliers_latencies_bio_values)
# print("fliers_amplitudes_bio_values = ", fliers_amplitudes_bio_values)

# lists of times (x coordinates) for the graphs of mean values
time = []
time_neuron = []
time_gpu = []
time_bio = []
# print("len(nest_means_amp) = ", len(nest_means_amp))
for i in range(len(neuron_means_amp)):
	time.append(i)
	time_neuron.append(i + 0.5)
	time_gpu.append(i + 0.25)
	time_bio.append(i + 0.15)
# print("time = ", time)
# print("time_neuron = ", time_neuron)
# print("time_bio = ", time_bio)

# times for clouds
times_nest = []
times_neuron = []
times_gpu = []
times_bio = []

# times of fliers
old_times_neuron = []
old_times_nest = []
old_times_gpu = []

for dot in range(len(corr_latencies_all_runs_nest)):
	times_tmp = []
	for l in range(len(corr_latencies_all_runs_nest[dot])):
		times_tmp.append(dot)
	times_nest.append(times_tmp)
# print("times_nest = ", len(times_nest[0]))

for dot in range(len(corr_latencies_all_runs_neuron)):
	times_tmp = []
	for l in range(len(corr_latencies_all_runs_neuron[dot])):
		times_tmp.append(dot + 0.5)
	times_neuron.append(times_tmp)
# print("times_neuron = ", len(times_neuron[0]))

for dot in range(len(corr_latencies_all_runs_gpu)):
	times_tmp = []
	for l in range(len(corr_latencies_all_runs_gpu[dot])):
		times_tmp.append(dot + 0.25)
	times_gpu.append(times_tmp)
# print("times_gpu = ", len(times_gpu[0]))

# for dot in range(len(corr_latencies_all_runs_bio)):
# 	times_tmp = []
# 	for l in range(len(corr_latencies_all_runs_bio[dot])):
# 		times_tmp.append(dot + 0.15)
# 	times_bio.append(times_tmp)
# print("times_bio = ", len(times_bio[0]))

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
# print("old_times_gpu = ", old_times_gpu)

# plot mean values (lines)
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.plot(time, nest_means_lat, nest_means_amp, color='green')
# ax.plot(time, nest_means_lat, nest_means_amp, '.', lw=0.5, color='r', markersize=5)

ax.plot(time_neuron, neuron_means_lat, neuron_means_amp, color='purple')
ax.plot(time_neuron, neuron_means_lat, neuron_means_amp, '.', lw=0.5, color='r', markersize=5)

# ax.plot(time_gpu, gpu_means_lat, gpu_means_amp, color='orange')
# ax.plot(time_gpu, gpu_means_lat, gpu_means_amp, '.', lw=0.5, color='r', markersize=5)

# print('-----')
# print("time_bio = ", len(time_bio))
# print("bio_means_lat = ", len(bio_means_lat))
# print("bio_means_amp = ", len(bio_means_amp))
# print('-----')

# ax.plot(time_bio, bio_means_lat, bio_means_amp, color='black')
# ax.plot(time_bio, bio_means_lat, bio_means_amp, '.', lw=0.5, color='r', markersize=5)

# y and z coordinates of the extreme points of nest clouds
nest_y = max(corr_latencies_all_runs_nest)
nest_z = max(corr_amplitudes_all_runs_nest)

# list of dots of all clouds
nest = []
for sl in range(len(corr_latencies_all_runs_nest)):
	nest_sl = []
	for dot in range(len(corr_latencies_all_runs_nest[sl])):
		one_dot = []
		one_dot.append(corr_latencies_all_runs_nest[sl][dot])
		one_dot.append(corr_amplitudes_all_runs_nest[sl][dot])
		nest_sl.append(one_dot)
	nest.append(nest_sl)
for n in nest:
	print("nest = ", n)
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


bio = []
# for sl in range(len(corr_latencies_all_runs_bio)):
# 	bio_sl = []
# 	for dot in range(len(corr_latencies_all_runs_bio[sl])):
# 		one_dot = []
# 		one_dot.append(corr_latencies_all_runs_bio[sl][dot])
# 		one_dot.append(corr_amplitudes_all_runs_bio[sl][dot])
# 		bio_sl.append(one_dot)
# 	bio.append(bio_sl)


def rotate(A, B, C):
	"""
	Function that determines what side of the vector AB is point C
	(positive returning value corresponds to the left side, negative -- to the right)
	Args:
		A: A coordinate of point
		B: B coordinate of point
		C: C coordinate of point

	Returns:

	"""
	return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])


def grahamscan(A):
	"""

	Args:
		A: list
			coordinates of dots in cloud

	Returns:
		list
			coordinates of dots of convex clouds

	"""
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


# convex clouds
convex_nests = []
convex_neurons = []
convex_gpus = []
convex_bios = []

for sl in range(len(nest)):
	print("nest[{}] = ".format(sl), nest[sl])
	convex_nest = grahamscan(nest[sl])
	convex_nests.append(convex_nest)

for sl in range(len(neuron)):
	convex_neuron = grahamscan(neuron[sl])
	convex_neurons.append(convex_neuron)

for sl in range(len(gpu)):
	convex_gpu = grahamscan(gpu[sl])
	convex_gpus.append(convex_gpu)

for sl in range(len(bio)):
	convex_bio = grahamscan(bio[sl])
	convex_bios.append(convex_bio)

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
latencies_convex_bio = []
amplitudes_convex_bio = []
for sl in range(len(convex_bios)):
	latencies_convex_bio_tmp = []
	amplitudes_convex_bio_tmp = []
	# for i in convex_bios[sl]:
		# latencies_convex_bio_tmp.append(corr_latencies_all_runs_bio[sl][i])
		# amplitudes_convex_bio_tmp.append(corr_amplitudes_all_runs_bio[sl][i])
	latencies_convex_bio.append(latencies_convex_bio_tmp)
	amplitudes_convex_bio.append(amplitudes_convex_bio_tmp)

# times of conves clouds
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
lens_bio = []
# for dot in range(len(corr_latencies_all_runs_bio)):
# 	lens_bio.append(len(latencies_convex_bio[dot]))
times_convex_bio = []
for i in range(len(lens_bio)):
	times_convex_bio_tmp = []
	for j in range(lens_bio[i]):
		times_convex_bio_tmp.append(i + 0.15)
	times_convex_bio.append(times_convex_bio_tmp)

# plot clouds
for dot in range(len(corr_latencies_all_runs_neuron)):
	# x_nest = times_convex_nest[dot] + [times_convex_nest[dot][0]]
	# y_nest = latencies_convex_nest[dot] + [latencies_convex_nest[dot][0]]
	# z_nest = amplitudes_convex_nest[dot] + [amplitudes_convex_nest[dot][0]]

	x_neuron = times_convex_neuron[dot] + [times_convex_neuron[dot][0]]
	y_neuron = latencies_convex_neuron[dot] + [latencies_convex_neuron[dot][0]]
	z_neuron = amplitudes_convex_neuron[dot] + [amplitudes_convex_neuron[dot][0]]
	print("x_neuron = ", x_neuron)
	print("y_neuron = ", y_neuron)
	print("z_neuron = ", z_neuron)
	# x_gpu = times_convex_gpu[dot] + [times_convex_gpu[dot][0]]
	# y_gpu = latencies_convex_gpu[dot] + [latencies_convex_gpu[dot][0]]
	# z_gpu = amplitudes_convex_gpu[dot] + [amplitudes_convex_gpu[dot][0]]

	# x_bio = times_convex_bio[dot] + [times_convex_bio[dot][0]]
	# y_bio = latencies_convex_bio[dot] + [latencies_convex_bio[dot][0]]
	# z_bio = amplitudes_convex_bio[dot] + [amplitudes_convex_bio[dot][0]]

	# ax.add_collection3d(plt.fill_between(y_nest, z_nest, min(z_nest), color='green', alpha=0.3, label="filled plot"),
	#                     x_nest[0], zdir='x')
	ax.add_collection3d(plt.fill_between(y_neuron, z_neuron, min(z_neuron), color='purple', alpha=0.3,
	                                     label="filled plot"), x_neuron[0], zdir='x')
	# ax.add_collection3d(plt.fill_between(y_gpu, z_gpu, min(z_gpu), color='orange', alpha=0.3,
	#                                      label="filled plot"), x_gpu[0], zdir='x')
	# ax.add_collection3d(plt.fill_between(y_bio, z_bio, min(z_bio), color='black', alpha=0.3,
	#                                      label="filled plot"), x_bio[0], zdir='x')

	# ax.plot(times_convex_nest[dot], latencies_convex_nest[dot], amplitudes_convex_nest[dot], color='green', alpha=0.3,
	#         label='nest')
	# print("old_times_nest = ", old_times_nest)
	print()
	# ax.plot(old_times_nest[dot], fliers_latencies_nest_values[dot], fliers_amplitudes_nest_values[dot], '.',
	#         color='green', alpha=0.7)

	# ax.plot(times_convex_gpu[dot], latencies_convex_gpu[dot], amplitudes_convex_gpu[dot], color='orange', alpha=0.3,
	#         label='gpu')
	# ax.plot(old_times_gpu[dot], fliers_latencies_gpu_values[dot], fliers_amplitudes_gpu_values[dot], '.',
	#         color='orange', alpha=0.7)

	# ax.plot(times_convex_bio[dot], latencies_convex_bio[dot], amplitudes_convex_bio[dot], color='black', alpha=0.3,
	#         label='bio')

for dot in range(len(corr_latencies_all_runs_neuron)):
	ax.plot(times_convex_neuron[dot], latencies_convex_neuron[dot], amplitudes_convex_neuron[dot], color='purple',
	        alpha=0.3)
for dot in range(len(fliers_neuron)):
	ax.plot(old_times_neuron[dot], fliers_latencies_neuron_values[dot], fliers_amplitudes_neuron_values[dot], '.',
	        color='purple', alpha=0.7)

ax.set_xlabel("Slice number")
ax.set_ylabel("Latencies ms")
ax.set_zlabel("Amplitudes ms")
ax.set_title("Slice - Latency - Amplitude")
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.0, hspace=0.09)
plt.show()