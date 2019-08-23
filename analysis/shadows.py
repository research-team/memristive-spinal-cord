import os
import pylab
import numpy as np
import h5py as hdf5
from collections import defaultdict
from analysis.functions import read_bio_data, normalization
# from analysis.bio_data_6runs import data_slices
from analysis.patterns_in_bio_data import bio_data_runs
from copy import deepcopy


def read_data(filepath):
	"""
	Read all data from hdf5 file
	Args:
		filepath (str):
			path to the file
	Returns:
		dict[str, list]: voltage data for each test
	"""
	data_by_test = {}
	with hdf5.File(filepath) as file:
		for test_name, test_values in file.items():
			print("len(test_values) = ", len(test_values))
			data_by_test[test_name] = test_values[:]  # [:] will return ndarray, don't use list() !!!
	return data_by_test


def restructure_data(original_data, sim_step):
	"""
	Restructuring data from test -> slices to slices -> test
	Args:
		original_data (dict[str, list]):
			data container of voltages for each test
	Returns:
		dict[int, dict[str, list]]: restuctured data for easiest way to build the shadows.
		slice index -> [test name -> test values which are corresponed to the current slice ]
	"""
	# constant
	slice_duration = 25
	i = 0
	for k, v in original_data.items():
		# if v.__contains__('nan'):
		# 	print("v = ", v)
		# first = -v[0]
		# transforming to extracellular form
		# original_data[k] = [-d - first for d in v]
	# 	# if i % 5 == 0:
	# 		# pylab.plot(original_data[k])
		i += 1
	# pylab.show()
	# get simulation time by len of records and multiplication by simulation step
	sim_time = len(next(iter(original_data.values()))) * sim_step #+ 0.1  # for nest
	# get number of slices by floor division of slice duration (25 ms)
	num_slices = int(int(sim_time) // slice_duration)
	# get normlization coefficient from slice duration and simulation step
	normalized_time_from_index = int(slice_duration / sim_step)
	# generate dict container
	voltages_by_slice = {slice_index: defaultdict(list) for slice_index in range(num_slices)}
	# relocate voltage data by their test and slice affiliation
	for test_name, test_values in original_data.items():
		for index, voltage in enumerate(test_values):
			# get index of the slice by floor division of normalize to 1 ms time
			slice_index = index // normalized_time_from_index
			# collect current voltage to the affiliated test name and slice number
			voltages_by_slice[slice_index][test_name].append(voltage)
	return voltages_by_slice


def plot(slices, sim_step, raw_data_filename, linewidth, alpha, color, save_path=None):
	# for i in range(len(slices)):
	# print("slices = ", slices[41])
	"""
	Plot shadows with mean
	Args:
		slices (dict[int, dict[str, list]]):
			restructured data: dict[slice index, dict[test name, voltages]]
		sim_step (float):
			simulation step
		raw_data_filename (str):
			path to the raw data
		save_path (str or None):
			folder path for saving results
	"""
	yticks = []
	all_maxes = []
	all_mins = []

	x_coor = []
	y_coor = []
	for slice_number, tests in slices.items():
		offset = slice_number * 96
		# for k, v in tests.items():
		# 	tests[k] = normalization(v, zero_relative=True)
		mean_data = list(map(lambda elements: np.mean(elements), zip(*tests.values())))  #
		times = [time * sim_step for time in range(len(mean_data))]  # divide by 10 to convert to ms step
		means = [voltage + offset for voltage in mean_data]
		yticks.append(means[0])
		minimal_per_step = [min(a) for a in zip(*tests.values())]
		all_mins.append([m + 10 for m in minimal_per_step])
		maximal_per_step = [max(a) for a in zip(*tests.values())]
		all_maxes.append([m + 10 for m in maximal_per_step])
							# plot mean with shadows
		# min_difference_indexes, max_difference_indexes, necessary_indexes = find_min_diff(all_maxes, all_mins, sim_step)
		pylab.plot(times, means, linewidth=linewidth, color=color)
		pylab.fill_between(times,
		                   [mini + offset for mini in minimal_per_step],  # the minimal values + offset (slice number)
		                   [maxi + offset for maxi in maximal_per_step],  # the maximal values + offset (slice number)
		                   alpha=alpha, color=color)

		# pylab.plot(times[min_difference_indexes[slice_number]],
		#            mean_data[min_difference_indexes[slice_number]] + offset, marker='.', markersize=12, color='red')
		# pylab.plot(times[max_difference_indexes[slice_number]],
		#            mean_data[max_difference_indexes[slice_number]] + offset, marker='.', markersize=12,	color='blue')
		# pylab.plot(times[necessary_indexes[slice_number]], mean_data[necessary_indexes[slice_number]] + offset,
		#            marker='.', markersize=12, color='black')

		# x_coor.append(times[necessary_indexes[slice_number]])
		# y_coor.append(mean_data[necessary_indexes[slice_number]] + offset)

		x_2_coors = []
		y_2_coors = []
		if len(x_coor) > 1:
			x_2_coors.append(x_coor[-2])
			x_2_coors.append(x_coor[-1])
			y_2_coors.append(y_coor[-2])
			y_2_coors.append(y_coor[-1])
			# pylab.plot(x_2_coors, y_2_coors, linestyle='--', color='black')

	# plot bio slices
	step = 0.25
	pylab.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
	pylab.yticks(yticks, range(1, len(slices) + 1), fontsize=14)
	pylab.xlim(0, 25)
	pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
	# pylab.show()
	# if the save path is not specified
	if not save_path:
		save_path = "/".join(raw_data_filename.split("/")[:-1])
	pylab.savefig(os.path.join(save_path, "shadows.png"), format="png", dpi=512)
	return all_maxes, all_mins


def plot_shadows(raw_data_filename, sim_step, save_path=None):
	data = read_data(raw_data_filename)
	slices, sim_time = restructure_data(data, sim_step=sim_step)
	plot(slices, sim_step, raw_data_filename, save_path)


def addFromTo(a, b, dict):
    d = {}
    for d_key, i in enumerate(range(a, b + 1)):
      d[d_key] = dict[i]
    return d


def debugging():
	path = '/home/anna/Desktop/data/4pedal/bio_E_21cms_40Hz_i100_4pedal_no5ht_T_0.25step.hdf5'
	# add info about simulation step. Neuron is 0.025ms, NEST is 0.1ms
	sim_step = 0.025    # don't forget to change the step size!

	data = read_data(path)
	slices = restructure_data(data, sim_step=sim_step)
	dictionary = deepcopy(slices)

	slices_1 = addFromTo(10, 21, dictionary)
	# slices_2 = addFromTo(18, 30, dictionary)

	all_maxes, all_mins = plot(slices_1, sim_step, path, 0.2, 0.45, '#ed553b')
	# all_maxes, all_mins = plot(slices_2, sim_step, path, 0.2, 0.45, '#079294')
	# all_maxes, all_mins = plot(slices3, sim_step, path, 0.2, 0.45, '#fbad18')
	pylab.show()
	return all_maxes, all_mins


if __name__ == "__main__":
	debugging()
