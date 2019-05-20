import os
import pylab
import numpy as np
import h5py as hdf5
from collections import defaultdict
from analysis.functions import read_bio_data, normalization
# from analysis.bio_data_6runs import data_slices
from analysis.patterns_in_bio_data import bio_data_runs
from analysis.neuron_data import neuron_data
from copy import deepcopy
from analysis.functions import bio_slices


def __read_data(filepath):
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
			# print("test_name = ", test_name)
			data_by_test[test_name] = test_values[:]  # [:] will return ndarray, don't use list() !!!
			# print("data_by_test = ", data_by_test)
	# print("len(data_by_test) = ", len(data_by_test))
	return data_by_test


def __restructure_data(original_data, sim_step):
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
	# print("num_slices = ", num_slices)
	# get normlization coefficient from slice duration and simulation step
	normalized_time_from_index = int(slice_duration / sim_step)
	# generate dict container
	voltages_by_slice = {slice_index: defaultdict(list) for slice_index in range(num_slices)}
	# relocate voltage data by their test and slice affiliation
	# print(len(voltages_by_slice))
	for test_name, test_values in original_data.items():
		for index, voltage in enumerate(test_values):
			# get index of the slice by floor division of normalize to 1 ms time
			slice_index = index // normalized_time_from_index
			# collect current voltage to the affiliated test name and slice number
			voltages_by_slice[slice_index][test_name].append(voltage)
	# for k, v in voltages_by_slice.items():
		# for i in range(41):
		# if k == 40:
			# print("voltages_by_slice = ", v)
	# for k, v in voltages_by_slice.values():
	# 	print("voltages_by_slice = ", v)
	return voltages_by_slice


def __plot(slices, sim_step, raw_data_filename, linewidth, alpha, color, save_path=None):
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
	# pylab.figure(figsize=(10, 5))
	# times_bio = []
	# offset_bio = 0
	# for index, sl in enumerate(range(len(bio_data_list))):
	# 	times_bio_tmp = []
	# 	for i in range(offset_bio, offset_bio + len(bio_data_list[sl])):
	# 		times_bio_tmp.append(i * 10)
	# 	offset_bio += len(bio_data_list[sl])
	# 	times_bio.append(times_bio_tmp)
	# times_for_bio = []
	# for sl in range(len(times_bio)):
	# 	times_for_bio_tmp = []
	# 	for i in range(len(times_bio[sl])):
	# 		times_for_bio_tmp.append(i * 10)
	# 	times_for_bio.append(times_for_bio_tmp)

	for slice_number, tests in slices.items():
		offset = slice_number * 124
		# for k, v in tests.items():
		# 	tests[k] = normalization(v, zero_relative=True)
		mean_data = list(map(lambda elements: np.mean(elements), zip(*tests.values())))  #
		times = [time * sim_step for time in range(len(mean_data))]  # divide by 10 to convert to ms step
		means = [voltage + offset for voltage in mean_data]
		# for i in range(len(means)):
		# 	if means[i] != None:
		# 		print("i = ", i)
		# 		break
		# print("means = ", means)
		yticks.append(means[0])
		minimal_per_step = [min(a) for a in zip(*tests.values())]
		maximal_per_step = [max(a) for a in zip(*tests.values())]
							# plot mean with shadows
		pylab.plot(times, means, linewidth=linewidth, color=color)
		pylab.fill_between(times,
		                   [mini + offset for mini in minimal_per_step],  # the minimal values + offset (slice number)
		                   [maxi + offset for maxi in maximal_per_step],  # the maximal values + offset (slice number)
		                   alpha=alpha, color=color)

	# for index, sl in enumerate(range(len(bio_data_list))):
	# 	if index == 9:
	# 		offset = index - 0.01
			# print("here")
		# elif index == 11:
		# 	offset = index - 0.2
			# print("h")
		# else:
		# 	offset = index - 0.05 * index
		# pylab.plot([t * 0.25 for t in range(len(data_slices[sl]))], [data + offset for data in data_slices[sl]],
		#            color='r', linewidth=0.8)
	# print("yticks = ", yticks)
	# bio = bio_slices()
	# plot bio slices
	step = 0.25
	# for index, sl in enumerate(bio):
	# 	offset = index
	# 	times = [time * step for time in range(len(bio[0]))]
	# 	for run in range(len(sl)):
	# 		pylab.plot(times, [s + offset for s in sl], linewidth=1, color='black')

	pylab.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
	pylab.yticks(yticks, range(1, len(slices) + 1), fontsize=12)
	pylab.xlim(0, 25)
	pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
	# pylab.show()
	# if the save path is not specified
	if not save_path:
		save_path = "/".join(raw_data_filename.split("/")[:-1])
	pylab.savefig(os.path.join(save_path, "shadows.png"), format="png", dpi=512)


def plot_shadows(raw_data_filename, sim_step, save_path=None):
	data = __read_data(raw_data_filename)
	slices, sim_time = __restructure_data(data, sim_step=sim_step)
	__plot(slices, sim_step, raw_data_filename, save_path)


def debugging():
	path = '../../neuron-data/3steps_speed15_EX.hdf5'
	# path2 = '../../neuron-data/15EX_20tests_no_serotonin.hdf5'
	# add info about simulation step. Neuron is 0.025ms, NEST is 0.1ms
	sim_step = 0.025    # don't forget to change the step size!

	data = __read_data(path)
	# data2 = __read_data(path2)
	# data = bio_data_runs()
	slices = __restructure_data(data, sim_step=sim_step)
	# slices2 = __restructure_data(data2, sim_step=sim_step)
	# print(type(slices))
	# print(slices.keys())
	dictionary = deepcopy(slices)
	# print(dictionary.keys())
	# for i in range(6, 33):
	# 	del slices[i]
	# for i in range(1, 7):
	# 	del slices[i]
	slices1_fl = {0: dictionary.get(0), 1: dictionary.get(1), 2: dictionary.get(2),
	           3: dictionary.get(3), 4: dictionary.get(4), 5: dictionary.get(5), 6: dictionary.get(6),
	           7: dictionary.get(7), 8: dictionary.get(8), 9: dictionary.get(9), 10: dictionary.get(10),
	             11: dictionary.get(11), 12: dictionary.get(12), 13: dictionary.get(13), 14: dictionary.get(14),
	              15: dictionary.get(15), 16: dictionary.get(16)
	              }
	slices1_ex = {0: dictionary.get(10), 1: dictionary.get(11), 2: dictionary.get(12),
	           3: dictionary.get(13), 4: dictionary.get(14), 5: dictionary.get(15), 6: dictionary.get(16),
	           7: dictionary.get(17), 8: dictionary.get(18), 9: dictionary.get(19), 10: dictionary.get(20),
	             11: dictionary.get(21)#, 12: dictionary.get(12), 13: dictionary.get(13), 14: dictionary.get(14),
	              # 15: dictionary.get(15), 16: dictionary.get(16), 17: dictionary.get(17), 18: dictionary.get(18),
	              # 19: dictionary.get(19), 20: dictionary.get(20), 21: dictionary.get(21)#, 22: dictionary.get(22),
	           #    # # 23: dictionary.get(23), 24: dictionary.get(24), 25: dictionary.get(25), 26: dictionary.get(26),
	           #    # # 27: dictionary.get(27), 28: dictionary.get(28), 29: dictionary.get(29)#, 30: dictionary.get(30),
	           #    # # 31: dictionary.get(31), 32: dictionary.get(32), 33: dictionary.get(33), 34: dictionary.get(34)
	              }
	slices2 = {0: dictionary.get(32), 1: dictionary.get(33), 2: dictionary.get(34),
	           3: dictionary.get(35), 4: dictionary.get(36), 5: dictionary.get(37), 6: dictionary.get(38),
	           7: dictionary.get(39), 8: dictionary.get(40), 9: dictionary.get(41), 10: dictionary.get(42),
	           11: dictionary.get(43)#, 12: dictionary.get(44), 13: dictionary.get(45), 14: dictionary.get(46),
	           # 15: dictionary.get(37), 16: dictionary.get(38), 17: dictionary.get(39), 18: dictionary.get(40),
	           # 19: dictionary.get(41), 20: dictionary.get(42), 21: dictionary.get(43)#, 22: dictionary.get(57),
	           # # # # 23: dictionary.get(58), 24: dictionary.get(59), 25: dictionary.get(60), 26: dictionary.get(61),
	           # # # # 27: dictionary.get(62), 28: dictionary.get(63), 29: dictionary.get(64)#, 30: dictionary.get(65),
	           # # # # 31: dictionary.get(66), 32: dictionary.get(67), 33: dictionary.get(68), 34: dictionary.get(69)
	           }
	slices3 = {0: dictionary.get(54), 1: dictionary.get(55), 2: dictionary.get(56),
	           3: dictionary.get(57), 4: dictionary.get(58), 5: dictionary.get(59), 6: dictionary.get(60),
	           7: dictionary.get(61), 8: dictionary.get(62), 9: dictionary.get(63), 10: dictionary.get(64),
	           11: dictionary.get(65)#, 12: dictionary.get(56), 13: dictionary.get(57), 14: dictionary.get(58),
	           # 15: dictionary.get(59), 16: dictionary.get(60), 17: dictionary.get(61), 18: dictionary.get(62),
	           # 19: dictionary.get(63), 20: dictionary.get(64), 21: dictionary.get(65)#, 22: dictionary.get(92),
	           # # # # 23: dictionary.get(93), 24: dictionary.get(94), 25: dictionary.get(95), 26: dictionary.get(96),
	           # # # # 27: dictionary.get(97), 28: dictionary.get(98), 29: dictionary.get(99)#, 30: dictionary.get(100),
	           # # # # 31: dictionary.get(101), 32: dictionary.get(102), 33: dictionary.get(103), 34: dictionary.get(104)
	           }
	# print(type(slices2))
	# print(slices2.keys())
	# print("len(slices) = ", len(slices))
	__plot(slices, sim_step, path, 0.2, 0.6, 'black')
	# __plot(slices2, sim_step, path, 0.1, 0.3, 'red')
	# __plot(slices3, sim_step, path, 0.1, 0.3, 'blue')
	pylab.show()


if __name__ == "__main__":
	debugging()
