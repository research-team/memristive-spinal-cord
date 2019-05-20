from collections import defaultdict
import h5py as hdf5
from copy import deepcopy
from matplotlib import pylab as plt


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
			data_by_test[test_name] = test_values[:]
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
	# get simulation time by len of records and multiplication by simulation step
	sim_time = len(next(iter(original_data.values()))) * sim_step
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


def select_slices(path, from_border, till_border):
	sim_step = 0.025
	data = __read_data(path)
	data = [v for v in data.values()]
	for test in range(len(data)):
		data[test] = data[test].tolist()[from_border:till_border]
		# print(len(data[test]))
		# for i in range(12000, 51000):
		# 	del data[test][i]
		# for i in range(24000, 28000):
		# 	del data[test][i]
		# for i in range(36000, 38999):
		# 	print("data[{}][{}] =".format(test, i), data[test][i])
			# del data[test][i]
		print(len(data[test]))
	for index, run in enumerate(data):
		offset = index * 128
		# plt.plot([r + offset for r in run])
	# plt.show()
	# print("data = ", len(data), len(data[0]))

	# slices = __restructure_data(data, sim_step=sim_step)
	# dictionary = deepcopy(slices)
	# selected_slices1 = {0: dictionary.get(0), 1: dictionary.get(1), 2: dictionary.get(2), 3: dictionary.get(3),
	#                    4: dictionary.get(4), 5: dictionary.get(5), 6: dictionary.get(6), 7: dictionary.get(7),
	#                    8: dictionary.get(8), 9: dictionary.get(9), 10: dictionary.get(10), 11: dictionary.get(11)}
	# selected_slices2 = {0: dictionary.get(17), 1: dictionary.get(18), 2: dictionary.get(19), 3: dictionary.get(20),
	#                    4: dictionary.get(21), 5: dictionary.get(22), 6: dictionary.get(23), 7: dictionary.get(24),
	#                    8: dictionary.get(25), 9: dictionary.get(26), 10: dictionary.get(27), 11: dictionary.get(28)}
	# selected_slices3 = {0: dictionary.get(34), 1: dictionary.get(35), 2: dictionary.get(36), 3: dictionary.get(37),
	#                    4: dictionary.get(38), 5: dictionary.get(39), 6: dictionary.get(40), 7: dictionary.get(41),
	#                    8: dictionary.get(42), 9: dictionary.get(43), 10: dictionary.get(44), 11: dictionary.get(45)}
	# list1 = [v for v in selected_slices1.values()]
	# list2 = [v for v in selected_slices2.values()]
	# list3 = [v for v in selected_slices3.values()]
	# selected_slices_list = [list1, list2, list3]
	# print("selected_slices_list = ", len(selected_slices_list), len(selected_slices_list[0]),
	#       len(selected_slices_list[1]), len(selected_slices_list[2]), selected_slices_list[2][0])
	return data


# select_slices('../../GRAS/F_15cms_40Hz_100%_2pedal_no5ht.hdf5')