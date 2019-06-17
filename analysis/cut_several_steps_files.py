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
	for index, run in enumerate(data):
		offset = index * 128
		# plt.plot([r + offset for r in run])
	# plt.show()
	return data


# select_slices('../../GRAS/F_15cms_40Hz_100%_2pedal_no5ht.hdf5')