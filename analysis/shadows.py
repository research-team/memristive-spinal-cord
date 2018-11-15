import os
import pylab
import numpy as np
import h5py as hdf5
from collections import defaultdict


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
			data_by_test[test_name] = test_values[:]  # [:] will return ndarray, don't use list() !!!
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
	num_slices = int(sim_time) // slice_duration
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


def __plot(slices, sim_step, raw_data_filename, save_path=None):
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
	pylab.figure(figsize=(10, 5))
	for slice_number, tests in slices.items():
		offset = -slice_number * 10

		mean_data = list(map(lambda elements: np.mean(elements), zip(*tests.values())))  #
		times = [time * sim_step for time in range(len(mean_data))]  # divide by 10 to convert to ms step

		means = [voltage + offset for voltage in mean_data]
		yticks.append(means[0])
		minimal_per_step = [min(a) for a in zip(*tests.values())]
		maximal_per_step = [max(a) for a in zip(*tests.values())]
		# plot mean with shadows
		pylab.plot(times, means, linewidth=0.5, color='k')
		pylab.fill_between(times,
		                   [mini + offset for mini in minimal_per_step],  # the minimal values + offset (slice number)
		                   [maxi + offset for maxi in maximal_per_step],  # the maximal values + offset (slice number)
		                   alpha=0.35)
	pylab.xticks(range(26), [i if i % 5 == 0 else "" for i in range(26)])
	pylab.yticks(yticks, range(1, len(slices) + 1), fontsize=7)
	pylab.xlim(0, 25)
	pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)

	# if the save path is not specified
	if not save_path:
		save_path = "/".join(raw_data_filename.split("/")[:-1])
	pylab.savefig(os.path.join(save_path, "shadows.png"), format="png", dpi=512)


def plot_shadows(raw_data_filename, sim_step, save_path=None):
	data = __read_data(raw_data_filename)
	slices, sim_time = __restructure_data(data, sim_step=sim_step)
	__plot(slices, sim_step, raw_data_filename, save_path)


def debugging():
	path = 'PASTE/PATH/TO/THE/RESULTS'
	# add info about simulation step. Neuron is 0.025ms, NEST is 0.1ms
	sim_step = 0.025

	data = __read_data(path)
	slices = __restructure_data(data, sim_step=sim_step)
	__plot(slices, sim_step, path)


if __name__ == "__main__":
	debugging()
