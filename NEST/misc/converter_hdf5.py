import os
# import logging
import numpy as np
import h5py as hdf5
import time as libtime
import multiprocessing as mp
from collections import defaultdict

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger('Converter')


def merge_nest_files(data_path):
	"""

	Args:
		data_path:

	Returns:
		dict: dddd
	"""
	data_by_test = {}
	test_names = set()
	# scan folder for test names
	for filename in os.listdir(data_path):
		if filename.endswith('.dat'):
			test_names.add("-".join(filename.split("-")[:2]))

	# logger.info("have found {} tests".format(len(test_names)))

	# grab data from these tests
	for test_name in test_names:
		# logger.info('gathering data from test {}'.format(test_name))
		voltage_data = defaultdict(list)
		for filename in os.listdir(data_path):
			if filename.startswith(test_name) and filename.endswith('.dat'):
				with open(os.path.join(data_path, filename)) as file:
					for line in file:
						gid, time, volt, *etc = line.split()
						time = float(time)
						volt = float(volt)
						voltage_data[time].append(volt)
		# calculate mean value of voltage
		for time in voltage_data:
			voltage_data[time] = np.mean(voltage_data[time])
		# save dict values as sorted by time list
		data_by_test[test_name] = [voltage_data[time] for time in sorted(voltage_data)]

	return data_by_test


def __init_file_reader(volt_data):
	"""
	Save a link of the local data container to the global voltage_data
	Args:
		volt_data (list):
			container of neuron voltages
	"""
	global voltage_data
	voltage_data = volt_data


def __file_reader(file_path):
	"""
	File reading by given filepath. Put the result to the global data container
	Args:
		file_path (str):
			absolute path to the file
	"""
	neuron_voltage = []
	# read the file data
	with open(file_path) as file:
		d = file.readlines()
		# read from a second line if a first value is 0, else from a first line
		offset = 1 if float(d[0]) == 0 else 0
		# read data
		for line in d[offset:]:
			neuron_voltage.append(float(line) * 10**10)
	# add all voltages for the neuron to the global data container
	voltage_data.append(neuron_voltage)


def merge_neuron_files(data_path, speed):
	data_by_test = {}       # main data container for each test
	manager = mp.Manager()  # init multiprocessing manager
	logger.info('will be utilized all {} cores'.format(mp.cpu_count()))

	# main loop for each test_number
	for test_number in range(25):   # [0, 24]
		logger.info('gathering data from test {}'.format(test_number))
		# main data container
		voltage_data = manager.list()

		# use all available CPUs
		process = mp.Pool(initializer=__init_file_reader, initargs=(voltage_data,))

		# generate path to the files
		file_paths = []
		for neuron in range(21):    # [0, 20]
			for thread in range(8): # [0, 7]
				file_path = os.path.join(data_path, "vMN{}r{}s{}v{}".format(neuron, thread, speed, test_number))
				file_paths.append(file_path)

		# parallelize the job
		process.imap_unordered(__file_reader, file_paths)
		process.close()
		process.join()

		# save mean data by each time slice for the neuron
		data_by_test[test_number] = list(map(lambda slice_of_time: np.mean(slice_of_time), zip(*voltage_data)))

	return data_by_test


def write_to_hdf5(data_by_test, filename):
	"""
	Write data to the HDF5 format.
	Args:
		data_by_test (dict[int, map_object]):
			has iterators of voltage (!)
		filename (str):
			name of the file
	"""
	with hdf5.File('{}.hdf5'.format(filename), 'w') as file:
		# foreach test data
		for test_index, test_data in enumerate(data_by_test.values()):
			# convert iterator 'test_data' to the list
			file.create_dataset("test_{}".format(test_index),
			                    data=list(test_data), compression="gzip")


if __name__ == "__main__":
	simulator = "nest"

	#path = '/home/alex/GitHub/testfolder_NEURON'
	path = '/home/alex/GitHub/memristive-spinal-cord/NEST/testfolder_NEST'

	start_merge = libtime.perf_counter()
	if simulator == "nest":
		data = merge_nest_files(path)
	else:
		data = merge_neuron_files(path, 125)
	end_merge = libtime.perf_counter()
	logger.info("Merged as {:.6f}s".format(end_merge - start_merge))

	write_to_hdf5(data, os.path.join(path, 'testME'))
