import os
import logging
import h5py as hdf5
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('Converter')


def merge_nest_files(data_path):
	"""

	Args:
		data_path:

	Returns:
		dict: dddd
	"""
	data_by_test = {}

	test_names = set()
	# scan folder
	for filename in os.listdir(data_path):
		if filename.endswith('.dat'):
			test_names.add("-".join(filename.split("-")[:2]))

	logger.info("have found {} tests".format(len(test_names)))

	# grab data from these tests
	for test_name in test_names:
		voltage_data = defaultdict(list)
		for filename in os.listdir(data_path):
			if filename.startswith(test_name) and filename.endswith('.dat'):
				logger.info('gathering data from {}'.format(filename))
				with open(os.path.join(data_path, filename)) as file:
					for line in file:
						gid, time, volt, *etc = line.split()
						time = float(time)
						volt = float(volt)
						voltage_data[time].append(volt)
		# calculate mean value of voltage
		for time in voltage_data.keys():
			voltage_data[time] = np.mean(voltage_data[time])
		# save as list
		data_by_test[test_name] = [voltage_data[time] for time in sorted(voltage_data)]
	logger.info('done')

	return data_by_test


def merge_neuron_files(data_path):
	speed = 125
	data_by_test = {}

	for test_number in range(25):
		logger.info('gathering data from {}'.format(test_number))
		voltage_data = defaultdict(list)
		for neuron in range(21):
			for thread in range(8):
				filename = "vMN{}r{}s{}v{}".format(neuron, thread, speed, test_number)
				with open(os.path.join(data_path, filename)) as file:
					time = 0.0
					for line in file.readlines()[1:]:
						volt = float(line)
						voltage_data[time].append(volt)
						time += 0.025
		# calculate mean value of voltage
		for time in voltage_data.keys():
			voltage_data[time] = np.mean(voltage_data[time]) # or just divide by 168
		# save as list
		data_by_test[test_number] = [voltage_data[time] for time in sorted(voltage_data)]

	return data_by_test


def write_to_hdf5(data_by_test, filename):
	"""

	Args:
		data_by_test (dict):
			ffffff
		filename (str):
			aaaa
	"""
	with hdf5.File('{}.hdf5'.format(filename), 'w') as file:
		for test_index, test_data in enumerate(data_by_test.values()):
			name = "test{}".format(test_index)
			file.create_dataset(name, data=test_data, compression="gzip")


def test_read_hdf5(path):
	with hdf5.File('{}.hdf5'.format(path), 'r') as f:
		for k, v in f.items():
			print(k, len(v))


if __name__ == "__main__":
	#path = '/home/alex/GitHub/memristive-spinal-cord/NEST/testfolder_NEST'
	path = '/home/alex/GitHub/testfolder_NEURON'

	#data = merge_nest_files(path)
	data = merge_neuron_files(path)
	write_to_hdf5(data, os.path.join(path, 'testME'))
	test_read_hdf5(os.path.join(path, 'testME'))
