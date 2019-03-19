import logging
import subprocess
import numpy as np
import h5py as hdf5
from analysis.shadows import plot_shadows

ABS_PATH = "/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU/"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('Converter')

def run_tests(command, tests_number):
	logger.info("start {} tests".format(tests_number))
	for i in range(tests_number):
		logger.info("running test #{}".format(i))
		subprocess.call([command, str(i)])


def write_to_hdf5(tests_number, filename, step):
	all_data_E = []
	all_data_F = []
	logger.info("start write data to the HDF5")
	flexor_end = int(125 / step)

	for test_index in range(tests_number):
		logger.info("process test #{}".format(test_index))
		with open('{}/volt_{}.dat'.format(ABS_PATH, test_index), 'r') as file:
			per_test_mean_data_extensor = []
			per_test_mean_data_flexor = []
			for line in file.readlines():
				nrn_id, *volts = line.split()
				if 1212 <= int(nrn_id) <= 1380:
					first_val = float(volts[0])
					per_test_mean_data_extensor.append([-float(volt) + first_val for volt in volts])
				if 1381 <= int(nrn_id) <= 1549:
					first_val = float(volts[0])
					per_test_mean_data_flexor.append([-float(volt) + first_val for volt in volts])
			mean_extensor = list(map(lambda x: np.mean(x), zip(*per_test_mean_data_extensor)))
			mean_flexor = list(map(lambda x: np.mean(x), zip(*per_test_mean_data_flexor)))

			if test_index == 1:
				import pylab
				shared_x = [x * step for x in range(len(mean_extensor))]
				pylab.plot(shared_x, mean_flexor, label="Flexor")
				pylab.plot(shared_x, mean_extensor, label="Extensor")
				pylab.xlim(0, 275)
				for i in range(0, 276, 25):
					pylab.axvline(x=i, color='grey', linestyle='--')
				pylab.xticks(range(0, 276, 25))
				pylab.yticks([])
				pylab.legend()
				pylab.show()

			all_data_F.append(mean_flexor)
			all_data_E.append(mean_extensor)

			with open("/home/alex/MP_E.dat", 'w') as f:
				for v in mean_extensor:
					f.write("{} ".format(v))
			with open("/home/alex/MP_F.dat", 'w') as f:
				for v in mean_flexor:
					f.write("{} ".format(v))

	logger.info("writing data to the HDF5")
	with hdf5.File('{}E.hdf5'.format(filename), 'w') as file:
		# foreach test data
		for test_index, test_data in enumerate(all_data_E):
			# convert iterator 'test_data' to the list
			file.create_dataset("test_{}".format(test_index), data=list(test_data), compression="gzip")
	with hdf5.File('{}F.hdf5'.format(filename), 'w') as file:
		# foreach test data
		for test_index, test_data in enumerate(all_data_F):
			# convert iterator 'test_data' to the list
			file.create_dataset("test_{}".format(test_index), data=list(test_data), compression="gzip")


def read_hdf5(filename, step=0.25, begin=0, end=0):
	step_begin = int(begin / step)
	step_end = int(end / step)

	all_data_E = []
	all_data_F = []
	logger.info("read data from {} ms to {} ms with step {}".format(begin, end, step))
	with hdf5.File('{}/{}E.hdf5'.format(ABS_PATH, filename), 'r') as file:
		for data in file.values():
			all_data_E.append(list(data[step_begin:step_end]))
	with hdf5.File('{}/{}F.hdf5'.format(ABS_PATH, filename), 'r') as file:
		for data in file.values():
			all_data_F.append(list(data[step_begin:step_end]))

	return all_data_F, all_data_E


def run():
	tests_number = 20
	command = "/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU/matrix_solution/kek"
	hdf5_filename = "two_muslces_{}_tests".format(tests_number)
	step = 0.25

	run_tests(command, tests_number)
	write_to_hdf5(tests_number, hdf5_filename, step)
	all_data_F, all_data_E = read_hdf5(hdf5_filename, step=step, begin=0, end=275)
	plot_shadows(all_data_F, step=step, save_folder=ABS_PATH, filename=hdf5_filename+"F")
	plot_shadows(all_data_E, step=step, save_folder=ABS_PATH, filename=hdf5_filename+"E")

	# ToDo send 'data' to the shadows plotting script

if __name__ == "__main__":
	run()
