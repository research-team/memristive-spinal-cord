import os
import logging
import subprocess
import numpy as np
import h5py as hdf5
from GRAS.shadows_boxplot import plot_shadows_boxplot


logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()

tests_number, cms, ees, inh, ped, ht5 = range(6)

def run_tests(build_folder, args):
	"""
	Run N-times cpp builded CUDA file via bash commands
	Args:
		build_folder (str): where cpp file is placed
		args (dict): special args for building properly simulation
	"""
	buildname = "build"
	assert args[ped] in [2, 4]
	assert args[ht5] in [0, 1]
	assert 0 <= args[inh] <= 100
	assert args[cms] in [21, 15, 6]

	nvcc = "/usr/local/cuda-10.0/bin/nvcc"
	buildfile = "two_muscle_simulation.cu"

	for itest in range(args[tests_number]):
		logger.info(f"running test #{itest}")
		cmd_build = f"{nvcc} -o {build_folder}/{buildname} {build_folder}/{buildfile}"
		cmd_run = f"{build_folder}/{buildname} {args[cms]} {args[ees]} {args[inh]} {args[ped]} {args[ht5]} {itest} 0"

		for cmd in [cmd_build, cmd_run]:
			logger.info(f"Execute: {cmd}")
			process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			out, err = process.communicate()

			for output in str(out.decode("UTF-8")).split("\n"):
				logger.info(output)
			for error in str(err.decode("UTF-8")).split("\n"):
				logger.info(error)


def convert_to_hdf5(result_folder):
	"""
	Converts dat files into hdf5 with compression
	Args:
		result_folder (str): folder where is the dat files placed
	"""
	# process only files with these muscle names
	for muscle in ["MN_E", "MN_F"]:
		logger.info(f"converting {muscle} dat files to hdf5")
		is_datfile = lambda f: f.endswith(f"{muscle}.dat")
		datfiles = filter(is_datfile, os.listdir(result_folder))
		# prepare hdf5 file for writing data per test
		with hdf5.File(f"{result_folder}/{muscle}.hdf5", 'w') as hdf5_file:
			for test_index, filename in enumerate(datfiles):
				with open(f"{result_folder}/{filename}") as datfile:
					data = [-float(v) for v in datfile.readline().split()]
					# check on NaN values (!important)
					if any(map(np.isnan, data)):
						logging.info(f"{filename} has NaN... skip")
						continue
					hdf5_file.create_dataset(f"{test_index}", data=data, compression="gzip")
		# check that hdf5 file was written properly
		with hdf5.File(f"{result_folder}/{muscle}.hdf5") as hdf5_file:
			assert all(map(len, hdf5_file.values()))


def plot_results(save_folder, ees_hz=40, sim_step=0.025):
	"""
	Plot hdf5 results by invoking special function of plotting shadows based on boxplots
	Args:
		save_folder (str): folder of hdf5 results and folder for saving current plots
		ees_hz (int): value of EES in Hz
		sim_step (float): simulation step (0.025 is standard)
	"""
	# for each hdf5 file get its data and plot
	for filename in filter(lambda f: f.endswith(".hdf5"), os.listdir(save_folder)):
		title = os.path.splitext(filename)[0]
		logging.info(f"start plotting {filename}")
		with hdf5.File(f"{save_folder}/{filename}") as hdf5_file:
			listed_data = np.array([data[:] for data in hdf5_file.values()])
			from time import time
			start = time()
			plot_shadows_boxplot(listed_data, ees_hz, sim_step, save_folder=save_folder, filename=title)
			end = time()
			print(end - start)


def testrunner():
	script_place = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/"
	save_folder = f"{script_place}/dat"

	args = {tests_number: 5,
	        cms: 21,
	        ees: 40,
	        inh: 100,
	        ped: 2,
	        ht5: 0}

	run_tests(script_place, args)
	# convert_to_hdf5(save_folder)
	# plot_results(save_folder, ees_hz=args[ees])


if __name__ == "__main__":
	testrunner()
