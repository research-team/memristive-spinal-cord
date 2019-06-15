import os
import time
import logging
import subprocess
import numpy as np
import h5py as hdf5
import pylab as plt


logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()


def run_tests(script_place, tests_number):
	for test_index in range(tests_number):
		logger.info(f"run test #{test_index}")

		cmd_run = f"{script_place}/kek {test_index} 0"

		start_time = time.time()

		process = subprocess.Popen(cmd_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
		output, error = process.communicate()

		end_time = time.time()

		logger.info(f"test #{test_index} elapsed {end_time - start_time:.2f}s")

		if len(error) > 0:
			error_text = str(error.decode("UTF-8")).split("\n")
			for error in error_text:
				logger.info(error)


def convert_to_hdf5(result_folder):
	for muscle in ["MN_E", "MN_F"]:
		logger.info(f"converting {muscle} dat files to hdf5")
		is_datfile = lambda f: f.endswith(f"{muscle}.dat")
		datfiles = filter(is_datfile, os.listdir(result_folder))

		with hdf5.File(f"{result_folder}/{muscle}.hdf5", 'w') as hdf5_file:
			for test_index, filename in enumerate(datfiles):
				with open(f"{result_folder}/{filename}") as dat_file:
					data = [-float(v) for v in dat_file.readline().split()]
					if any(map(np.isnan, data)):
						logging.info(f"{filename} has NaN... skip")
						continue
					hdf5_file.create_dataset(f"{test_index}", data=data, compression="gzip")
		# check hdf5
		with hdf5.File(f"{result_folder}/{muscle}.hdf5") as hdf5_file:
			assert all(map(len, hdf5_file.values()))


def calc_boxplots(data_per_slice):
	medians = []
	boxes_y_high = []
	boxes_y_low = []
	whiskers_y_high = []
	whiskers_y_low = []
	fliers_high = []
	fliers_low = []

	for dots in data_per_slice:
		low_box_Q1, median, high_box_Q3 = np.percentile(dots, [25, 50, 75])
		# calc borders
		IQR = high_box_Q3 - low_box_Q1
		Q1_15 = low_box_Q1 - 1.5 * IQR
		Q3_15 = high_box_Q3 + 1.5 * IQR
		# median and high/low box
		medians.append(median)
		boxes_y_high.append(high_box_Q3)
		boxes_y_low.append(low_box_Q1)
		# whisker maximal
		high_whisker = [dot for dot in dots if high_box_Q3 < dot < Q3_15]
		high_whisker = max(high_whisker) if high_whisker else high_box_Q3
		whiskers_y_high.append(high_whisker)
		# whisker minimal
		low_whisker = [dot for dot in dots if Q1_15 < dot < low_box_Q1]
		low_whisker = min(low_whisker) if low_whisker else low_box_Q1
		whiskers_y_low.append(low_whisker)
		# flier maximal
		high_flier = [dot for dot in dots if dot > Q3_15]
		fliers_high.append(max(high_flier) if high_flier else high_whisker)
		# flier minimal
		low_flier = [dot for dot in dots if dot < Q1_15]
		fliers_low.append(min(low_flier) if low_flier else low_whisker)

	return medians, boxes_y_high, boxes_y_low, whiskers_y_high, whiskers_y_low, fliers_high, fliers_low


def boxplot_shadows(data_per_test, ees_hz, step, save_folder=None, filename=None, debugging=False):
	"""
	Plot shadows (and/or save) based on the input data
	Args:
		data_per_test (list of list): data per test with list of dots
		ees_hz (int): EES value
		step (float): step size of the data for human-read normalization time
		save_folder (str): saving folder path
		filename (str): filename
		debugging (bool): show debug info
	Returns:
		kawai pictures =(^-^)=
	"""
	if save_folder is None:
		save_folder = os.getcwd()

	# stuff variables
	slice_length_ms = int(1 / ees_hz * 1000)
	slices_number = int(len(data_per_test[0]) / slice_length_ms * step)
	steps_in_slice = int(slice_length_ms / step)
	# tests dots at each time -> N (test number) dots at each time
	bars_per_step = zip(*data_per_test)
	data_per_slice = map(calc_boxplots, zip(*[iter(bars_per_step)] * steps_in_slice))

	# build plot
	yticks = []
	shared_x = [x * step for x in range(steps_in_slice)]

	fig, ax = plt.subplots(figsize=(16, 9))
	# plot each slice
	for slice_index, bp_data_per_slice in enumerate(data_per_slice, 1):
		y_offset = slice_index * 30
		med, b_high, b_low, w_high, w_low, f_high, f_low = bp_data_per_slice
		ax.fill_between(shared_x, [y + y_offset for y in f_low], [y + y_offset for y in f_high], alpha=0.1, color='r')
		ax.fill_between(shared_x, [y + y_offset for y in w_low], [y + y_offset for y in w_high], alpha=0.3, color='r')
		ax.fill_between(shared_x, [y + y_offset for y in b_low], [y + y_offset for y in b_high], alpha=0.6, color='r')
		ax.plot(shared_x, [y + y_offset for y in med], color='k', linewidth=0.7)
		yticks.append(med[0] + y_offset)
	# plotting stuff
	ax.set_xlim(0, slice_length_ms)
	ax.set_xticks(range(slice_length_ms + 1))
	ax.set_xticklabels(range(slice_length_ms + 1))
	ax.set_yticks(yticks)
	ax.set_yticklabels(range(1, slices_number + 1))
	fig.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99)
	fig.savefig(f"{save_folder}/shadow_{filename}.png", dpi=250, format="png")

	if debugging:
		plt.show()

	plt.close()

	logging.info(f"saved file in {save_folder}")


def plot_results(save_folder, ees_hz=40, sim_step=0.025):
	for filename in filter(lambda f: f.endswith(".hdf5"), os.listdir(save_folder)):
		logging.info(f"start plotting {filename}")
		with hdf5.File(f"{save_folder}/{filename}") as hdf5_file:
			listed_data = [data[:] for data in hdf5_file.values()]
			title = os.path.splitext(filename)[0]
			boxplot_shadows(listed_data, ees_hz, sim_step, save_folder=save_folder, filename=title)


def testrunner():
	tests_number = 25
	script_place = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/"
	save_folder = f"{script_place}/dat"

	# run_tests(script_place, tests_number)
	convert_to_hdf5(save_folder)
	plot_results(save_folder, sim_step=0.025, ees_hz=40)


if __name__ == "__main__":
	testrunner()
