import os
import time
import logging
import subprocess
import numpy as np
import h5py as hdf5
import pylab as plt
from multiprocessing import Pool

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('Converter')


def measure_time(func):
	def wrapper(*args, **kwargs):
		start_time = time.time()
		func(*args, **kwargs)
		end_time = time.time()
		print(f"Elapsed {end_time - start_time:.2f} s")
	return wrapper


@measure_time
def run_tests(script_place, tests_number):
	for test_index in range(tests_number):
		logger.info(f"Run test #{test_index}")
		command = f"{script_place} {test_index} 0"
		process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
		output, error = process.communicate()

		error_text = str(error.decode("UTF-8")).split("\n")
		if error_text:
			for error in error_text:
				logger.info(error)


def convert_to_hdf5(result_folder, tests_number):
	for muscle in ["MN_E", "MN_F"]:
		logger.info("writing data to the HDF5")
		with hdf5.File(f'{result_folder}/{muscle}.hdf5', 'w') as hdf5_file:
			for test_index in range(tests_number):
				logger.info(f"process test #{test_index}")
				with open(f"{result_folder}/{test_index}_{muscle}.dat") as dat_file:
					data = list(map(lambda x: -float(x), dat_file.readline().split()))
					if any(map(lambda x: np.isnan(x), data)):
						logging.info(f"{muscle} in {test_index} has NaN... skip")
						continue
					hdf5_file.create_dataset(f"{test_index}", data=data, compression="gzip")
		# check HDF5
		with hdf5.File(f'{result_folder}/{muscle}.hdf5') as hdf5_file:
			for data in hdf5_file.values():
				assert len(data) > 0


def boxplot_processing(data):
	"""
	Special function for preparing data from the boxplot (medians, whiskers, fliers)
	Args:
		data:
	Returns:
		tuple: data per parameter
	"""
	# extract the data
	slice_index = data[0]
	slice_data = data[1]
	# set offset for Y
	y_offset = slice_index * 30
	# calculate fliers, whiskers and medians (thanks to pylab <3)
	boxplot_data = plt.boxplot(slice_data, showfliers=True, showcaps=True)
	# get the necessary data
	medians = boxplot_data['medians']
	whiskers_data = boxplot_data['whiskers']
	fliers = boxplot_data['fliers']
	# separate data
	whiskers_data_high = whiskers_data[1::2]
	whiskers_data_low = whiskers_data[::2]

	# check on equal size
	assert len(whiskers_data_low) == len(whiskers_data_high)
	assert len(whiskers_data_low) == len(fliers)

	get_y0 = lambda bp_data: bp_data.get_ydata()[0]
	get_y1 = lambda bp_data: bp_data.get_ydata()[1]

	# calc Y for median
	median_y = list(map(lambda median: y_offset + get_y0(median), medians))
	# calc Y for boxes
	boxes_y_high = list(map(lambda whisker: y_offset + get_y0(whisker), whiskers_data_high))
	boxes_y_low = list(map(lambda whisker: y_offset + get_y0(whisker), whiskers_data_low))
	# calc Y for whiskers
	whiskers_y_high = list(map(lambda whisker: y_offset + get_y1(whisker), whiskers_data_high))
	whiskers_y_low = list(map(lambda whisker: y_offset + get_y1(whisker), whiskers_data_low))

	# calc Y for fliers, compute each flier point
	fliers_y_max = []
	fliers_y_min = []
	for index, flier in enumerate(fliers):
		lowest_whisker = whiskers_y_low[index]
		highest_whisker = whiskers_y_high[index]
		flier_y_data = flier.get_ydata()
		# if more than 1 dot
		if len(flier_y_data) > 1:
			flier_max = max(flier_y_data) + y_offset
			flier_min = min(flier_y_data) + y_offset
			fliers_y_max.append(highest_whisker if flier_max < highest_whisker else flier_max)
			fliers_y_min.append(lowest_whisker if flier_min > lowest_whisker else flier_min)
		# if only 1 dot
		elif len(flier_y_data) == 1:
			fliers_y_max.append(max(flier_y_data[0] + y_offset, highest_whisker))
			fliers_y_min.append(min(flier_y_data[0] + y_offset, lowest_whisker))
		# no dots in flier -- use whiskers
		else:
			fliers_y_max.append(highest_whisker)
			fliers_y_min.append(lowest_whisker)

	logging.info(f"calculated slice #{slice_index + 1}")

	return fliers_y_min, fliers_y_max, whiskers_y_low, whiskers_y_high, boxes_y_low, boxes_y_high, median_y


def boxplot_shadows(data_per_test, ees_hz, step, save_folder=None, filename=None, debugging=False):
	"""
	Plot shadows (and/or save) based on the input data
	Args:
		data_per_test (list of list): data per test with list of dots
		step (float): step size of the data for human-read normalization time
		ees_hz (int): EES value
		save_folder (str): saving folder path
		filename (str): filename
		debugging (bool): show debug info
	Returns:
		kawai pictures =(^-^)=
	"""
	if len(data_per_test) == 0:
		raise Exception("Empty input data")

	yticks = []         # y ticks for slices
	prepared_data = []  # prepare data for parallelizing

	# stuff variables
	slice_time_length = int(1 / ees_hz * 1000)
	slices_number = int(len(data_per_test[0]) / slice_time_length * step)
	steps_in_slice = int(slice_time_length / step)
	# tests dots at each time -> N (test number) dots at each time
	data_per_step = list(zip(*data_per_test))
	shared_x = [x * step for x in range(steps_in_slice)]

	for slice_index in range(slices_number):
		dots_per_slice = data_per_step[slice_index * steps_in_slice:(slice_index + 1) * steps_in_slice]
		prepared_data.append((slice_index, dots_per_slice))

	# parallelized calculations
	with Pool(processes=4) as pool:
		all_data = pool.map(boxplot_processing, prepared_data)

	# build plot
	plt.figure(figsize=(16, 9))

	for slice_index, bp_data_per_slice in enumerate(all_data, 1):
		flow, fhigh, wlow, whigh, blow, bhigh, med = bp_data_per_slice
		plt.fill_between(shared_x, flow, fhigh, alpha=0.1, color='r')   # fliers shadow
		plt.fill_between(shared_x, wlow, whigh, alpha=0.25, color='r')  # whiskers shadow
		plt.fill_between(shared_x, blow, bhigh, alpha=0.5, color='r')   # boxes shadow
		plt.plot(shared_x, med, color='k', linewidth=0.7)               # median line
		yticks.append(med[0])
		logging.info(f"plotted slice #{slice_index}/{slices_number}")

	# plotting stuff
	plt.xticks(range(slice_time_length + 1), range(slice_time_length + 1))
	plt.xlim(0, slice_time_length)
	plt.yticks(yticks, range(1, slices_number + 1))

	if save_folder is None:
		save_folder = os.getcwd()

	abs_saving_path = f"{save_folder}/shadow_{filename}.png"

	plt.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99)
	plt.savefig(abs_saving_path, dpi=250, format="png")

	if debugging:
		plt.show()

	plt.close()

	logging.info(f"Saved file at {abs_saving_path}")


@measure_time
def plot_results(save_folder, step=0.1, ees_hz=40):
	chunk_size = int(step / 0.025)
	hdf5_filenames = [filename for filename in os.listdir(save_folder) if filename.endswith(".hdf5")]

	for filename in hdf5_filenames:
		title = "".join(filename.split(".")[:-1])
		with hdf5.File(f'{save_folder}/{filename}') as hdf5_file:
			slimmed_data_per_test = []
			for data in hdf5_file.values():
				slimmed_data_per_test.append([np.mean(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)])
		boxplot_shadows(slimmed_data_per_test, ees_hz, step, save_folder=save_folder, filename=title)


def testrunner():
	tests_number = 25
	script_place = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/kek"
	save_folder = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/dat"

	run_tests(script_place, tests_number)
	convert_to_hdf5(save_folder, tests_number)
	plot_results(save_folder, step=0.1, ees_hz=40)


if __name__ == "__main__":
	testrunner()
