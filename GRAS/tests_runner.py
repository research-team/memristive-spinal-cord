import os
import time
import logging
import subprocess
import numpy as np
import h5py as hdf5
import pylab as plt

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
					data = list(map(float, dat_file.readline().split()))
					if any(map(lambda x: np.isnan(x), data)):
						logging.info(f"{muscle} in {test_index} has NaN... skip")
						continue
					hdf5_file.create_dataset(f"{test_index}", data=data, compression="gzip")
		# check HDF5
		with hdf5.File(f'{result_folder}/{muscle}.hdf5') as hdf5_file:
			for data in hdf5_file.values():
				assert len(data) > 0


def boxplot_shadows(data_per_test, ees_hz=40, step=0.1, save_folder=None, filename="shadows", debugging=False):
	"""
	Plot shadows (and/or save) based on the input data
	Args:
		data_per_test (list of list): data per test with list of points
		step (float): step size of the data for human-read normalization time
		debugging (bool): show debug info
		save_folder (str): folder path to save the graphic
		filename (str): filename
	Returns:
		kawai pictures =(^-^)=
	"""
	if len(data_per_test) == 0:
		raise Exception("Empty input data")
	if type(data_per_test[0]) is not list:
		raise Exception("Not valid input data -- should be list of lists [[],[] ... []]")

	# stuff variables
	slice_time_length = 25
	slices_number = int(len(data_per_test[0]) / slice_time_length * step)
	steps_in_slice = int(slice_time_length / step)
	shared_x = [x * step for x in range(steps_in_slice)]
	# swap rows and columns
	data_per_step = list(zip(*data_per_test))

	plt.figure(figsize=(16, 9))

	# y ticks for slices
	yticks = []
	# process each slice
	for slice_index in range(slices_number):
		logging.info("plot slice #{}".format(slice_index + 1))
		# set offset for Y
		y_offset = slice_index * 40
		# get data for curent slice
		sliced_data = data_per_step[slice_index * steps_in_slice:(slice_index + 1) * steps_in_slice]

		# calculate fliers, whiskers and medians (thanks to pylab <3)
		tmp_fig = plt.figure()
		boxplot_data = plt.boxplot(sliced_data, showfliers=True, showcaps=True)
		plt.close(tmp_fig)

		# get the necessary data
		medians = boxplot_data['medians']
		whiskers_data = boxplot_data['whiskers']
		fliers = boxplot_data['fliers']

		whiskers_data_high = whiskers_data[1::2]
		whiskers_data_low = whiskers_data[::2]

		# check on equal size
		assert len(whiskers_data_low) == len(whiskers_data_high)
		assert len(whiskers_data_low) == steps_in_slice
		assert len(whiskers_data_low) == len(fliers)

		# debug info
		if debugging:
			# back to previous data structure
			data_per_test = list(zip(*sliced_data))
			# plot for each test
			for data_per_test in data_per_test:
				plt.plot(shared_x, [y_offset + y for y in data_per_test], color='g')

		# calc Y for median
		median_y = [y_offset + median.get_ydata()[0] for median in medians]
		# calc Y for boxes
		boxes_y_high = [y_offset + whisker.get_ydata()[0] for whisker in whiskers_data_high]
		boxes_y_low = [y_offset + whisker.get_ydata()[0] for whisker in whiskers_data_low]
		# calc Y for whiskers
		whiskers_y_high = [y_offset + whisker.get_ydata()[1] for whisker in whiskers_data_high]
		whiskers_y_low = [y_offset + whisker.get_ydata()[1] for whisker in whiskers_data_low]
		# calc Y for fliers
		fliers_y_max = []
		fliers_y_min = []
		# compute each flier point
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

		# plot fliers shadow (fliers top or bottom)
		plt.fill_between(shared_x, fliers_y_min, fliers_y_max, alpha=0.1, color='r')

		# plot whiskers shadow (whiskers top or bottom)
		plt.fill_between(shared_x, whiskers_y_low, whiskers_y_high, alpha=0.3, color='r')

		# plot boxes shadow (like a boxes -- top or bottom, but it is still whisker)
		plt.fill_between(shared_x, boxes_y_low, boxes_y_high, alpha=0.7, color='r')

		# plot median
		plt.plot(shared_x, median_y, color='k')

		# add Y tick value
		yticks.append(median_y[0])

	# plot stuff
	plt.xticks(range(26), range(26))
	plt.xlim(0, 25)
	plt.yticks(yticks, range(1, slices_number + 1))

	if save_folder is None:
		save_folder = os.getcwd()

	logging.info("Save file: {}/{}.png".format(save_folder, filename))

	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
	plt.savefig("{}/{}.png".format(save_folder, filename), dpi=512, format="png")

	if debugging:
		plt.show()

	plt.close()


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
		boxplot_shadows(slimmed_data_per_test, ees_hz=ees_hz, step=step, save_folder=save_folder, filename=title)
		break


def testrunner():
	tests_number = 25
	script_place = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/kek"
	save_folder = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/dat"

	# run_tests(script_place, tests_number)
	convert_to_hdf5(save_folder, tests_number)
	plot_results(save_folder, step=0.1, ees_hz=40)


if __name__ == "__main__":
	testrunner()
