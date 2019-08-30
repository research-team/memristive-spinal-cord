import logging
import numpy as np
import pylab as plt
from analysis.functions import calc_boxplots
from analysis.functions import split_by_slices

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()


def plot_shadows_boxplot(data_per_test, ees_hz, step, save_folder, filename, debugging=False):
	"""
	Plot shadows (and/or save) based on the input data
	Args:
		data_per_test (np.ndarray): data per test with list of dots
		ees_hz (int): EES value
		step (float): step size of the data for human-read normalization time
		save_folder (str): saving folder path
		filename (str): filename
		debugging (bool): show debug info
	Returns:
		kawai pictures =(^-^)=
	"""
	yticks = []
	k_median = 0
	k_flier_low = 5
	k_flier_high = 6
	# stuff variables
	data_per_test = np.array(data_per_test)
	# calc number of steps in one slice
	# WARNING: to get correct splitting, you need to round slice length in ms and do the same for number of steps
	steps_in_slice = int(int(1000 / ees_hz) / step)
	# 'slice in ms' represented as float type (for plotting)
	slice_in_ms = steps_in_slice * step
	# calc boxplots per iter in each N-runs
	boxplots = np.array([calc_boxplots(dot) for dot in data_per_test.T])
	# shrink on equal parts
	splitted_boxplots = split_by_slices(boxplots, steps_in_slice)
	splitted_original = np.array([split_by_slices(d, steps_in_slice) for d in data_per_test])

	slices_number = len(splitted_boxplots)
	shared_x = np.arange(steps_in_slice) * step

	plt.subplots(figsize=(16, 9))
	for slice_index, slice_data in enumerate(splitted_boxplots):
		slice_data += slice_index * 20
		plt.fill_between(shared_x, slice_data[:, k_flier_low], slice_data[:, k_flier_high], alpha=0.8)
		# plt.fill_between(shared_x, data[:, 4], data[:, 3], alpha=0.3)  # 4 w_low, 3 w_high
		# plt.fill_between(shared_x, data[:, 2], data[:, 1], alpha=0.6)  # 2 b_low, 1 b_high
		plt.plot(shared_x, slice_data[:, k_median], color='k', linewidth=0.3, linestyle='--')
		# setup the slice index position by first point
		yticks.append(slice_data[0, 0])
		# plot original data as thin colorized lines
		for exp_run in splitted_original:
			orig = exp_run[slice_index] + slice_index * 20
			plt.plot(shared_x, orig, linewidth=0.1)

	# plotting stuff
	plt.xlim(0, slice_in_ms)
	plt.xticks(np.arange(slice_in_ms) + 1, np.arange(slice_in_ms).astype(int) + 1)
	plt.yticks(yticks, range(1, slices_number + 1))
	plt.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99)
	plt.savefig(f"{save_folder}/shadow_{filename}.png", dpi=250, format="png")

	if debugging:
		plt.show()

	plt.close()

	logging.info(f"saved file in {save_folder}")
