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
	k_flier_low = 5
	k_flier_high = 6
	# stuff variables
	data_per_test = np.array(data_per_test)

	slice_length_ms = int(1000 / ees_hz)
	steps_in_slice = int(slice_length_ms / step)

	boxplots = np.array([calc_boxplots(dot) for dot in data_per_test.T])

	splitted_boxplots = split_by_slices(boxplots, steps_in_slice)
	splitted_original = np.array([split_by_slices(d, steps_in_slice) for d in data_per_test])

	shared_x = np.arange(steps_in_slice) * step
	plt.subplots(figsize=(16, 9))
	for i, d in enumerate(splitted_boxplots):
		d += i * 20
		plt.fill_between(shared_x, d[:, 6], d[:, 5], alpha=0.8)  # 6 f_low, 5 f_high
	plt.show()
	slices_number = len(splitted_boxplots)

	# build plot
	yticks = []
	shared_x = np.arange(steps_in_slice) * step

	fig, ax = plt.subplots(figsize=(16, 9))

	for i, data in enumerate(splitted_boxplots):
		data += i * 30
		ax.fill_between(shared_x, data[:, 6], data[:, 5], alpha=0.8)  # 6 f_low, 5 f_high
		# ax.fill_between(shared_x, data[:, 4], data[:, 3], alpha=0.3)  # 4 w_low, 3 w_high
		# ax.fill_between(shared_x, data[:, 2], data[:, 1], alpha=0.6)  # 2 b_low, 1 b_high
		ax.plot(shared_x, data[:, 0], color='k', linewidth=0.3, linestyle='--')  # 0 med

		for exp_run in splitted_original:
			orig = exp_run[i] + i * 30
			ax.plot(shared_x, orig, linewidth=0.1)
		yticks.append(data[0, 0])

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
