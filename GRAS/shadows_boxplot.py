import logging
import numpy as np
import pylab as plt
from analysis.functions import calc_boxplots

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
	# stuff variables
	data_per_test = np.array(data_per_test)

	slice_length_ms = int(1 / ees_hz * 1000)
	slices_number = int(len(data_per_test[0]) / slice_length_ms * step)
	steps_in_slice = int(slice_length_ms / step)

	# tests dots at each time -> N (test number) dots at each time
	splitted = np.split(np.array([calc_boxplots(dot) for dot in data_per_test.T]), slices_number)
	original_splitted = np.split(data_per_test.T, slices_number)

	# build plot
	yticks = []
	shared_x = np.arange(steps_in_slice) * step

	fig, ax = plt.subplots(figsize=(16, 9))

	for i, data in enumerate(splitted):
		data += i * 30
		ax.fill_between(shared_x, data[:, 6], data[:, 5], alpha=0.8)  # 6 f_low, 5 f_high
		# ax.fill_between(shared_x, data[:, 4], data[:, 3], alpha=0.3)  # 4 w_low, 3 w_high
		# ax.fill_between(shared_x, data[:, 2], data[:, 1], alpha=0.6)  # 2 b_low, 1 b_high
		ax.plot(shared_x, data[:, 0], color='k', linewidth=0.3, linestyle='--')  # 0 med
		orig = original_splitted[i] + i * 30
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
