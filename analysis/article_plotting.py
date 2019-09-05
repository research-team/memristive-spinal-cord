import os
import ntpath
import logging
import numpy as np
import pylab as plt
from analysis.functions import auto_prepare_data, get_boxplots, calc_boxplots
from analysis.PCA import plot_3D_PCA, get_lat_per_exp, get_amp_per_exp, get_peak_per_exp

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()

k_median = 0
k_fliers_high = 5
k_fliers_low = 6
bar_width = 0.9


def plot_slices(extensor_data, flexor_data, e_latencies, f_latencies, step_size, folder, filename):
	"""
	TODO: add docstring
	Args:
		extensor_data (np.ndarray): values of extensor motoneurons membrane potential
		flexor_data (np.ndarray): values of flexor motoneurons membrane potential
		e_latencies (np.ndarray): extensor latencies of poly answers per slice
		f_latencies (np.ndarray): flexor latencies of poly answers per slice
		step_size (float): data step
		folder (str): save folder path
		filename (str): name of the future path
	"""
	bio_ideal_y_data = None
	lighted_filename = "_".join(filename.split("_")[:-1])
	# Parsing example: bio_F_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5
	meta = filename.split("_")
	meta_type = meta[1].lower()
	meta_speed = meta[2]
	ideal_filename = f"{meta_type}_{meta_speed}"

	steps_in_slice = len(extensor_data[0][0])
	slice_in_ms = len(extensor_data[0][0]) * step_size

	e_latencies = (e_latencies / step_size).astype(int)
	f_latencies = (f_latencies / step_size).astype(int)

	e_slices_number = len(extensor_data[0])
	f_slices_number = len(flexor_data[0])

	e_box_latencies = np.array([calc_boxplots(dots) for dots in e_latencies.reshape(len(extensor_data), -1).T])
	f_box_latencies = np.array([calc_boxplots(dots) for dots in f_latencies.reshape(len(flexor_data), -1).T])

	e_splitted_per_slice_boxplots = get_boxplots(extensor_data)
	f_splitted_per_slice_boxplots = get_boxplots(flexor_data)

	all_splitted_per_slice_boxplots = list(e_splitted_per_slice_boxplots) + list(f_splitted_per_slice_boxplots)

	yticks = []

	slices_number = e_slices_number + f_slices_number
	colors = iter(['#287a72', '#f2aa2e', '#472650'] * slices_number)

	#
	e_slices_indexes = range(e_slices_number)
	f_slices_indexes = range(e_slices_number + 1, e_slices_number + f_slices_number + 1)

	# plot an example of good pattern for bio data or median for sim data of current mode
	if "bio_" in filename and os.path.exists(f"{folder}/{ideal_filename}"):
		bio_ideal_y_data = []
		# collect extensor data
		with open(f"{folder}/{ideal_filename}") as file:
			for d in file.readlines():
				bio_ideal_y_data.append(list(map(float, d.split())))
		# collect flexor_data
		with open(f"{folder}/{ideal_filename.replace('e_', 'f_')}") as file:
			for d in file.readlines():
				bio_ideal_y_data.append(list(map(float, d.split())))
		# convert list to array for more simplicity using
		bio_ideal_y_data = np.array(bio_ideal_y_data)

	fig, ax = plt.subplots(figsize=(16, 9))
	# plot latency shadow for extensor
	plt.fill_betweenx(e_slices_indexes,
	                  e_box_latencies[:, 1] * step_size,
	                  e_box_latencies[:, 2] * step_size, color='#275B78', alpha=0.3, zorder=3)
	# plot latency shadow for flexor (+1 to slice index because of space between extensor and flexor)
	plt.fill_betweenx(f_slices_indexes,
	                  f_box_latencies[:, 1] * step_size,
	                  f_box_latencies[:, 2] * step_size, color='#275B78', alpha=0.3, zorder=3)
	# plot fliers and median line
	for slice_index, data in enumerate(all_splitted_per_slice_boxplots):
		offset = slice_index + (0 if slice_index < e_slices_number else 1)
		data += offset
		shared_x = np.arange(len(data[:, k_fliers_high])) * step_size
		plt.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color=next(colors), alpha=0.7, zorder=3)
		if bio_ideal_y_data is None:
			plt.plot(shared_x, data[:, k_median], color='k', linewidth=3, zorder=4)
		else:
			plt.plot(shared_x, bio_ideal_y_data[slice_index] + offset, color='k', linewidth=3, zorder=4)
		yticks.append(data[:, k_median][0])

	plt.plot(e_box_latencies[:, 0] * step_size, e_slices_indexes, linewidth=5, color='#A6261D', zorder=6, alpha=1)
	plt.plot(f_box_latencies[:, 0] * step_size, f_slices_indexes, linewidth=5, color='#A6261D', zorder=6, alpha=1)

	xticks = range(int(slice_in_ms) + 1)
	if len(xticks) <= 35:
		xticklabels = [x if i % 5 == 0 else None for i, x in enumerate(xticks)]
	else:
		xticklabels = [x if i % 25 == 0 else None for i, x in enumerate(xticks)]
	yticklabels = [None] * slices_number
	slice_indexes = range(1, slices_number + 1)
	for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
		yticklabels[i] = slice_indexes[i]

	plt.xticks(xticks, xticklabels, fontsize=56)
	plt.yticks(yticks, yticklabels, fontsize=56)
	plt.xlim(0, slice_in_ms)

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.tight_layout()
	# plt.savefig(f"{folder}/{lighted_filename}.pdf", dpi=250, format="pdf")
	plt.show()
	plt.close()


def recolor(boxplot_elements, color, fill_color):
	"""
	Add colors to bars (setup each element)
	Args:
		boxplot_elements (dict):
			components of the boxplot
		color (str):
			HEX color of outside lines
		fill_color (str):
			HEX color of filling
	"""
	for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
		plt.setp(boxplot_elements[element], color=color, linewidth=3)
	plt.setp(boxplot_elements["fliers"], markeredgecolor=color)
	for patch in boxplot_elements['boxes']:
		patch.set(facecolor=fill_color)


def plot_histograms(lat_per_slice, amp_per_slice, peaks_per_slice, dataset, folder, filename, step_size):
	"""
	TODO: add docstring
	Args:
		amp_per_slice (np.ndarray): amplitudes per slice
		peaks_per_slice (np.ndarray): number of peaks per slice
		lat_per_slice (np.ndarray): latencies per slice
		dataset (np.ndarray): data per test run
		folder (str): folder path
		filename (str): filename of the future file
		step_size (float):
	"""
	box_distance = 1.2
	color = "#472650"
	fill_color = "#9D8DA3"
	slices_number = len(lat_per_slice)
	slice_indexes = np.array(range(slices_number))

	lighted_filename = "_".join(filename.split("_")[:-1])

	xticks = [x * box_distance for x in slice_indexes]
	# set labels
	xticklabels = [None] * len(slice_indexes)
	human_read = [i + 1 for i in slice_indexes]
	for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
		xticklabels[i] = human_read[i]

	# plot histograms
	for data, title in (amp_per_slice, "amplitudes"), (peaks_per_slice, "peaks"):
		# create subplots
		fig, ax = plt.subplots(figsize=(16, 9))
		# plot amplitudes or peaks
		plt.bar(xticks, data, width=bar_width, color=color, zorder=2)
		# set Y ticks
		yticks = ax.get_yticks()
		human_read = list(yticks)
		yticklabels = [None] * len(yticks)
		for i in [0, -1, int(1 / 3 * len(yticks)), int(2 / 3 * len(yticks))]:
			yticklabels[i] = int(human_read[i]) if human_read[i] >= 10 else f"{human_read[i]:.1f}"
		# plot properties
		plt.grid(axis="y")
		plt.xticks(xticks, xticklabels, fontsize=56)
		plt.yticks(yticks, yticklabels, fontsize=56)
		plt.xlim(-0.5, len(slice_indexes) * box_distance - bar_width / 2)
		plt.tight_layout()
		plt.savefig(f"{folder}/{lighted_filename}_{title}.pdf", dpi=250, format="pdf")
		plt.close()

		log.info(f"Plotted {title} for {filename}")

	# form areas
	raise NotImplemented

	splitted_per_slice_boxplots = get_boxplots(dataset)
	mono_area = [slice_data[:int(time / step_size)] for time, slice_data in splitted_per_slice_boxplots]
	poly_area = [slice_data[int(time / step_size):] for time, slice_data in zip(lat_per_slice, splitted_per_slice_boxplots)]

	# plot per area
	for data_test_runs, title in (mono_area, "mono"), (poly_area, "poly"):
		area_data = []
		data_test_runs = np.array(data_test_runs)
		# calc diff per slice
		for slice_data in data_test_runs:
			area_data.append(abs(slice_data[:, k_fliers_high] - slice_data[:, k_fliers_low]))

		fig, ax = plt.subplots(figsize=(16, 9))

		fliers = dict(markerfacecolor='k', marker='*', markersize=3)
		# plot latencies
		plt.xticks(fontsize=56)
		plt.yticks(fontsize=56)

		lat_plot = ax.boxplot(area_data, positions=xticks, widths=bar_width, patch_artist=True, flierprops=fliers)
		recolor(lat_plot, color, fill_color)

		yticks = np.array(ax.get_yticks())
		yticks = yticks[yticks >= 0]
		human_read = list(yticks)
		yticklabels = [None] * len(yticks)
		for i in [0, -1, int(1 / 3 * len(yticks)), int(2 / 3 * len(yticks))]:
			if human_read[i] >= 10:
				yticklabels[i] = int(human_read[i])
			else:
				yticklabels[i] = f"{human_read[i]:.1f}"
		# plot properties
		plt.xticks(xticks, xticklabels, fontsize=56)
		plt.yticks(yticks, yticklabels, fontsize=56)
		plt.grid(axis="y")
		plt.xlim(-0.5, len(slice_indexes) * box_distance - bar_width / 2)
		plt.ylim(0, ax.get_yticks()[-1])
		plt.tight_layout()
		plt.savefig(f"{folder}/{lighted_filename}_{title}.pdf", dpi=250, format="pdf")
		plt.close()

		log.info(f"Plotted {title} for {filename}")


def __process_dataset(filepaths, save_pca_to, plot_histogram_flag=False, plot_slices_flag=False,
                      plot_pca_flag=False, plot_correlation=False, step_size_to=0.1):
	"""
	ToDo add info
	Args:
		filepaths (list):
		plot_histogram_flag (bool):
		plot_slices_flag (bool):
		plot_pca_flag (bool):
	"""
	all_pack = []
	colors = iter(["#275b78", "#287a72", "#f2aa2e", "#472650", "#a6261d", "#f27c2e", "#2ba7b9"] * 10)

	# process each file
	for filepath in filepaths:
		folder = ntpath.dirname(filepath)
		filename = ntpath.basename(filepath)
		data_label = filename.replace('.hdf5', '')
		# get prepared data, EES frequency and data step size
		e_prepared_data = auto_prepare_data(folder, filename, step_size_to=step_size_to)
		# process latencies and amplitudes per slice
		e_lat_per_slice = get_lat_per_exp(e_prepared_data, step_size_to)
		amp_per_slice = get_amp_per_exp(e_prepared_data, step_size_to)
		peaks_per_slice = get_peak_per_exp(e_prepared_data, step_size_to, split_by_intervals=True)
		# form data pack
		coords_meta = (np.stack((e_lat_per_slice, amp_per_slice, peaks_per_slice), axis=1), next(colors), data_label)
		all_pack.append(coords_meta)
		# plot histograms of amplitudes and number of peaks
		if plot_histogram_flag:
			plot_histograms(e_lat_per_slice, amp_per_slice, peaks_per_slice, e_prepared_data,
			                folder=folder, filename=filename, step_size=step_size_to)
		# plot all slices with pattern
		if plot_slices_flag:
			flexor_filename = filename.replace('_E_', '_F_')
			f_prepared_data = auto_prepare_data(folder, flexor_filename, step_size_to=step_size_to)
			f_lat_per_slice = get_lat_per_exp(f_prepared_data, step_size_to)
			plot_slices(e_prepared_data, f_prepared_data, e_lat_per_slice, f_lat_per_slice,
			            folder=folder, filename=filename, step_size=step_size_to)
	# plot 3D PCA for each plane
	if plot_pca_flag or plot_correlation:
		plot_3D_PCA(all_pack, save_to=save_pca_to, correlation=plot_correlation)


def for_article():
	"""
	TODO: add docstring
	"""
	save_pca_to = '/home/alex/GitHub/DATA/gras/hz'

	compare_pack = [
		'/home/alex/GitHub/DATA/bio/foot/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5',
		# '/home/alex/GitHub/DATA/neuron/foot/neuron_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5',
		# '/home/alex/GitHub/DATA/gras/foot/gras_E_15cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5',
	]

	# control
	step_size_to = 0.1
	plot_pca_flag = True
	plot_correlation = True
	plot_slices_flag = True
	plot_histogram_flag = False

	__process_dataset(compare_pack, save_pca_to, plot_histogram_flag,
	                  plot_slices_flag, plot_pca_flag, plot_correlation, step_size_to)


def run():
	for_article()


if __name__ == "__main__":
	run()
