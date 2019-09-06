import os
import ntpath
import logging
import numpy as np
import pylab as plt
import matplotlib.patches as mpatches

from analysis.functions import auto_prepare_data, get_boxplots, calc_boxplots
from analysis.PCA import plot_3D_PCA, get_lat_per_exp, get_amp_per_exp, get_peak_per_exp

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()

k_median = 0
k_box_high = 1
k_box_low = 2
k_fliers_high = 5
k_fliers_low = 6


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


def plot_slices(extensor_data, flexor_data, e_latencies, f_latencies, step_size, folder, save_to, filename):
	"""
	TODO: add docstring
	Args:
		extensor_data (np.ndarray): values of extensor motoneurons membrane potential
		flexor_data (np.ndarray): values of flexor motoneurons membrane potential
		e_latencies (np.ndarray): extensor latencies of poly answers per slice
		f_latencies (np.ndarray): flexor latencies of poly answers per slice
		step_size (float): data step
		folder (str): original folder path
		save_to (str): save folder path
		filename (str): name of the future path
	"""
	bio_ideal_y_data = None
	lighted_filename = "_".join(filename.split("_")[:-1])
	# Parsing example: bio_F_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5
	meta = filename.split("_")
	meta_type = meta[1].lower()
	meta_speed = meta[2]
	ideal_filename = f"{meta_type}_{meta_speed}"
	# for human read plotting
	steps_in_slice = len(extensor_data[0][0])
	slice_in_ms = steps_in_slice * step_size
	# convert ms latencies to steps
	e_latencies = (e_latencies / step_size).astype(int)
	f_latencies = (f_latencies / step_size).astype(int)
	# get number of slices per muscle
	e_slices_number = len(extensor_data[0])
	f_slices_number = len(flexor_data[0])
	# calc boxplots of latencies per slice (reshape 1D array to 2D based on experiments number)
	e_box_latencies = np.array([calc_boxplots(dots) for dots in e_latencies.reshape(len(extensor_data), -1).T])
	f_box_latencies = np.array([calc_boxplots(dots) for dots in f_latencies.reshape(len(flexor_data), -1).T])
	# calc boxplots of original data
	e_splitted_per_slice_boxplots = get_boxplots(extensor_data)
	f_splitted_per_slice_boxplots = get_boxplots(flexor_data)
	# combine
	all_splitted_per_slice_boxplots = list(e_splitted_per_slice_boxplots) + list(f_splitted_per_slice_boxplots)


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


	fig, ax = plt.subplots(figsize=(20, 20) if "6cms" in filename else (16, 9))
	# plot latency shadow for extensor
	plt.fill_betweenx(e_slices_indexes,
	                  e_box_latencies[:, k_box_low] * step_size,
	                  e_box_latencies[:, k_box_high] * step_size, color='#275B78', alpha=0.3, zorder=3)
	# plot latency shadow for flexor (+1 to slice index because of space between extensor and flexor)
	plt.fill_betweenx(f_slices_indexes,
	                  f_box_latencies[:, k_box_low] * step_size,
	                  f_box_latencies[:, k_box_high] * step_size, color='#275B78', alpha=0.3, zorder=3)
	yticks = []
	shared_x = np.arange(steps_in_slice) * step_size
	# plot fliers and median line
	for slice_index, data in enumerate(all_splitted_per_slice_boxplots):
		offset = slice_index + (0 if slice_index < e_slices_number else 1)
		data += offset
		# fliers shadow
		plt.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color=next(colors), alpha=0.7, zorder=3)
		# median line or ideal pattern for bio data
		if bio_ideal_y_data is None:
			plt.plot(shared_x, data[:, k_median], color='k', linewidth=3, zorder=4)
		else:
			plt.plot(shared_x, bio_ideal_y_data[slice_index] + offset, color='k', linewidth=3, zorder=4)
		yticks.append(data[:, k_median][0])
	# plot pattern based on median latency per slice
	plt.plot(e_box_latencies[:, 0] * step_size, e_slices_indexes, linewidth=5, color='#A6261D', zorder=6, alpha=1)
	plt.plot(f_box_latencies[:, 0] * step_size, f_slices_indexes, linewidth=5, color='#A6261D', zorder=6, alpha=1)
	# plot properties
	xticks = range(int(slice_in_ms) + 1)
	xtick_step = 5 if len(xticks) <= 35 else 25
	xticklabels = [x if i % xtick_step == 0 else None for i, x in enumerate(xticks)]
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
	plt.savefig(f"{save_to}/{lighted_filename}.pdf", dpi=250, format="pdf")
	plt.close()


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
	bar_width = 0.9
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
		plt.xticks(fontsize=50)
		plt.yticks(fontsize=50)

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


def plot_lat_amp_dependency(pack_datas_per_exp, pack_lats_per_exp, names, step_size):
	cap_size = 0.3
	patches = []
	colors = iter(["#a6261d", "#f2aa2e", "#275b78", "#472650"])
	for pack_index, pack_data in enumerate(pack_datas_per_exp):
		pack_color = next(colors)
		box_latencies = np.array([calc_boxplots(dots) for dots in pack_lats_per_exp[pack_index].reshape(len(pack_data), -1).T])

		for slice_index, latencies_per_slices in enumerate(box_latencies):
			slice_of_exp = pack_data[:, slice_index]
			#
			lat_high = box_latencies[slice_index, k_box_high]
			lat_med = box_latencies[slice_index, k_median]
			lat_low = box_latencies[slice_index, k_box_low]
			#
			amp_from_low_lat = np.sum(np.abs(slice_of_exp[:, int(lat_low / step_size): ])) / len(pack_data)
			amp_from_med_lat = np.sum(np.abs(slice_of_exp[:, int(lat_med / step_size): ])) / len(pack_data)
			amp_from_high_lat = np.sum(np.abs(slice_of_exp[:, int(lat_high / step_size): ])) / len(pack_data)

			x = lat_med
			# plt caps
			plt.plot([x - cap_size, x + cap_size], [amp_from_low_lat, amp_from_low_lat], linewidth=1, color=pack_color)
			plt.plot([x - cap_size, x + cap_size], [amp_from_high_lat, amp_from_high_lat], linewidth=1, color=pack_color)
			# plt dot
			plt.plot(x, amp_from_med_lat, '.', markersize='10', color=pack_color)
			# plt line
			plt.plot([x, x], [amp_from_low_lat, amp_from_high_lat], linewidth=1, color=pack_color)

		patches.append(mpatches.Patch(color=pack_color, label=f"{names[pack_index]}"))

	plt.xlabel("Latency")
	plt.ylabel("Amplitude")
	plt.xticks(range(10, 26), range(10, 26))
	plt.legend(handles=patches)
	plt.show()


def plot_peaks_bar_intervals(pack_peaks_per_interval, names, step_size, save_to):
	"""
	ToDo add info
	Args:
		pack_peaks_per_interval (list of np.ndarrays): grouped datasets by different sim/bio data
		step_size (float): data step size
		names (list): filenames of grouped datasets
		save_to (str): path for saving a built picture
	"""
	patches = []
	dist_coef = 3
	bar_width = 0.5
	pack_size = len(pack_peaks_per_interval)
	slices_number = len(pack_peaks_per_interval[0])
	colors = ["#275b78", "#287a72", "#f2aa2e", "#472650", "#a6261d"]
	intervals = np.array([[0, 3], [7, 10], [10, 15], [15, 20], [20, 25]]) / step_size
	# form the new name of the file
	mode = "_".join(names[0].split("_")[1:-1])
	names = "_".join((name.split("_")[0] for name in names))
	new_path = f"{save_to}/{names}_{mode}_bar.pdf"

	plt.figure(figsize=(15, 5))
	# calc each pack -- pack is a simulation (NEURON/GRAS/NEST) or bio data
	for pack_index, pack_data in enumerate(pack_peaks_per_interval):
		for slice_index, slice_peaks in enumerate(pack_data):
			bottom = 0
			# plot stacked bars (intervals are stacked, slices separated)
			for interval_index, interval_data in enumerate(slice_peaks):
				plt.bar(slice_index * dist_coef + 2 * pack_index / pack_size, interval_data,
				        bottom=bottom, color=colors[interval_index], width=bar_width, alpha=0.9)
				bottom += interval_data
	# form patches for legend (intervals and their colors)
	intervals[0] += intervals[-1][-1]
	c = intervals[0, :].copy()
	intervals[0:-1, :] = intervals[1:, :]
	intervals[-1] = c
	for interval_index, interval_data in enumerate(intervals[::-1] * step_size):
		patches.append(mpatches.Patch(color=colors[::-1][interval_index], label=f'{interval_data}'))
	plt.legend(handles=patches)

	xticks = np.arange(slices_number) * dist_coef + 2 * bar_width

	xticklabels = [None] * slices_number
	slice_indexes = range(1, slices_number + 1)
	for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
		xticklabels[i] = slice_indexes[i]

	plt.xticks(xticks, xticklabels, fontsize=50)
	plt.yticks(fontsize=50)
	plt.xlim(xticks[0] - 3 * bar_width, xticks[-1] + 3 * bar_width)
	plt.tight_layout()
	plt.savefig(new_path, dpi=250, format="pdf")
	log.info(f"Saved to {new_path}")
	plt.close()


def __process_dataset(filepaths, save_to, plot_slices_flag=False, plot_pca_flag=False,
                      plot_correlation=False, plot_peaks_by_intervals=False,
                      plot_lat_amp_dep=False, step_size_to=0.1):
	"""
	ToDo add info
	Args:
		filepaths (list):
		plot_slices_flag (bool):
		plot_pca_flag (bool):
	"""
	pca_pack = []
	e_latencies_pack = []
	e_filenames_pack = []
	e_prepared_data_pack = []
	e_peaks_per_interval_pack = []
	colors = iter(["#275b78", "#287a72", "#f2aa2e", "#472650", "#a6261d", "#f27c2e", "#2ba7b9"] * 10)

	# process each file
	for filepath in filepaths:
		folder = ntpath.dirname(filepath)
		filename = ntpath.basename(filepath)
		data_label = filename.replace('.hdf5', '')
		# get extensor prepared data (centered, normalized, subsampled and sliced)
		e_prepared_data = auto_prepare_data(folder, filename, step_size_to=step_size_to)

		# form data pack for peaks per interval
		if plot_peaks_by_intervals:
			e_peaks_per_interval = get_peak_per_exp(e_prepared_data, step_size=step_size_to, split_by_intervals=True)
			e_peaks_per_interval_pack.append(e_peaks_per_interval)

		# process latencies, amplitudes, peaks (per dataset per slice)
		e_latencies = get_lat_per_exp(e_prepared_data, step_size_to)
		e_amplitudes = get_amp_per_exp(e_prepared_data, step_size_to)
		e_peaks = get_peak_per_exp(e_prepared_data, step_size_to)

		# form PCA data pack
		if plot_pca_flag or plot_correlation:
			coords_meta = (np.stack((e_latencies, e_amplitudes, e_peaks), axis=1), next(colors), data_label)
			pca_pack.append(coords_meta)

		# plot slices with pattern
		if plot_slices_flag:
			# get flexor prepared data (centered, normalized, subsampled and sliced)
			flexor_filename = filename.replace('_E_', '_F_')
			f_prepared_data = auto_prepare_data(folder, flexor_filename, step_size_to=step_size_to)
			f_lat_per_slice = get_lat_per_exp(f_prepared_data, step_size_to)
			#
			plot_slices(e_prepared_data, f_prepared_data,
			            e_latencies, f_lat_per_slice,
			            folder=folder, save_to=save_to, filename=filename, step_size=step_size_to)

		# fillfor plot_lat_amp_depend
		e_prepared_data_pack.append(e_prepared_data)
		e_latencies_pack.append(e_latencies)
		e_filenames_pack.append(filename)

	if plot_pca_flag or plot_correlation:
		plot_3D_PCA(pca_pack, save_to=save_to, correlation=plot_correlation)

	if plot_lat_amp_dep:
		plot_lat_amp_dependency(e_prepared_data_pack, e_latencies_pack, e_filenames_pack, step_size=step_size_to)

	if plot_peaks_by_intervals:
		plot_peaks_bar_intervals(e_peaks_per_interval_pack, e_filenames_pack, step_size=step_size_to, save_to=save_to)


def for_article():
	"""
	TODO: add docstring
	"""
	save_all_to = '/home/alex/GitHub/DATA/'

	compare_pack = [
		'/home/alex/GitHub/DATA/bio/foot/bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5',
		'/home/alex/GitHub/DATA/neuron/foot/neuron_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5',
		'/home/alex/GitHub/DATA/gras/foot/gras_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5',
		# '/home/alex/GitHub/DATA/nest/foot/nest_E_21cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5',
	]

	# control
	step_size_to = 0.1
	plot_pca_flag = False
	plot_correlation = False
	plot_slices_flag = False
	plot_lat_amp_dep = False
	plot_peaks_by_intervals = True

	__process_dataset(compare_pack, save_all_to, plot_slices_flag, plot_pca_flag, plot_correlation,
	                  plot_peaks_by_intervals, plot_lat_amp_dep, step_size_to)


def run():
	for_article()


if __name__ == "__main__":
	run()
