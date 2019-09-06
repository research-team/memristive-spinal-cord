import os
import ntpath
import logging
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, MultipleLocator

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


def axis_article_style(ax, axis='both', auto_nbins=False, xshift=None, xmin=None, xmax=None):
	"""
	ToDo add info
	Args:
		ax:
		axis:
		auto_nbins:
		xshift:
		xmin:
		xmax:
	"""
	# hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	# make ticks more visible
	ax.tick_params(which='major', length=10, width=3, labelsize=50)
	ax.tick_params(which='minor', length=4, width=2, labelsize=50)
	# set automatical locator for chosen axis
	if axis == 'x' or axis == 'both':
		ax.xaxis.set_minor_locator(MultipleLocator())
		if auto_nbins:
			ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
		else:
			ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=len(ax.get_xticks()), integer=True))

		if xshift or xmin or xmax:
			if xshift:
				xticks = (ax.get_xticks() + 1).astype(int)
			else:
				xticks = ax.get_xticks()
			xmax = np.inf if xmax is None else xmax
			xmin = -np.inf if xmin is None else xmin
			xticklabels = [int(label) if xmin <= label <= xmax else "" for label in xticks]
			ax.set_xticklabels(xticklabels)

	if axis == 'y' or axis == 'both':
		ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=len(ax.get_yticks()), integer=True))
		ax.yaxis.set_minor_locator(MultipleLocator())


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
	pattern_clr = "#275B78"
	bio_ideal_y_data = None
	new_filename = "_".join(filename.split("_")[:-1])
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

	# use 1:1 ratio for 6cms data and 4:3 for others
	fig, ax = plt.subplots(figsize=(20, 20) if "6cms" in filename else (16, 12))
	# plot latency shadow for extensor
	ax.fill_betweenx(e_slices_indexes,
	                 e_box_latencies[:, k_box_low] * step_size,
	                 e_box_latencies[:, k_box_high] * step_size, color=pattern_clr, alpha=0.3, zorder=3)
	# plot latency shadow for flexor (+1 to slice index because of space between extensor and flexor)
	ax.fill_betweenx(f_slices_indexes,
	                 f_box_latencies[:, k_box_low] * step_size,
	                 f_box_latencies[:, k_box_high] * step_size, color=pattern_clr, alpha=0.3, zorder=3)
	yticks = []
	shared_x = np.arange(steps_in_slice) * step_size
	# plot fliers and median line
	for slice_index, data in enumerate(all_splitted_per_slice_boxplots):
		offset = slice_index + (0 if slice_index < e_slices_number else 1)
		data += offset
		# fliers shadow
		ax.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color=next(colors), alpha=0.7, zorder=3)
		# median line or ideal pattern for bio data
		if bio_ideal_y_data is None:
			ax.plot(shared_x, data[:, k_median], color='k', linewidth=3, zorder=4)
		else:
			ax.plot(shared_x, bio_ideal_y_data[slice_index] + offset, color='k', linewidth=3, zorder=4)
		yticks.append(data[:, k_median][0])
	# plot pattern based on median latency per slice
	ax.plot(e_box_latencies[:, 0] * step_size, e_slices_indexes, linewidth=5, color='#A6261D', zorder=6, alpha=1)
	ax.plot(f_box_latencies[:, 0] * step_size, f_slices_indexes, linewidth=5, color='#A6261D', zorder=6, alpha=1)

	axis_article_style(ax, axis='x')

	yticklabels = [None] * slices_number
	slice_indexes = range(1, slices_number + 1)
	for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
		yticklabels[i] = slice_indexes[i]
	plt.yticks(yticks, yticklabels, fontsize=50)
	plt.xlim(0, slice_in_ms)
	plt.tight_layout()
	plt.savefig(f"{save_to}/{new_filename}.pdf", dpi=250, format="pdf")
	plt.close()
	log.info(f"saved to {save_to}/{new_filename}")


def plot_lat_amp_dependency(pack_datas_per_exp, pack_lats_per_exp, names, step_size, save_to):
	cap_size = 0.3
	patches = []
	colors = iter(["#a6261d", "#f2aa2e", "#275b78", "#472650"])
	# form the new name of the file
	mode = "_".join(names[0].split("_")[1:-1])
	names = "_".join((name.split("_")[0] for name in names))
	new_filename = f"{names}_{mode}_dependency.pdf"
	# for the best visual result use an 1:1 ratio for 6cms data and 15:5 for others
	fig, ax = plt.subplots(figsize=(20, 20) if "6cms" in mode else (15, 5))
	#
	for pack_index, pack_data in enumerate(pack_datas_per_exp):
		pack_color = next(colors)
		reshaped_lat_of_pack = pack_lats_per_exp[pack_index].reshape(len(pack_data), -1).T
		box_latencies = np.array([calc_boxplots(dots) for dots in reshaped_lat_of_pack])
		#
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
			# plot a caps
			x = lat_med
			ax.plot([x - cap_size, x + cap_size], [amp_from_low_lat, amp_from_low_lat], linewidth=3, color=pack_color)
			ax.plot([x - cap_size, x + cap_size], [amp_from_high_lat, amp_from_high_lat], linewidth=3, color=pack_color)
			# plot a median dot
			ax.plot(x, amp_from_med_lat, '.', markersize=15, color=pack_color)
			# plot a line
			ax.plot([x, x], [amp_from_low_lat, amp_from_high_lat], linewidth=3, color=pack_color)
		# form the legend
		patches.append(mpatches.Patch(color=pack_color, label=f"{names[pack_index]}"))
	# use article from
	axis_article_style(ax, auto_nbins=True)
	plt.legend(handles=patches)
	plt.tight_layout()
	plt.savefig(f"{save_to}/{new_filename}", dpi=250, format="pdf")
	plt.close()
	log.info(f"saved to {save_to}/{new_filename}")


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
	dist_coef = 2
	bar_width = 0.1
	colors = ["#275b78", "#287a72", "#f2aa2e", "#472650", "#a6261d"]
	intervals = np.array([[0, 3], [7, 10], [10, 15], [15, 20], [20, 25]]) / step_size
	# form the new name of the file
	mode = "_".join(names[0].split("_")[1:-1])
	names = "_".join((name.split("_")[0] for name in names))
	new_filename = f"{names}_{mode}_bar.pdf"
	slices_number = len(pack_peaks_per_interval[0])
	fig, ax = plt.subplots(figsize=(15, 5))
	# calc each pack -- pack is a simulation (NEURON/GRAS/NEST) or bio data
	for pack_index, pack_data in enumerate(pack_peaks_per_interval):
		for slice_index, slice_peaks in enumerate(pack_data):
			bottom = 0
			# plot stacked bars (intervals are stacked, slices separated)
			for interval_index, interval_data in enumerate(slice_peaks):
				plt.bar(slice_index + dist_coef * pack_index * bar_width, interval_data,
				        bottom=bottom, color=colors[interval_index], width=bar_width, alpha=0.9)
				bottom += interval_data
	# reshape intervals for forming legend patches
	intervals[0] += intervals[-1][-1]
	c = intervals[0, :].copy()
	intervals[0:-1, :] = intervals[1:, :]
	intervals[-1] = c
	for interval_index, interval_data in enumerate(intervals[::-1] * step_size):
		patches.append(mpatches.Patch(color=colors[::-1][interval_index], label=f'{interval_data}'))
	plt.legend(handles=patches)

	# use an article style axis
	axis_article_style(ax, xshift=1, xmin=1, xmax=slices_number)

	plt.tight_layout()
	plt.savefig(f"{save_to}/{new_filename}", dpi=250, format="pdf")
	plt.close()
	log.info(f"saved to {save_to}/{new_filename}")


def __process_dataset(filepaths, save_to, flags, step_size_to=0.1):
	"""
	ToDo add info
	Args:
		filepaths (list):
		plot_slices_flag (bool):
		plot_pca_flag (bool):
	"""
	pca_pack = []
	e_lat_pack = []
	e_names_pack = []
	e_data_pack = []
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
		if flags['plot_peaks_by_intervals']:
			e_peaks_per_interval = get_peak_per_exp(e_prepared_data, step_size=step_size_to, split_by_intervals=True)
			e_peaks_per_interval_pack.append(e_peaks_per_interval)

		# process latencies, amplitudes, peaks (per dataset per slice)
		e_latencies = get_lat_per_exp(e_prepared_data, step_size_to)
		e_amplitudes = get_amp_per_exp(e_prepared_data, step_size_to)
		e_peaks = get_peak_per_exp(e_prepared_data, step_size_to)

		# form PCA data pack
		if flags['plot_pca_flag'] or flags['plot_correlation']:
			coords_meta = (np.stack((e_latencies, e_amplitudes, e_peaks), axis=1), next(colors), data_label)
			pca_pack.append(coords_meta)

		# plot slices with pattern
		if flags['plot_slices_flag']:
			# get flexor prepared data (centered, normalized, subsampled and sliced)
			flexor_filename = filename.replace('_E_', '_F_')
			f_prepared_data = auto_prepare_data(folder, flexor_filename, step_size_to=step_size_to)
			f_lat_per_slice = get_lat_per_exp(f_prepared_data, step_size_to)
			#
			plot_slices(e_prepared_data, f_prepared_data,
			            e_latencies, f_lat_per_slice,
			            folder=folder, save_to=save_to, filename=filename, step_size=step_size_to)

		# fill for plot latency/amplitude dependency
		e_data_pack.append(e_prepared_data)
		e_lat_pack.append(e_latencies)
		e_names_pack.append(filename)

	if flags['plot_pca_flag'] or flags['plot_correlation']:
		plot_3D_PCA(pca_pack, save_to=save_to, correlation=flags['plot_correlation'])

	if flags['plot_lat_amp_dep']:
		plot_lat_amp_dependency(e_data_pack, e_lat_pack, e_names_pack, step_size=step_size_to, save_to=save_to)

	if flags['plot_peaks_by_intervals']:
		plot_peaks_bar_intervals(e_peaks_per_interval_pack, e_names_pack, step_size=step_size_to, save_to=save_to)


def for_article():
	"""
	TODO: add docstring
	"""
	save_all_to = '/home/alex/GitHub/DATA/'

	compare_pack = [
		'/home/alex/GitHub/DATA/bio/foot/bio_E_6cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5',
		'/home/alex/GitHub/DATA/neuron/foot/neuron_E_6cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5',
		'/home/alex/GitHub/DATA/gras/foot/gras_E_6cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5',
		# '/home/alex/GitHub/DATA/nest/foot/nest_E_21cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5',
	]

	# control
	step_size_to = 0.1
	flags = dict(plot_pca_flag=True,
	             plot_correlation=False,
	             plot_slices_flag=False,
	             plot_lat_amp_dep=False,
	             plot_peaks_by_intervals=False)

	__process_dataset(compare_pack, save_all_to, flags, step_size_to)


def run():
	for_article()


if __name__ == "__main__":
	run()
