import os
import ntpath
import logging
import numpy as np
import pylab as plt
from itertools import chain
from matplotlib import gridspec
from scipy.stats import ks_2samp
from sklearn.utils import resample
import matplotlib.patches as mpatches
from analysis.functions import ks2d2s, peacock2
from matplotlib.ticker import MaxNLocator, MultipleLocator
from analysis.functions import auto_prepare_data, get_boxplots, calc_boxplots
from analysis.PCA import plot_3D_PCA, get_lat_matrix, joint_plot, contour_plot, get_area_extrema_matrix, get_all_peak_amp_per_slice

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
		boxplot_elements (dict): components of the boxplot
		color (str): HEX color of outside lines
		fill_color (str): HEX color of filling
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
		ax (matplotlib.axes): currect figure axes
		axis (str): which axes change, both -- all
		auto_nbins (bool):
		xshift (None or float): offset of xticklabels
		xmin (None or float): set xlim for minimum
		xmax (None or float): set xlim for maximum
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
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		else:
			ax.xaxis.set_major_locator(MaxNLocator(nbins=len(ax.get_xticks()), integer=True))

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
		ax.yaxis.set_major_locator(MaxNLocator(nbins=len(ax.get_yticks()), integer=True))
		ax.yaxis.set_minor_locator(MultipleLocator(base=(ax.get_yticks()[1] - ax.get_yticks()[0]) / 2))


def plot_slices(extensor_data, flexor_data, e_latencies, f_latencies, dstep, save_to, filename, best_sample):
	"""
	TODO: add docstring
	Args:
		extensor_data (np.ndarray): values of extensor motoneurons membrane potential
		flexor_data (np.ndarray): values of flexor motoneurons membrane potential
		e_latencies (np.ndarray): extensor latencies of poly answers per slice
		f_latencies (np.ndarray): flexor latencies of poly answers per slice
		dstep (float): data step
		save_to (str): save folder path
		filename (str): name of the future path
		best_sample (int of np.ndarray): dataset index or np.ndarray of a best example
	"""
	pattern_color = "#275B78"
	new_filename = "_".join(filename.split("_")[:-1])
	# for human read plotting
	steps_in_slice = len(extensor_data[0][0])
	slice_in_ms = steps_in_slice * dstep
	# get number of slices per muscle
	e_slices_number = len(extensor_data[0])
	f_slices_number = len(flexor_data[0])
	slices_number = e_slices_number + f_slices_number
	colors = iter(['#287a72', '#f2aa2e', '#472650'] * slices_number)
	# calc boxplots of latencies per slice (reshape 1D array to 2D based on experiments number)
	e_box_latencies = np.array([calc_boxplots(dots) for dots in e_latencies.T])
	f_box_latencies = np.array([calc_boxplots(dots) for dots in f_latencies.T])
	# calc boxplots of original data
	e_splitted_per_slice_boxplots = get_boxplots(extensor_data)
	f_splitted_per_slice_boxplots = get_boxplots(flexor_data)
	# combine data into one list
	all_splitted_per_slice_boxplots = list(e_splitted_per_slice_boxplots) + list(f_splitted_per_slice_boxplots)
	# form slices indexes for a pattern shadows
	e_slices_indexes = range(e_slices_number)
	f_slices_indexes = range(e_slices_number + 1, e_slices_number + f_slices_number + 1)
	# use 1:1 ratio for 6cms data and 4:3 for others
	fig, ax = plt.subplots(figsize=(20, 20) if "6cms" in filename else (16, 12))
	# plot latency shadow for extensor
	ax.fill_betweenx(e_slices_indexes,
	                 e_box_latencies[:, k_box_low] * dstep,
	                 e_box_latencies[:, k_box_high] * dstep, color=pattern_color, alpha=0.3, zorder=3)
	yticks = []
	shared_x = np.arange(steps_in_slice) * dstep
	# plot fliers and median line
	for slice_index, data in enumerate(all_splitted_per_slice_boxplots):
		offset = slice_index + (1 if slice_index >= e_slices_number else 0)
		# set ideal or median
		if type(best_sample) is int:
			if slice_index >= e_slices_number:
				ideal_data = flexor_data[best_sample][slice_index - e_slices_number]
			else:
				ideal_data = extensor_data[best_sample][slice_index]
		else:
			ideal_data = best_sample[slice_index]
		data += offset
		ideal_data += offset
		# fliers shadow
		ax.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color=next(colors), alpha=0.7, zorder=3)
		# ideal pattern
		ax.plot(shared_x, ideal_data, color='k', linewidth=3, zorder=4)
		yticks.append(ideal_data[0])

	# plot pattern based on median latency per slice
	ax.plot(e_box_latencies[:, 0] * dstep, e_slices_indexes, linewidth=5, color='#A6261D', zorder=6, alpha=1)
	# form ticks
	axis_article_style(ax, axis='x')
	# plot settings
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


def plot_ks2d(peaks_times_pack, peaks_ampls_pack, names, colors, dstep, save_to):
	"""
	ToDo add info
	Args:
		peaks_times_pack (list): pack of peak lists for different origin of data (bio/neuron/gras/nest)
		peaks_ampls_pack (list): pack of amplitude lists for different origin of data (bio/neuron/gras/nest)
		names (list): filenames for different origin of data
		colors (list): hex colors for graphics
		dstep (float): data step size (common for all)
		save_to (str): save folder
	"""
	ks_test_datas = []

	mode = "_".join(names[0].split("_")[1:-1])
	datasets = "_".join(name.split("_")[0] for name in names)
	new_filename = f"{datasets}_{mode}_dependency.pdf"

	# define grid for subplots
	gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])
	fig = plt.figure()
	kde_ax = plt.subplot(gs[1, 0])
	kde_ax.spines['top'].set_visible(False)
	kde_ax.spines['right'].set_visible(False)

	biggest_dsize = max(map(len, peaks_times_pack))

	# 1D and 2D Kolmogorov-Smirnov test
	for peaks_times, peaks_ampls, name, color in zip(peaks_times_pack, peaks_ampls_pack, names, colors):
		x = peaks_times
		y = peaks_ampls

		diff_size = biggest_dsize - len(x)
		if diff_size != 0:
			boot_x = resample(x, replace=True, n_samples=diff_size, random_state=1)
			boot_y = resample(y, replace=True, n_samples=diff_size, random_state=0)
			x = np.append(x, boot_x)
			y = np.append(y, boot_y)

		print(f"dsize {len(x)} (bootstrapped {diff_size} samples)")
		ks_test_datas.append((x, y))

	x1, y1 = ks_test_datas[0]
	x2, y2 = ks_test_datas[1]

	dvalue, pvalue = ks_2samp(x1, x2)
	print(f"1D K-S peaks TIME -> D-value: {dvalue}, p-value: {pvalue}")

	dvalue, pvalue = ks_2samp(y1, y2)
	print(f"1D K-S peaks AMPLITUDE -> D-value: {dvalue}, p-value: {pvalue}")

	dvalue, pvalue = ks2d2s(x1, y1, x2, y2)
	print(f"2D K-S TIME/AMPLITUDE -> D-value: {dvalue}, p-value: {pvalue}")

	d1 = np.stack((x1, y1), axis=1)
	d2 = np.stack((x2, y2), axis=1)
	dvalue, pvalue = peacock2(d1, d2)
	print(f"2D peacock TIME/AMPLITUDE -> D-value: {dvalue}, p-value: {pvalue}")

	# plot 2D
	for data, name, color in zip(ks_test_datas, names, colors):
		x, y = data
		contour_plot(x=x, y=y, color=color, ax=kde_ax)
		joint_plot(x, y, kde_ax, gs, **{"color": color})

	kde_ax.set_xlabel("peak time (ms)")
	kde_ax.set_ylabel("peak amplitude")
	kde_ax.set_xlim(0, 25)
	kde_ax.set_ylim(0, 2)
	plt.tight_layout()
	plt.show()
	# plt.savefig(f"{save_to}/{new_filename}_kde2d.pdf", dpi=250, format="pdf")
	# plt.savefig(f"{save_to}/{new_filename}_kde2d.png", dpi=250, format="png")
	plt.close(fig)

	log.info(f"saved to {save_to}/{new_filename}")


def plot_peaks_bar_intervals(pack_peaks_per_interval, names, save_to):
	"""
	ToDo add info
	Args:
		pack_peaks_per_interval (list of np.ndarrays): grouped datasets by different sim/bio data
		names (list): filenames of grouped datasets
		save_to (str): path for saving a built picture
	"""
	yticks = []
	datasets_number = len(names)
	dist_coef = 2
	bar_height = 0.1
	coef = 0.3 * datasets_number
	colors = ["#275b78", "#f27c2e", "#f2aa2e", "#472650", "#a6261d", "#287a72", "#2ba7b9"] * 10
	ms_intervals = np.array([[0, 3], [7, 10], [10, 15], [15, 20], [20, 25]])
	# form the new name of the file
	mode = "_".join(names[0].split("_")[1:-1])
	merged_names = "_".join(name.split("_")[0] for name in names)
	new_filename = f"{merged_names}_{mode}_bar.pdf"

	fig, ax = plt.subplots(figsize=(15, 5))

	for pack_index, pack_data in enumerate(pack_peaks_per_interval):
		# set color based on filename
		if "bio" in names[pack_index]:
			marker = '✚'
		elif "neuron" in names[pack_index]:
			marker = '★'
		elif "gras" in names[pack_index]:
			marker = '▴'
		elif "nest" in names[pack_index]:
			marker = '⚫'
		else:
			raise Exception("Can not set marker for data!")
		# plot bar chart
		for interval_index, interval_data in enumerate(pack_data.T):
			left = 0
			# plot stacked bars (intervals are stacked, slices separated)
			y_pos = interval_index * coef + dist_coef * pack_index * bar_height
			if pack_index == 0:
				yticks.append(y_pos)
			for slice_index, slice_data in enumerate(interval_data):
				plt.barh(y=y_pos, width=slice_data, left=left, color=colors[slice_index], height=bar_height, alpha=0.9)
				left += slice_data
			plt.text(x=left + 0.1, y=y_pos - bar_height / 4, s=marker)

	yticks = np.array(yticks) + datasets_number * bar_height - bar_height
	yticklabels = [f"{interval[0]}-{interval[1]}" for interval in ms_intervals]
	plt.yticks(yticks, yticklabels)
	axis_article_style(ax, axis='x')
	plt.yticks(fontsize=30)
	plt.tight_layout()
	plt.savefig(f"{save_to}/{new_filename}", dpi=250, format="pdf")
	plt.close()
	log.info(f"saved to {save_to}/{new_filename}")


def example_bio_sample(folder, filename):
	"""
	Return y-data of best bio sample. File must exists
	Args:
		folder (str): current folder with hdf5 files and best sample
		filename (str): best sample filename
	Returns:
		np.ndarray: y-data of best sample
	"""
	meta = filename.split("_")
	meta_type = meta[1].lower()
	meta_speed = meta[2]
	ideal_filename = f"{meta_type}_{meta_speed}"

	if not os.path.exists(f"{folder}/{ideal_filename}"):
		raise Exception(f"Where is best sample for bio data?! I can't find it here '{folder}'")

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
	ideal_sample = np.array(bio_ideal_y_data)

	return ideal_sample


def example_sample(latencies_matrix, peaks_matrix, step_size):
	"""
	ToDo add info
	Args:
		latencies_matrix (np.ndarray): 
		peaks_matrix (np.ndarray): 
		step_size (float): data step size 
	Returns:
		int: index of sample
	"""
	ideal_example_index = 0
	peaks_sum = np.sum(peaks_matrix, axis=1)
	index = np.arange(len(peaks_sum))
	merged = np.array(list(zip(index, peaks_sum)))
	# at the top located experimental runs with the greatest number of peaks
	sorted_by_sum = merged[merged[:, 1].argsort()][::-1]
	for index, value in sorted_by_sum:
		index = int(index)
		# check difference between latencies -- how far they are from each other
		diff = np.diff(latencies_matrix[index] * step_size, n=1)
		# acceptable border is -3 .. 3 ms
		if all(map(lambda x: -3 <= x <= 3, diff)):
			ideal_example_index = index
			break
	return ideal_example_index


def get_color(filename, clrs):
	if "bio" in filename:
		color = next(clrs)
	elif "gras" in filename:
		color = '#287a72'
	elif "neuron" in filename:
		color = '#F2AA2E'
	elif "nest" in filename:
		color = '#472650'
	else:
		raise Exception("Can't set color for data")
	return color


def __process_dataset(filepaths, save_to, flags, dstep_to=None):
	"""
	ToDo add info
	Args:
		filepaths (list of str): absolute paths to the files
		save_to (str): save folder
		flags (dict): pack of flags
		dstep_to (float): data step size
	"""
	ks1d_ampls = []
	ks1d_peaks = []
	ks1d_names = []
	ks1d_colors = []
	ks3d_pack = []
	peaks_per_interval_pack = []
	colors = iter(['#A6261D', '#F2AA2E', '#287a72', '#472650'])

	# be sure that folder is exist
	if not os.path.exists(save_to):
		os.makedirs(save_to)

	# process each file
	for filepath in filepaths:
		folder = ntpath.dirname(filepath)
		e_filename = ntpath.basename(filepath)
		data_label = e_filename.replace('.hdf5', '')
		# default file's step size if dstep is not provided
		if dstep_to is None:
			dstep_to = float(data_label.replace('step', '').split("_")[-1])
		# set color based on filename
		color = get_color(e_filename, colors)
		# get extensor/flexor prepared data (centered, normalized, subsampled and sliced)
		e_prepared_data = auto_prepare_data(folder, e_filename, dstep_to=dstep_to)

		# for 1D or 2D Kolmogorod-Smirnov test (without pattern)
		e_peak_times_per_slice, e_peak_ampls_per_slice = get_all_peak_amp_per_slice(e_prepared_data, dstep_to)
		flatten = chain.from_iterable
		times = np.array(list(flatten(flatten(e_peak_times_per_slice)))) * dstep_to
		ampls = np.array(list(flatten(flatten(e_peak_ampls_per_slice))))

		ks1d_peaks.append(times)
		ks1d_ampls.append(ampls)
		ks1d_names.append(e_filename)
		ks1d_colors.append(color)

		# for 3D Kolmogorod-Smirnov test (with pattern)
		e_lat_matrix = get_lat_matrix(e_prepared_data, dstep_to)
		e_peak_sum_matrix, e_ampl_sum_matrix = get_area_extrema_matrix(e_prepared_data, e_lat_matrix, dstep_to)
		coords = np.stack((e_lat_matrix.flatten() * dstep_to,
		                   e_ampl_sum_matrix.flatten(),
		                   e_peak_sum_matrix.flatten()), axis=1)
		ks3d_metadata = (coords, color, data_label)
		ks3d_pack.append(ks3d_metadata)

		# plot slices
		if flags['plot_slices_flag']:
			f_filename = e_filename.replace('_E_', '_F_')
			f_prepared_data = auto_prepare_data(folder, f_filename, dstep_to=dstep_to)
			f_lat_per_slice = get_lat_matrix(f_prepared_data, dstep_to)

			# find an ideal example of dataset
			if "bio_" in e_filename:
				ideal_sample = example_bio_sample(folder, e_filename)
			else:
				ideal_sample = example_sample(e_lat_matrix, e_peak_sum_matrix, dstep_to)

			plot_slices(e_prepared_data, f_prepared_data, e_lat_matrix, f_lat_per_slice,
			            best_sample=ideal_sample, save_to=save_to, filename=e_filename, dstep=dstep_to)

		# if flags['plot_peaks_by_intervals']:
		# 	e_peaks_per_interval = get_peak_amp_matrix(e_prepared_data, step_size_to, split_by_intervals=True)
		# 	peaks_per_interval_pack.append(e_peaks_per_interval)

		log.info(f"Processed '{folder}'")

	if flags['plot_ks2d']:
		plot_ks2d(ks1d_peaks, ks1d_ampls, ks1d_names, ks1d_colors, dstep=dstep_to, save_to=save_to)

	if flags['plot_ks3d'] or flags['plot_pca3d']:
		plot_3D_PCA(ks3d_pack, ks1d_names, save_to=save_to, corr_flag=flags['plot_correlation'])

	if flags['plot_peaks_by_intervals']:
		plot_peaks_bar_intervals(peaks_per_interval_pack, ks1d_names, save_to=save_to)


def for_article():
	"""
	TODO: add docstring
	"""
	save_all_to = '/home/alex/GitHub/DATA/test'

	comb = [
		("foot", 21, 2, "no", 0.1),
		# ("foot", 13.5, 2, "no", 0.1),
		# ("foot", 6, 2, "no", 0.1),
		# ("toe", 21, 2, "no", 0.1),
		# ("toe", 13.5, 2, "no", 0.1),
		# ("air", 13.5, 2, "no", 0.1),
		# ("4pedal", 21, 4, "no", 0.25),
		# ("4pedal", 13.5, 4, "no", 0.25),
		# ("qpz", 13.5, 2, "", 0.1),
		# ("str", 21, 2, "no", 0.1),
		# ("str", 13.5, 2, "no", 0.1),
		# ("str", 6, 2, "no", 0.1),
	]

	for c in comb:
		compare_pack = [
			f'/home/alex/GitHub/DATA/bio/{c[0]}/bio_E_{c[1]}cms_40Hz_i100_{c[2]}pedal_{c[3]}5ht_T_{c[4]}step.hdf5',
			f'/home/alex/GitHub/DATA/neuron/{c[0]}/neuron_E_{c[1]}cms_40Hz_i100_{c[2]}pedal_{c[3]}5ht_T_0.025step.hdf5',
			f'/home/alex/GitHub/DATA/gras/{c[0]}/gras_E_{c[1]}cms_40Hz_i100_{c[2]}pedal_{c[3]}5ht_T_0.025step.hdf5',
			f'/home/alex/GitHub/DATA/nest/{c[0]}/nest_E_{c[1]}cms_40Hz_i100_{c[2]}pedal_{c[3]}5ht_T_0.025step.hdf5',
		]
		# control
		flags = dict(plot_ks3d=False,
		             plot_pca3d=False,
		             plot_contour_flag=False,
		             plot_correlation=False,
		             plot_slices_flag=True,
		             # plot_ks2d=True,
		             plot_peaks_by_intervals=False)
		__process_dataset(compare_pack, f"{save_all_to}/TEST", flags, dstep_to=c[4])


def run():
	for_article()


if __name__ == "__main__":
	run()
