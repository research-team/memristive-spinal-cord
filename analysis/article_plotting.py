import logging
import numpy as np
import pylab as plt
from sklearn.decomposition import PCA
from analysis.PCA import calc_boxplots
from analysis.PCA import split_by_slices, read_data, select_slices, prepare_data, get_lat_amp, get_peaks
from analysis.PCA import Arrow3D, form_ellipse, plot_ellipsoid
from analysis.pearson_correlation import calc_correlation

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()

bar_width = 0.9

# keys
k_index = 0
k_value = 1
k_median = 0
k_box_Q3 = 1
k_box_Q1 = 2
k_whiskers_high = 3
k_whiskers_low = 4
k_fliers_high = 5
k_fliers_low = 6


def plot_slices(extensor_data, flexor_data, latencies, ees_hz, data_step, folder, filename):
	"""
	TODO: add docstring
	Args:
		extensor_data (list): values of extensor motoneurons membrane potential
		flexor_data (list): values of flexor motoneurons membrane potential
		latencies (list): latencies of poly answers per slice
		ees_hz (int): EES stimulation frequency
		data_step (float): data step
		folder (str): save folder path
		filename (str): name of the future path
	"""
	flexor_data = np.array(flexor_data)
	extensor_data = np.array(extensor_data)
	latencies = (np.array(latencies) / data_step).astype(int)

	# additional properties
	slice_in_ms = 1000 / ees_hz
	slice_in_steps = int(slice_in_ms / data_step)

	# calc boxplot per dot
	e_boxplots_per_iter = np.array([calc_boxplots(dot) for dot in extensor_data.T])
	f_boxplots_per_iter = np.array([calc_boxplots(dot) for dot in flexor_data.T])

	e_splitted_per_slice_boxplots = split_by_slices(e_boxplots_per_iter, slice_in_steps)
	f_splitted_per_slice_boxplots = split_by_slices(f_boxplots_per_iter, slice_in_steps)

	all_splitted_per_slice_boxplots = np.vstack((e_splitted_per_slice_boxplots,
	                                             f_splitted_per_slice_boxplots))

	yticks = []
	slices_number = int((len(extensor_data[0]) + len(flexor_data[0])) / (slice_in_ms / data_step))
	colors = iter(['#287a72', '#f2aa2e', '#472650'] * slices_number)

	plt.subplots(figsize=(16, 9))

	for slice_index, data in enumerate(all_splitted_per_slice_boxplots):
		data += slice_index
		shared_x = np.arange(len(data[:, k_fliers_high])) * data_step
		plt.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color=next(colors), alpha=0.7, zorder=3)
		plt.plot(shared_x, data[:, k_median], color='k', zorder=3)
		yticks.append(data[:, k_median][0])

	lat_x = latencies * data_step
	lat_y = [all_splitted_per_slice_boxplots[slice_index][:, k_median][lat] for slice_index, lat in enumerate(latencies)]
	print(lat_y)
	plt.plot(lat_x, lat_y, linewidth=4, linestyle='--', color='k', zorder=3)
	plt.plot(lat_x, lat_y, '.', markersize=25, color='k', zorder=3)

	xticks = range(int(slice_in_ms) + 1)
	xticklabels = [x if i % 5 == 0 else None for i, x in enumerate(xticks)]
	yticklabels = [None] * slices_number
	slice_indexes = range(1, slices_number + 1)
	for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
		yticklabels[i] = slice_indexes[i]

	plt.xticks(xticks, xticklabels, fontsize=56)
	plt.yticks(yticks, yticklabels, fontsize=56)
	plt.xlim(0, slice_in_ms)
	plt.grid(axis='x', alpha=0.5)

	plt.tight_layout()
	plt.savefig(f"{folder}/{filename}.pdf", dpi=250, format="pdf")
	plt.close()


def plot_3D_PCA(all_pack, folder):
	"""
	TODO: add docstring
	Args:
		all_pack (list of list): special structure to easily work with (coords, color and label)
		folder (str): save folder path
	"""
	for elev, azim, title in (0, -90.1, "Lat Peak"), (0.1, 0.1, "Amp Peak"), (89.9, -90.1, "Lat Amp"):
		# init 3D projection figure
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')
		# plot each data pack
		# coords is a matrix of coordinates, stacked as [[x1, y1, z1], [x2, y2, z2] ...]
		for coords, color, label in all_pack:
			# create PCA instance and fit the model with coords
			pca = PCA(n_components=3)
			pca.fit(coords)
			# get the center (mean value of points cloud)
			center = pca.mean_
			# get PCA vectors' head points (semi axis)
			vectors_points = [3 * np.sqrt(val) * vec for val, vec in zip(pca.explained_variance_, pca.components_)]
			vectors_points = np.array(vectors_points)
			# form full axis points (original vectors + mirrored vectors)
			axis_points = np.concatenate((vectors_points, -vectors_points), axis=0)
			# centering vectors and axis points
			vectors_points += center
			axis_points += center
			# calculate radii and rotation matrix based on axis points
			radii, rotation = form_ellipse(axis_points)
			# plot PCA vectors
			for point_head in vectors_points:
				arrow = Arrow3D(*zip(center.T, point_head.T), mutation_scale=20, lw=3, arrowstyle="-|>", color=color)
				ax.add_artist(arrow)
			# plot cloud of points
			ax.scatter(*coords.T, alpha=0.5, s=30, color=color, label=label)
			# plot ellipsoid
			plot_ellipsoid(center, radii, rotation, plot_axes=False, color=color, alpha=0.1)
		# figure properties
		ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=35)
		ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=35)
		ax.set_zticklabels(ax.get_zticks().astype(int), fontsize=35)

		ax.xaxis._axinfo['tick']['inward_factor'] = 0
		ax.yaxis._axinfo['tick']['inward_factor'] = 0
		ax.zaxis._axinfo['tick']['inward_factor'] = 0

		if "Lat" not in title:
			ax.set_xticks([])
		if "Amp" not in title:
			ax.set_yticks([])
		if "Peak" not in title:
			ax.set_zticks([])

		plt.legend()
		ax.view_init(elev=elev, azim=azim)
		plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
		title = str(title).lower().replace(" ", "_")
		plt.savefig(f"{folder}/{title}.pdf", dpi=250, format="pdf")

		plt.close(fig)


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


def plot_histograms(amp_per_slice, peaks_per_slice, lat_per_slice, all_data, mono_per_slice, folder, filename, ees_hz):
	"""
	TODO: add docstring
	Args:
		amp_per_slice (list): amplitudes per slice
		peaks_per_slice (list): number of peaks per slice
		lat_per_slice (list): latencies per slice
		all_data (list of list): data per test run
		mono_per_slice (list): end of mono area per slice
		folder (str): folder path
		filename 9str): filename of the future file
	"""
	step = 0.25
	box_distance = 1.2
	color = "#472650"
	fill_color = "#9D8DA3"
	slices_number = len(amp_per_slice)
	slice_length = int(1000 / ees_hz / step)
	slice_indexes = np.array(range(slices_number))

	# calc boxplots per iter
	boxplots_per_iter = np.array([calc_boxplots(dot) for dot in np.array(all_data).T])

	# plot histograms
	for data, title in (amp_per_slice, "amplitudes"), (peaks_per_slice, "peaks"):
		# create subplots
		fig, ax = plt.subplots(figsize=(16, 9))
		# property of bar fliers
		xticks = [x * box_distance for x in slice_indexes]
		# plot amplitudes or peaks
		plt.bar(xticks, data, width=bar_width, color=color, zorder=2)
		# set labels
		xticklabels = [None] * len(slice_indexes)
		human_read = [i + 1 for i in slice_indexes]
		for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
			xticklabels[i] = human_read[i]
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
		plt.savefig(f"{folder}/{filename}_{title}.pdf", dpi=250, format="pdf")
		plt.close()

		log.info(f"Plotted {title} for {filename}")

	# form areas
	splitted_per_slice_boxplots = split_by_slices(boxplots_per_iter, slice_length)
	mono_area = [slice_data[:int(time / step)] for time, slice_data in zip(mono_per_slice, splitted_per_slice_boxplots)]
	poly_area = [slice_data[int(time / step):] for time, slice_data in zip(lat_per_slice, splitted_per_slice_boxplots)]

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
		xticks = [x * box_distance for x in slice_indexes]
		plt.xticks(fontsize=56)
		plt.yticks(fontsize=56)

		lat_plot = ax.boxplot(area_data, positions=xticks, widths=bar_width, patch_artist=True, flierprops=fliers)
		recolor(lat_plot, color, fill_color)

		xticks = [i * box_distance for i in slice_indexes]
		xticklabels = [None] * len(slice_indexes)
		human_read = [i + 1 for i in slice_indexes]
		for i in [0, -1, int(1 / 3 * slices_number), int(2 / 3 * slices_number)]:
			xticklabels[i] = human_read[i]

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
		plt.savefig(f"{folder}/{filename}_{title}.pdf", dpi=250, format="pdf")
		plt.close()

		log.info(f"Plotted {title} for {filename}")


def extract_extensor_flexor(folder, filename, original_data_step, data_step_to):
	e_slices_number = {"6": 30, "15": 12, "21": 6}
	slice_in_steps = int(25 / original_data_step)
	print(filename)

	ees_hz = int(filename[:filename.find("Hz")].split("_")[-1])
	speed = filename[:filename.find("cms")].split("_")[-1]
	log.info(f"prepare {filename}")
	path_extensor = f"{folder}/{filename}.hdf5"
	path_flexor = f"{folder}/{filename.replace('_E_', '_F_')}.hdf5"
	# check if it is a bio data -- use another function
	if "bio_" in filename:
		e_dataset = read_data(path_extensor)
		f_dataset = read_data(path_flexor)
	# simulation data computes by the common function
	else:
		# calculate extensor borders
		extensor_begin = 0
		extensor_end = e_slices_number[speed] * slice_in_steps
		# calculate flexor borders
		flexor_begin = extensor_end
		flexor_end = extensor_end + (7 if "4pedal" in filename else 5) * slice_in_steps
		# use native funcion for get needful data
		e_dataset = select_slices(path_extensor, extensor_begin, extensor_end, original_data_step, data_step_to)
		f_dataset = select_slices(path_flexor, flexor_begin, flexor_end, original_data_step, data_step_to)

	# prepare each data (stepping, centering, normalization)
	e_data = prepare_data(e_dataset)
	f_data = prepare_data(f_dataset)

	return e_data, f_data, ees_hz


def for_article():
	"""
	TODO: add docstring
	"""
	# stuff variables
	all_pack = []
	colors = iter(["#275b78", "#287a72", "#f2aa2e", "#472650", "#a6261d", "#f27c2e", "#2ba7b9"] * 10)

	# list of filenames for easily reading data
	bio_folder = "/home/alex/bio_data_hdf/toe"
	bio_filenames = [
		"bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T",
		"bio_E_21cms_40Hz_i100_2pedal_no5ht_T",
	]

	neuron_folder = "/home/alex/GitHub/memristive-spinal-cord/data/neuron"
	neuron_filenames = [
	    # "neuron_E_15cms_40Hz_i100_2pedal_5ht_T",
	    "neuron_E_15cms_40Hz_i100_2pedal_no5ht_T",
	    # "neuron_E_15cms_40Hz_i100_4pedal_no5ht_T",
	    # "neuron_E_21cms_40Hz_i100_2pedal_no5ht_T",
	    # "neuron_E_21cms_40Hz_i100_4pedal_no5ht_T"
	]

	gras_folder = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/dat/reduce 40Hz/5"
	gras_filenames = [
		"gras_E_21cms_40Hz_i100_2pedal_no5ht_T",
	]

	# control
	folder = gras_folder
	filenames_pack = gras_filenames
	plot_pca_flag = False
	plot_slices_flag = True
	plot_histogram_flag = True

	data_step_to = 0.25
	original_data_step = 0.025

	# prepare data per file
	for filename in filenames_pack:
		# get extensor and flexor from the data (they are prepared for processing)
		e_data, f_data, ees_hz = extract_extensor_flexor(folder, filename,
		                                                 original_data_step=original_data_step,
		                                                 data_step_to=data_step_to)
		# get latencies, amplitudes and begining of poly answers
		lat_per_slice, amp_per_slice, mono_per_slice = get_lat_amp(e_data, ees_hz=ees_hz, data_step=data_step_to)
		# lat_per_slice += get_lat_amp(f_data, ees_hz=ees_hz, data_step=data_step_to)[0]
		# get number of peaks per slice
		peaks_per_slice = get_peaks(e_data, ees_hz=ees_hz, step=data_step_to)
		# form data pack
		print(lat_per_slice)
		print(amp_per_slice)
		print(peaks_per_slice)
		all_pack.append([np.stack((lat_per_slice, amp_per_slice, peaks_per_slice), axis=1), next(colors), filename])
		# plot histograms of amplitudes and number of peaks
		if plot_histogram_flag:
			plot_histograms(amp_per_slice, peaks_per_slice, lat_per_slice, e_data, mono_per_slice,
			                folder=folder, filename=filename, ees_hz=ees_hz)
		# plot all slices with pattern
		if plot_slices_flag:
			plot_slices(e_data, f_data, lat_per_slice, ees_hz=ees_hz, data_step=data_step_to, folder=folder, filename=filename)
	# plot 3D PCA for each plane
	if plot_pca_flag:
		plot_3D_PCA(all_pack, folder=folder)


def plot_correlation():
	# FixMe: don't forget to change!
	save_to = "/home/alex/testfolder"

	data_a_folder = "/home/alex/GitHub/memristive-spinal-cord/data/bio"
	data_a_filename = "bio_sci_E_15cms_40Hz_i100_2pedal_5ht_T_2016-05-12"

	data_b_folder = "/home/alex/GitHub/memristive-spinal-cord/data/gras"
	data_b_filename = "gras_E_15cms_40Hz_i100_2pedal_5ht_T"

	# get extensor from data
	extensor_data_a = extract_extensor_flexor(data_a_folder, data_a_filename)[0]
	extensor_data_b = extract_extensor_flexor(data_b_folder, data_b_filename)[0]

	mono_corr, poly_corr = calc_correlation(extensor_data_a, extensor_data_b)

	plt.figure(figsize=(16, 9))

	title = f"{data_a_filename.split('_')[0]}_{data_b_filename}"
	box_colors = iter(["#275b78", "#287a72"])
	# plot boxplots
	for i, data in enumerate([poly_corr, mono_corr]):
		color = next(box_colors)
		box = plt.boxplot(data, positions=[i], vert=False, whis=[5, 95], widths=0.7, patch_artist=True)
		# change colors for all elements
		for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
			for patch in box[element]:
				if element == "fliers":
					patch.set_markersize(10)
					patch.set_markerfacecolor(color)
					patch.set_markeredgecolor(None)
				patch.set_color(color)
				patch.set_linewidth(3)
		for patch in box['boxes']:
			patch.set(facecolor=f"{color}55")

	yticks = [0, 1]
	ylabels = ["poly", "mono"]
	plt.yticks(yticks, ylabels)

	plt.xticks(fontsize=56)
	plt.yticks(fontsize=56)
	plt.tight_layout()
	plt.savefig(f"{save_to}/{title}.pdf", format="pdf")


def run():
	for_article()
	# plot_correlation()


if __name__ == "__main__":
	run()
