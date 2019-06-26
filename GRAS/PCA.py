import numpy as np
import pylab as plt
import h5py as hdf5
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy.signal import argrelextrema
from analysis.functions import normalization
from analysis.histogram_lat_amp import sim_process

bio_step = 0.25
sim_step = 0.025

data_folder = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/bio_data/"

k_median = 0
k_box_Q3 = 1
k_box_Q1 = 2
k_whiskers_high = 3
k_whiskers_low = 4
k_fliers_high = 5
k_fliers_low = 6

min_color = "#00FFFF"
max_color = "#ED1B24"

percents = [25, 50, 75]


def read_data(filepath):
	with hdf5.File(filepath) as file:
		data_by_test = [test_values[:] for test_values in file.values()]
	return data_by_test


def select_slices(path, begin, end):
	return [data[begin:end] for data in read_data(path)]


def hex2rgb(hex_color):
	hex_color = hex_color.lstrip('#')
	return [int("".join(gr), 16) / 256 for gr in zip(*[iter(hex_color)] * 2)]


def length(v):
	return np.sqrt(v[0] ** 2 + v[1] ** 2)


def unit_vector(vector):
	return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def poly_area_by_coords(x, y):
	return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calc_boxplots(dots):
	"""
	ToDo fill the docstring
	Args:
		dots:
	Returns:
	"""
	low_box_Q1, median, high_box_Q3 = np.percentile(dots, percents)
	# calc borders
	IQR = high_box_Q3 - low_box_Q1
	Q1_15 = low_box_Q1 - 1.5 * IQR
	Q3_15 = high_box_Q3 + 1.5 * IQR

	high_whisker, low_whisker = high_box_Q3, low_box_Q1,

	for dot in dots:
		if high_box_Q3 < dot <= Q3_15 and dot > high_whisker:
			high_whisker = dot
		if Q1_15 <= dot < low_box_Q1 and dot < low_whisker:
			low_whisker = dot

	high_flier, low_flier = high_whisker, low_whisker
	for dot in dots:
		if dot > Q3_15 and dot > high_flier:
			high_flier = dot

		if dot < Q1_15 and dot < low_flier:
			low_flier = dot

	return median, high_box_Q3, low_box_Q1, high_whisker, low_whisker, high_flier, low_flier


def smooth(data, box_pts):
	"""
	Smooth the data by N box_pts number
	Args:
		data (np.ndarray): original data
		box_pts (int):
	Returns:
		np.ndarray: smoothed data
	"""
	box = np.ones(box_pts) / box_pts
	return np.convolve(data, box, mode='same')


def min_at(array):
	"""
	Wrapper of numpy.argmin for simplifying code
	Args:
		array (np.ndarray):
	Returns:
		np.ndarray: index of min value
		np.ndarray: min value
	"""
	index = np.argmin(array).astype(int)
	value = array[index]
	return index, value


def max_at(array):
	"""
	Wrapper of numpy.argmax for simplifying code
	Args:
		array (np.ndarray):
	Returns:
		np.ndarray: index of max value
		np.ndarray: max value
	"""
	index = np.argmax(array).astype(int)
	value = array[index]
	return index, value


def find_extremuma(array, condition):
	"""
	Wrapper of numpy.argrelextrema for siplifying code
	Args:
		array (np.ndarray):
		condition (np.ufunc):
	Returns:
		np.ndarray: indexes of extremuma
		np.ndarray: values of extremuma
	"""
	indexes = argrelextrema(array, condition)[0]
	values = array[indexes]

	diff_neighbor_extremuma = np.abs(np.diff(values, n=1))

	indexes = np.array([index for index, diff in zip(indexes, diff_neighbor_extremuma) if diff > 0] + [indexes[-1]])
	values = array[indexes]

	return indexes, values


def indexes_where(indexes, less_than=None, greater_than=None):
	"""
	Filter indexes which less or greater than value
	Args:
		indexes: array of indexes
		less_than (float): optional, if is not None uses '<' sign
		greater_than (float): optional, if is not None uses '>' sign
	Returns:
		np.ndarray: compressed array
	"""
	if less_than is not None:
		return np.compress(indexes < less_than, indexes).astype(int)
	if greater_than is not None:
		return np.compress(indexes > greater_than, indexes).astype(int)
	raise Exception("You didn't choose any condtinion!")


def slice_metainfo(data, ees_hz, debugging=True):
	"""
	ToDo fill the docstring
	Args:
		data:
		ees_hz:
		debugging:
	Returns:
	"""
	global_amp_values = []
	global_lat_indexes = []
	global_peaks_numbers = []

	# keys
	X = 0
	Y = 1
	k_index = 0
	k_value = 1

	# additional properties
	allowed_diff = 0.11
	min_index_interval = 1
	slice_in_ms = int(1000 / ees_hz)

	# set ees area, before that time we don't try to find latencies
	ees_zone_time = int(8 / bio_step)
	shared_x = np.arange(slice_in_ms / bio_step) * bio_step

	# read all bio data (list of tests)
	bio_data = np.array(data)
	# get number of slices based on length of the bio data
	slices_number = int(len(bio_data[0]) / (slice_in_ms / bio_step))

	# split original data only for visualization
	splitted_per_slice_original = np.split(bio_data.T, slices_number)
	# calc boxplot per step and split it by number of slices
	splitted_per_slice_boxplots = np.split(np.array([calc_boxplots(dot) for dot in bio_data.T]), slices_number)

	# set the poly area
	l_poly_border = ees_zone_time
	r_poly_border = int(slice_in_ms / bio_step)
	poly_area = slice(l_poly_border, r_poly_border)

	# just for visualization
	if debugging:
		for slice_index, slice_data in enumerate(splitted_per_slice_original):
			y_offset = slice_index * 10
			plt.plot(np.arange(len(slice_data)) * bio_step, slice_data * 1.5 + y_offset)
		plt.xticks(range(0, slice_in_ms + 1), range(0, slice_in_ms + 1))
		plt.grid(axis='x')
		plt.xlim(0, slice_in_ms)
		plt.show()

	# compute latency per slice
	for slice_index, slice_data in enumerate(splitted_per_slice_boxplots):
		latencies_Q1 = []
		latencies_Q3 = []

		'''[1] preparing data'''
		# get data by keys
		data_Q1 = slice_data[:, k_fliers_low]
		data_Q3 = slice_data[:, k_fliers_high]
		median = slice_data[:, k_median]
		# smooth the data to avoid micropeaks
		smoothed_Q1 = smooth(data_Q1, 2)
		smoothed_Q3 = smooth(data_Q3, 2)
		smoothed_median = smooth(median, 2)
		# fix the last broken data after smoothing
		smoothed_Q1[-2:] = data_Q1[-2:]
		smoothed_Q3[-2:] = data_Q3[-2:]
		smoothed_median[-2:] = median[-2:]
		# get a delta of NOT smoothed fliers data
		delta_data = np.abs(data_Q1 - data_Q3)
		# get a delta of smoothed fliers data
		delta_smoothed_data = np.abs(smoothed_Q1 - smoothed_Q3)

		'''[2] finding extremuma'''
		# get all Q1 extremuma indexes and values
		e_all_Q1_minima_indexes, e_all_Q1_minima_values = find_extremuma(smoothed_Q1, np.less_equal)
		e_all_Q1_maxima_indexes, e_all_Q1_maxima_values = find_extremuma(smoothed_Q1, np.greater_equal)
		# get all Q3 extremuma indexes and values
		e_all_Q3_minima_indexes, e_all_Q3_minima_values = find_extremuma(smoothed_Q3, np.less_equal)
		e_all_Q3_maxima_indexes, e_all_Q3_maxima_values = find_extremuma(smoothed_Q3, np.greater_equal)

		'''[3] finding EES'''
		# get the lowest (Q1) mono minima extremuma
		e_mono_Q1_minima_indexes = e_all_Q1_minima_indexes[e_all_Q1_minima_indexes < ees_zone_time].astype(int)
		# find an index of the biggest delta between median[0] and mono extremuma
		max_delta_Q1_index, _ = max_at(np.abs(smoothed_Q1[e_mono_Q1_minima_indexes] - median[0]))
		# get index of extremuma with the biggest delta
		ees_index = e_mono_Q1_minima_indexes[max_delta_Q1_index]

		'''[4] finding poly answer'''
		# get only poly Q1 extremuma indexes
		e_poly_Q1_minima_indexes = e_all_Q1_minima_indexes[e_all_Q1_minima_indexes > ees_zone_time]
		e_poly_Q1_maxima_indexes = e_all_Q1_maxima_indexes[e_all_Q1_maxima_indexes > ees_zone_time]
		# get only poly Q3 extremuma indexes
		e_poly_Q3_minima_indexes = e_all_Q3_minima_indexes[e_all_Q3_minima_indexes > ees_zone_time]
		e_poly_Q3_maxima_indexes = e_all_Q3_maxima_indexes[e_all_Q3_maxima_indexes > ees_zone_time]
		# merge poly extremuma indexes and sort them to create a pairs in future
		e_poly_Q1_indexes = np.sort(np.concatenate((e_poly_Q1_minima_indexes, e_poly_Q1_maxima_indexes))).astype(int)
		e_poly_Q3_indexes = np.sort(np.concatenate((e_poly_Q3_minima_indexes, e_poly_Q3_maxima_indexes))).astype(int)

		# find latencies in Q1 (zip into the pairs)
		for e_left, e_right in zip(e_poly_Q1_indexes, e_poly_Q1_indexes[1:]):
			# if dots are too close
			if e_right - e_left == 1:
				latency_index = e_right
			# else find indexes of minimal variance index in [dot left, dot right) interval
			else:
				latency_index = e_left + min_at(delta_smoothed_data[e_left:e_right])[k_index]
			latencies_Q1.append(latency_index)

		# find latencies in Q3
		for e_left, e_right in zip(e_poly_Q3_indexes, e_poly_Q3_indexes[1:]):
			# if dots are too close
			if e_right - e_left == 1:
				latency_index = e_right
			# else find indexes of minimal variance index in [dot left, dot right) interval
			else:
				latency_index = e_left + min_at(delta_smoothed_data[e_left:e_right])[k_index]
			latencies_Q3.append(latency_index)

		'''[5] finding best borders by delta'''
		# prepare lists
		merged_names = []
		merged_values = []
		merged_indexes = []

		# find extremuma for deltas
		poly_hist_indexes_min, poly_hist_values_min = find_extremuma(delta_data[poly_area], np.less_equal)
		poly_hist_indexes_max, poly_hist_values_max = find_extremuma(delta_data[poly_area], np.greater_equal)

		# TODO OPTIMIZE ME
		# concatenate lists of extremuma with some rules
		if poly_hist_indexes_min[0] < poly_hist_indexes_max[0]:
			if len(poly_hist_indexes_max) > len(poly_hist_indexes_min):
				length = len(poly_hist_indexes_max)
			else:
				length = len(poly_hist_indexes_min)

			for x in range(length):
				if x < len(poly_hist_indexes_min):
					merged_names.append("min")
					merged_indexes.append(poly_hist_indexes_min[x])
					merged_values.append(poly_hist_values_min[x])
				if x < len(poly_hist_indexes_max):
					merged_names.append("max")
					merged_indexes.append(poly_hist_indexes_max[x])
					merged_values.append(poly_hist_values_max[x])
			if len(poly_hist_indexes_max) > len(poly_hist_indexes_min):
				merged_names.append('max')
				merged_indexes.append(poly_hist_indexes_max[-1])
				merged_values.append(poly_hist_values_max[-1])
		else:
			if len(poly_hist_indexes_max) > len(poly_hist_indexes_min):
				length = len(poly_hist_indexes_max)
			else:
				length = len(poly_hist_indexes_min)

			for x in range(length):
				if x < len(poly_hist_indexes_max):
					merged_names.append("max")
					merged_indexes.append(poly_hist_indexes_max[x])
					merged_values.append(poly_hist_values_max[x])
				if x < len(poly_hist_indexes_min):
					merged_names.append("min")
					merged_indexes.append(poly_hist_indexes_min[x])
					merged_values.append(poly_hist_values_min[x])
			if len(poly_hist_indexes_max) > len(poly_hist_indexes_min):
				merged_names.append('max')
				merged_indexes.append(poly_hist_indexes_max[-1])
				merged_values.append(poly_hist_values_max[-1])

		merged_names = np.array(merged_names)
		merged_values = np.array(merged_values)
		merged_indexes = np.array(merged_indexes)

		# get difference of merged indexes with step 1
		differed_indexes = np.abs(np.diff(merged_indexes, n=1))
		# filter closest indexes and add the True to the end, because the last dot doesn't have diff with next point
		is_index_ok = np.append(differed_indexes > min_index_interval, True)

		index = 0
		while True:
			if index + 1 >= len(is_index_ok):
				break
			if not is_index_ok[index] and index + 1 < len(is_index_ok):
				is_index_ok[index + 1] = False
				index += 2
				continue
			index += 1

		assert len(is_index_ok) == len(merged_indexes)
		assert len(merged_names) == len(merged_indexes)

		# create filters for indexes by name and filtering array 'is_index_ok'
		min_mask = is_index_ok & (merged_names == "min")
		max_mask = is_index_ok & (merged_names == "max")
		# use filters
		poly_hist_indexes_min = merged_indexes[min_mask]
		poly_hist_indexes_max = merged_indexes[max_mask]

		# find the best right border
		i_min = 0
		r_best_border = 0
		l_best_border = poly_hist_indexes_min[0] + l_poly_border
		is_border_found = False

		while not is_border_found and i_min < len(merged_names):
			if merged_names[i_min] == 'min' and is_index_ok[i_min]:
				for i_max in range(i_min + 1, len(merged_names)):
					if merged_names[i_max] == 'max' and is_index_ok[i_max] and abs(merged_values[i_max] - merged_values[i_min]) > allowed_diff:
						r_best_border = merged_indexes[i_max] + l_poly_border
						is_border_found = True
						break
			i_min += 1

		if not is_border_found:
			raise Exception("WHERE IS MAXIMAL BORDER???")

		'''[6] finding best latency in borders'''
		best_latency = (l_best_border, delta_smoothed_data[l_best_border])
		# find the latency in Q1 and Q3 data (based on the maximal variance)
		for lat_Q1 in filter(lambda dot: l_best_border < dot < r_best_border, latencies_Q1):
			if delta_smoothed_data[lat_Q1] > best_latency[k_value]:
				best_latency = (lat_Q1, delta_smoothed_data[lat_Q1])
		for lat_Q3 in filter(lambda dot: l_best_border < dot < r_best_border, latencies_Q3):
			if delta_smoothed_data[lat_Q3] > best_latency[k_value]:
				best_latency = (lat_Q3, delta_smoothed_data[lat_Q3])

		# append found latency to the global list of the all latencies per slice
		global_lat_indexes.append(best_latency[k_index])

		# all debugging info
		if debugging:
			print("\n")
			print("- " * 20)
			print("{:^40}".format(f"SLICE {slice_index + 1}"))
			print("- " * 20)
			print(f"  EES was found at: {ees_index * bio_step}ms (index {ees_index})")
			print(f" minima_indexes Q1: {e_poly_Q1_minima_indexes}")
			print(f" maxima_indexes Q1: {e_poly_Q1_maxima_indexes}")
			print(f" merged indexes Q1: {e_poly_Q1_indexes}")
			print(f"found latencies Q1: {latencies_Q1}")
			print("- " * 20)
			print(f" minima_indexes Q3: {e_poly_Q3_minima_indexes}")
			print(f" maxima_indexes Q3: {e_poly_Q3_maxima_indexes}")
			print(f" merged indexes Q3: {e_poly_Q3_indexes}")
			print(f"found latencies Q3: {latencies_Q3}")
			print("- " * 20)
			print(f" poly answers area: [{l_poly_border * bio_step}, {r_poly_border * bio_step}]ms")
			print(f"  hist indexes min: {poly_hist_indexes_min}")
			print(f"  hist indexes max: {poly_hist_indexes_max}")
			print(f"      merged names: {merged_names}")
			print(f"    merged indexes: {merged_indexes}")
			print(f"     merged values: {merged_values}")
			print(f"  differed_indexes: {differed_indexes}")
			print(f" indexes that okay: {is_index_ok}")
			print(f"best area between {l_best_border * bio_step}ms and {r_best_border * bio_step}ms")
			print(f"latency at {best_latency[0] * bio_step}ms")

			# plot an area of EES
			plt.axvspan(xmin=0, xmax=ees_zone_time, color='g', alpha=0.3, label="EES area")
			# plot EES
			plt.axvline(x=ees_index, color='orange', linewidth=3, label="EES")
			# plot an area where we try to find a best latency
			plt.axvspan(xmin=l_best_border, xmax=r_best_border, color='#175B99', alpha=0.3, label="best latency area")
			# plot bars (delta of variance) and colorize extremuma
			colors = ['#2B2B2B'] * len(delta_data)
			for i in poly_hist_indexes_min + l_poly_border:
				colors[i] = 'b'
			for i in poly_hist_indexes_max + l_poly_border:
				colors[i] = 'r'
			plt.bar(range(len(delta_data)), delta_data, bottom=-10, width=0.4, color=colors)
			# plot not filtered extremuma
			plt.plot(merged_indexes + l_poly_border, merged_values - 9.7, '.', markersize=7, color='k')
			# plot latencies for Q1 and Q3
			plt.plot(latencies_Q1, smoothed_Q1[latencies_Q1], '.', markersize=20, color='#227734', label="Q1 latencies")
			plt.plot(latencies_Q3, smoothed_Q3[latencies_Q3], '.', markersize=20, color="#FF6600", label="Q3 latencies")
			# plot the best latency with guidline
			best_lat_x = best_latency[0]
			best_lat_y = smoothed_Q1[best_lat_x]
			plt.plot([best_lat_x] * 2, [best_lat_y - 2, best_lat_y], color="k", linewidth=0.5)
			plt.text(best_lat_x, best_lat_y - 2, best_lat_x * bio_step)
			# plot original bio data per slice
			plt.plot(splitted_per_slice_original[slice_index], linewidth=0.7)
			# plot Q1 and Q3 areas, and median
			plt.plot(smoothed_Q1, color='k', linewidth=3.5, label="Q1/Q3 values")
			plt.plot(smoothed_Q3, color='k', linewidth=3.5)
			plt.plot(median, linestyle='--', color='k', label="median value")
			# plot extremuma
			plt.plot(e_all_Q1_minima_indexes, e_all_Q1_minima_values, '.', color=min_color, label="minima extremuma")
			plt.plot(e_all_Q1_maxima_indexes, e_all_Q1_maxima_values, '.', color=max_color, label="maxima extremuma")
			plt.plot(e_all_Q3_minima_indexes, e_all_Q3_minima_values, '.', color=min_color)
			plt.plot(e_all_Q3_maxima_indexes, e_all_Q3_maxima_values, '.', color=max_color)
			# figure properties
			plt.suptitle(f"Slice #{slice_index + 1}")
			plt.xticks(range(0, 101, 4), (np.arange(0, 101, 4) * bio_step).astype(int))
			plt.xlim(0, 100)
			plt.tight_layout()
			plt.legend()
			plt.show()

	# show latencies on the all slices (shadows)
	if debugging:
		plt.subplots(figsize=(16, 9))
		yticks = []
		y_offset = 3

		for slice_index, data in enumerate(splitted_per_slice_boxplots):
			data += slice_index * y_offset  # is a link (!)
			plt.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color='r', alpha=0.3, label="flier")
			plt.fill_between(shared_x, data[:, k_whiskers_high], data[:, k_whiskers_low], color='r', alpha=0.5, label="whisker")
			plt.fill_between(shared_x, data[:, k_box_Q3], data[:, k_box_Q1], color='r', alpha=0.7, label="box")
			plt.plot(shared_x, data[:, k_median], linestyle='--', color='k')
			yticks.append(data[:, k_median][0])
		plt.xticks(range(26), range(26))
		plt.grid(axis='x')

		lat_x = [x * bio_step for x in global_lat_indexes]
		lat_y = [splitted_per_slice_boxplots[slice_index][:, k_median][lat] for slice_index, lat in enumerate(global_lat_indexes)]
		plt.plot(lat_x, lat_y, linewidth=3, color='g')

		# plt.xticks(range(100), [x * bio_step for x in range(100) if x % 4 == 0])
		plt.yticks(yticks, range(1, slices_number + 1))
		plt.xlim(0, slice_in_ms)

		plt.show()

	return global_lat_indexes, global_amp_values, global_peaks_numbers

	raise Exception


def plot_pca():
	splitted_per_slice_boxplots = np.split(np.array([calc_boxplots(dot) for dot in bio_data.T]), slices_number)

	bio_meta = slice_metainfo(read_data(f"{data_folder}/21cms_40Hz_100%_slices5-10.hdf5"), ees_hz=40)

	neuron_means = np.sum(np.array(
		[np.absolute(normalization(data, -1, 1)) for data in select_slices(f"{data_folder}/neuron_15.hdf5", 0, 12000)]),
	                      axis=0)

	gras_means = np.sum(np.array([np.absolute(normalization(data, -1, 1)) for data in
	                              select_slices(f"{data_folder}/gras_15.hdf5", 10000, 22000)]), axis=0)

	# calculating latencies and amplitudes of mean values
	bio_means_lat = sim_process(bio_meta, bio_step, inhibition_zero=True, debugging=True)[0]
	bio_means_amp = sim_process(bio_meta, bio_step, inhibition_zero=True, after_latencies=after_latencies)[1]

	neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=True, debugging=True)[0]
	neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=True, after_latencies=after_latencies)[1]

	gras_means_lat = sim_process(gras_means, sim_step, inhibition_zero=True, debugging=True)[0]
	gras_means_amp = sim_process(gras_means, sim_step, inhibition_zero=True, after_latencies=after_latencies)[1]

	bio_pack = [np.array(list(zip(bio_means_amp, bio_means_lat))), '#a6261d', 'bio']
	neuron_pack = [np.array(list(zip(neuron_means_amp, neuron_means_lat))), '#f2aa2e', 'neuron']
	gras_pack = [np.array(list(zip(gras_means_amp, gras_means_lat))), '#287a72', 'gras']

	# start plotting
	fig, ax = plt.subplots()

	bio_S = 0

	# plot per data pack
	for coords, color, label in [bio_pack, neuron_pack, gras_pack]:
		pca = PCA(n_components=2)  # create PCA instance
		pca.fit(coords)  # fit the model with coords
		center = np.array(pca.mean_)  # get the center (mean value)

		# calc vectors
		vectors = []
		for v_length, vector in zip(pca.explained_variance_, pca.components_):
			y = vector * 3 * np.sqrt(v_length)
			vectors.append((center, center + y))

		# calc an angle between vector[first vector][top coords of vector] and vertical vector from the center
		first_vector = np.array(vectors[0][1])
		vertical = np.array([center[X], center[Y] + 10])
		angle_degrees = angle_between(vertical - center, first_vector - center)

		# check on angle sign (vector[first vector][top coord of vector][x coord]) > point1[x coord]
		sign = -1 if vectors[0][1][0] > vertical[0] else 1

		# calculate ellipse size
		ellipse_width = length(vectors[1][1] - center) * 2
		ellipse_height = length(vectors[0][1] - center) * 2

		# plot vectors
		for vector in vectors:
			ax.annotate('', vector[1], vector[0], arrowprops=dict(facecolor=color, linewidth=1.0))

		# plot dots
		ax.scatter(coords[:, X], coords[:, Y], color=color, label=label, s=80)
		if debugging:
			for index, x, y in zip(range(len(coords[:, X])), coords[:, X], coords[:, Y]):
				ax.text(x, y, index + 1)

		# plot ellipse
		ellipse = Ellipse(xy=tuple(center), width=ellipse_width, height=ellipse_height, angle=angle_degrees * sign)
		ellipse.set_fill(False)
		ellipse.set_edgecolor(hex2rgb(color))
		ax.add_artist(ellipse)

		# fill convex
		hull = ConvexHull(coords)
		S = poly_area_by_coords(coords[hull.vertices, X], coords[hull.vertices, Y])
		if label == "bio":
			bio_S = S
		print(label, S / bio_S)
		ax.fill(coords[hull.vertices, X], coords[hull.vertices, Y], color=color, alpha=0.3)

	# plot atributes
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.xlabel('Amplitudes, mV', fontsize=28)
	plt.ylabel('Latencies, ms', fontsize=28)
	plt.legend()
	plt.show()


def run():
	plot_pca()

if __name__ == "__main__":
	run()

