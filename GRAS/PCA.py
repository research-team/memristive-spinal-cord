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


def poly_area(x, y):
	return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calc_boxplots(dots):
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
		data (numpy.ndarray): original data
		box_pts (int):
	Returns:
		numpy.ndarray: smoothed data
	"""
	box = np.ones(box_pts) / box_pts
	return np.convolve(data, box, mode='same')


def min_at(array):
	"""
	Wrapper of numpy.argmin for simplifying code
	Args:
		array (numpy.ndarray):
	Returns:
		numpy.ndarray: index of min value
		numpy.ndarray: min value
	"""
	index = np.argmin(array).astype(int)
	value = array[index]
	return index, value


def max_at(array):
	"""
	Wrapper of numpy.argmax for simplifying code
	Args:
		array (numpy.ndarray):
	Returns:
		numpy.ndarray: index of max value
		numpy.ndarray: max value
	"""
	index = np.argmax(array).astype(int)
	value = array[index]
	return index, value


def find_extremuma(array, condition):
	"""
	Wrapper of numpy.argrelextrema for siplifying code
	Args:
		array (numpy.ndarray):
		condition (numpy.ufunc):
	Returns:
		numpy.ndarray: indexes of extremuma
		numpy.ndarray: values of extremuma
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
		numpy.ndarray: compressed array
	"""
	if less_than is not None:
		return np.compress(indexes < less_than, indexes).astype(int)
	if greater_than is not None:
		return np.compress(indexes > greater_than, indexes).astype(int)
	raise Exception("You didn't choose any condtinion!")


def run(with_mono=False, debugging=True):
	after_latencies = not with_mono
	# keys
	X = 0
	Y = 1
	# read all bio data (list of tests)
	bio_data = np.array(read_data(f"{data_folder}/21cms_40Hz_100%_slices0-600.hdf5"))
	slices_number = int(len(bio_data[0]) / (25 / bio_step))

	splitted_per_slice_boxplots = np.split(np.array([calc_boxplots(dot) for dot in bio_data.T]), slices_number)

	splitted_per_slice_original = np.split(bio_data.T, slices_number)

	shared_x = np.arange(25 / bio_step) * bio_step

	ees_zone_time = int(10 / bio_step)

	k_Q1 = 'low_Q1'
	k_Q3 = 'high_Q3'
	k_minima = 'e_minima'
	k_maxima = 'e_maxima'
	k_indexes = 0
	k_values = 1

	global_lat_indexes = []

	# compute per slice data
	for slice_index, slice_data in enumerate(splitted_per_slice_boxplots):
		print("- " * 20)
		print("{:^40}".format(f"Slice {slice_index + 1}"))
		print("- " * 20)

		latencies_Q1 = []
		latencies_Q3 = []

		extremuma = {k_Q1: {}, k_Q3: {}}

		original_Q1 = slice_data[:, k_box_Q1]
		original_Q3 = slice_data[:, k_box_Q3]

		smoothed_Q1 = smooth(slice_data[:, k_box_Q1], 2)
		smoothed_Q3 = smooth(slice_data[:, k_box_Q3], 2)
		median = smooth(slice_data[:, k_median], 2)

		# fix the last broken data after smoothing
		smoothed_Q1[-2:] = original_Q1[-2:]
		smoothed_Q3[-2:] = original_Q3[-2:]

		# for Q1 maxima
		indexes, values = find_extremuma(smoothed_Q1, np.greater_equal)
		extremuma[k_Q1][k_maxima] = np.stack((indexes, values))
		# for Q1 minima
		indexes, values = find_extremuma(smoothed_Q1, np.less_equal)
		extremuma[k_Q1][k_minima] = np.stack((indexes, values))

		# for Q3 maxima
		indexes, values = find_extremuma(smoothed_Q3, np.greater_equal)
		extremuma[k_Q3][k_maxima] = np.stack((indexes, values))
		# for Q3 minima
		indexes, values = find_extremuma(smoothed_Q3, np.less_equal)
		extremuma[k_Q3][k_minima] = np.stack((indexes, values))

		diff_per_iter = np.abs(smoothed_Q1 - smoothed_Q3)

		# get Q1 extremuma of mono and poly answers
		e_all_Q1_maxima_indexes = extremuma[k_Q1][k_maxima][k_indexes, :]
		e_poly_Q1_maxima_indexes = indexes_where(e_all_Q1_maxima_indexes, greater_than=ees_zone_time)

		e_all_Q1_minima_indexes = extremuma[k_Q1][k_minima][k_indexes, :]
		e_poly_Q1_minima_indexes = indexes_where(e_all_Q1_minima_indexes, greater_than=ees_zone_time)
		e_mono_Q1_minima_indexes = indexes_where(e_all_Q1_minima_indexes, less_than=ees_zone_time)

		e_poly_Q1_indexes = np.sort(np.concatenate((e_poly_Q1_minima_indexes, e_poly_Q1_maxima_indexes))).astype(int)

		# get Q3 extremuma of mono and poly answers
		e_all_Q3_maxima_indexes = extremuma[k_Q3][k_maxima][k_indexes, :]
		e_poly_Q3_maxima_indexes = indexes_where(e_all_Q3_maxima_indexes, greater_than=ees_zone_time)

		e_all_Q3_minima_indexes = extremuma[k_Q3][k_minima][k_indexes, :]
		e_poly_Q3_minima_indexes = indexes_where(e_all_Q3_minima_indexes, greater_than=ees_zone_time)
		e_mono_Q3_maxima_indexes = indexes_where(e_all_Q3_maxima_indexes, less_than=ees_zone_time)

		e_poly_Q3_indexes = np.sort(np.concatenate((e_poly_Q3_minima_indexes, e_poly_Q3_maxima_indexes))).astype(int)

		# find EES
		max_diff_Q1_index, max_diff_Q1_value = max_at(np.abs(smoothed_Q1[e_mono_Q1_minima_indexes] - median[0]))
		max_diff_Q3_index, max_diff_Q3_value = max_at(np.abs(smoothed_Q3[e_mono_Q3_maxima_indexes] - median[0]))

		if max_diff_Q3_value > max_diff_Q1_value:
			ees_index = e_mono_Q3_maxima_indexes[max_diff_Q3_index]
		else:
			ees_index = e_mono_Q1_minima_indexes[max_diff_Q1_index]

		# find latencies in Q1
		for dot_left, dot_right in zip(e_poly_Q1_indexes, e_poly_Q1_indexes[1:]):
			dot_left += 1
			# if dots are too close
			if dot_right - dot_left == 0:
				global_ind = dot_right
			# else find indexes of minimal variance in (dot left, dot right] interval
			else:
				local_ind, _ = min_at(diff_per_iter[dot_left:dot_right])
				global_ind = local_ind + dot_left
			latencies_Q1.append(global_ind)

		# find latencies in Q3
		for dot_left, dot_right in zip(e_poly_Q3_indexes, e_poly_Q3_indexes[1:]):
			dot_left += 1
			# if dots are too close
			if dot_right - dot_left == 0:
				global_ind = dot_right
			# else find indexes of minimal variance in (dot left, dot right] interval
			else:
				local_ind, _ = min_at(diff_per_iter[dot_left:dot_right])
				global_ind = local_ind + dot_left
			latencies_Q3.append(global_ind)


		# ToDo find best (minimal size)
		common = sorted(list(set(latencies_Q1) & set(latencies_Q3)))

		print(f"EES at {ees_index * bio_step}ms (index {ees_index})")

		print(f"maxima_indexes Q1: {e_poly_Q1_maxima_indexes}")
		print(f"minima_indexes Q1: {e_poly_Q1_indexes}")
		print(f"merged Q1: {e_poly_Q1_indexes}")
		print(f"latencies Q1: {latencies_Q1}")

		print("- " * 20)

		print(f"maxima_indexes Q3: {e_poly_Q3_maxima_indexes}")
		print(f"minima_indexes Q3: {e_poly_Q3_minima_indexes}")
		print(f"merged Q3: {e_poly_Q3_indexes}")
		print(f"latencies Q3: {latencies_Q3}")

		print("- " * 20)

		print(f"latencies of slice #{slice_index + 1}")
		print(f"common Q1/Q3 = {common}")

		print("- " * 20)

		smallest_diff_index = np.argmin(diff_per_iter[common])
		smallest_diff_index += common[int(smallest_diff_index)]

		print(f"differed_per_iter = {diff_per_iter[common]}")
		print(f"smallest_diff_index {smallest_diff_index}")

		smallest_diff_index = common[0]
		global_lat_indexes.append(smallest_diff_index)

		# plot EES
		plt.axvline(x=ees_index, color='orange', linewidth=3)

		# plot original bio data per slice
		plt.plot(splitted_per_slice_original[slice_index], linewidth=0.7)

		# plot latencies
		plt.plot(latencies_Q3, smoothed_Q3[latencies_Q3], '.', markersize=20, color="#FF6600", alpha=0.9, label="Q3 latencies")
		plt.plot(latencies_Q1, smoothed_Q1[latencies_Q1], '.', markersize=20, color='#35A53F', alpha=0.9, label="Q1 latencies")

		# plot common dots with guidlines
		for dot_x in common:
			plt.plot([dot_x] * 2, [-2, smoothed_Q1[dot_x]], color="k", linewidth=0.5)
		plt.plot(common, [-2] * len(common), '.', markersize=30, color='#084D14', label="Common variance")

		# plot the best latency with guidline
		best_lat_x = smallest_diff_index
		best_lat_y = smoothed_Q3[smallest_diff_index]
		plt.plot([best_lat_x], [best_lat_y + 2], '.', markersize=15,  color='k', label="Best latency (no)")
		plt.plot([best_lat_x] * 2, [best_lat_y, best_lat_y + 2], color="k", linewidth=0.5)

		# plot an EES area
		plt.axvspan(xmin=0, xmax=ees_zone_time, color='g', alpha=0.3, label="EES area")

		# plot Q1 and Q3 areas, and median
		plt.plot(smoothed_Q3, color='k', linewidth=3.5, label="Q1/Q3 values")
		plt.plot(smoothed_Q1, color='k', linewidth=3.5)
		plt.plot(smooth(median, 5), linestyle='--', color='k', label="Median")

		# plot extrema (minima and maxima)
		for q_data in extremuma.values():
			for name, extremuma in q_data.items():
				plt.plot(extremuma[k_indexes, :], extremuma[k_values, :], '.',
				         color=max_color if 'maxima' in name else min_color)

		# figure properties
		plt.suptitle(f"Slice #{slice_index + 1}")
		plt.xticks(range(0, 101, 4), [int(x * bio_step) for x in range(0, 101, 4)])
		plt.tight_layout()
		plt.legend()
		plt.show()

	plt.close('all')

	yticks = []

	plt.subplots(figsize=(16, 9))

	y_offset = 3
	slice_in_ms = 25

	for slice_index, data in enumerate(splitted_per_slice_boxplots):
		data += slice_index * y_offset  # is a link (!)
		plt.fill_between(shared_x, data[:, k_box_Q3], data[:, k_box_Q1], color='r', alpha=0.7)
		plt.plot(shared_x, data[:, k_median], linestyle='--', color='k')
		yticks.append(data[:, k_median][0])

	print("GLOBAL: ", global_lat_indexes)

	lat_x = [x * bio_step for x in global_lat_indexes]
	lat_y = [splitted_per_slice_boxplots[slice_index][:, k_median][lat] for slice_index, lat in enumerate(global_lat_indexes)]
	plt.plot(lat_x, lat_y, linewidth=3, color='g')

	# plt.xticks(range(100), [x * bio_step for x in range(100) if x % 4 == 0])
	plt.yticks(yticks, range(1, slices_number + 1))
	plt.xlim(0, 25)

	plt.show()


	raise Exception


	splitted_per_slice_boxplots = np.split(np.array([calc_boxplots(dot) for dot in bio_data.T]), slices_number)


	bio_means = np.sum(np.array([np.absolute(normalization(data, -1, 1)) for data in read_data(f"{data_folder}/bio_15.hdf5")]), axis=0)

	neuron_means = np.sum(np.array([np.absolute(normalization(data, -1, 1)) for data in select_slices(f"{data_folder}/neuron_15.hdf5", 0, 12000)]), axis=0)

	gras_means = np.sum(np.array([np.absolute(normalization(data, -1, 1)) for data in select_slices(f"{data_folder}/gras_15.hdf5", 10000, 22000)]), axis=0)

	# calculating latencies and amplitudes of mean values
	bio_means_lat = sim_process(bio_means, bio_step, inhibition_zero=True, debugging=True)[0]
	bio_means_amp = sim_process(bio_means, bio_step, inhibition_zero=True, after_latencies=after_latencies)[1]

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
		pca = PCA(n_components=2)     # create PCA instance
		pca.fit(coords)               # fit the model with coords
		center = np.array(pca.mean_)  # get the center (mean value)

		# calc vectors
		vectors = []
		for v_length, vector in zip(pca.explained_variance_, pca.components_):
			v = vector * 3 * np.sqrt(v_length)
			vectors.append((center, center + v))

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
		ellipse = Ellipse(xy=center, width=ellipse_width, height=ellipse_height, angle=angle_degrees * sign)
		ellipse.set_fill(False)
		ellipse.set_edgecolor(hex2rgb(color))
		ax.add_artist(ellipse)

		# fill convex
		hull = ConvexHull(coords)
		S = poly_area(coords[hull.vertices, X], coords[hull.vertices, Y])
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


if __name__ == "__main__":
	run()
