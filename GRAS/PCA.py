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
k_box_high = 1
k_box_low = 2
k_whiskers_high = 3
k_whiskers_low = 4
k_fliers_high = 5
k_fliers_low = 6

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


def PolyArea(x, y):
	return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

percents = [25, 50, 75]

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


def print_e(t, x, y=None):
	print(t)
	for v in x:
		print("{:7.2f}".format(v), end='')
	print()
	if y is not None:
		for v in y:
			print("{:7.2f}".format(v), end='')
		print()
	print("- " * 40)


def remove_flatters(array):
	return np.array([point for point, diff in zip(array, np.diff(array, n=1)) if diff > 1] + [array[-1]])


def run(with_mono=False, debugging=True):
	after_latencies = not with_mono
	# keys
	X = 0
	Y = 1
	# read all bio data (list of tests)
	bio_data = np.array(read_data(f"{data_folder}/bio_15.hdf5"))
	#
	slices_number = int(len(bio_data[0]) / (25 / 0.25))

	splitted_per_slice = np.split(np.array([calc_boxplots(dot) for dot in bio_data.T]), slices_number)

	first_10ms = slice(0, int(10 / bio_step))

	shared_x = np.arange(25 / bio_step) * bio_step

	after_ees_step_offset = int(4 / bio_step)

	for slice_index, slice_data in enumerate(splitted_per_slice):
		# check sign
		# if max()
		y_high = slice_data[:, k_box_high]
		y_low = slice_data[:, k_box_low]
		median = slice_data[:, k_median]

		e_maxima_index = remove_flatters(argrelextrema(y_high, np.greater_equal)[0])
		e_minima_index = remove_flatters(argrelextrema(y_low, np.less_equal)[0])
		e_maxima_value = y_high[e_maxima_index]
		e_minima_value = y_low[e_minima_index]

		# finding EES
		max_diff_in_high_index = np.argmax(np.absolute(e_maxima_value[first_10ms] - median[0]))
		max_diff_in_low_index = np.argmax(np.absolute(e_minima_value[first_10ms] - median[0]))

		if e_maxima_value[max_diff_in_high_index] > e_minima_value[max_diff_in_low_index]:
			ees_index = e_maxima_index[max_diff_in_high_index]
		else:
			ees_index = e_maxima_index[max_diff_in_low_index]

		e_maxima_index = remove_flatters(argrelextrema(y_high, np.greater_equal)[0])
		e_minima_index = remove_flatters(argrelextrema(y_low, np.less_equal)[0])
		e_maxima_value = y_high[e_maxima_index]

		print_e("maxima", e_maxima_index, e_maxima_value)
		print_e("minima", e_minima_index, e_minima_value)

		# (III parameter) finding number of waves
		# exclude first 4 ms after EES
		e_maxima_poly_index = np.compress(e_maxima_index > ees_index + after_ees_step_offset, e_maxima_index)
		e_minima_poly_index = np.compress(e_minima_index > ees_index + after_ees_step_offset, e_minima_index)
		e_maxima_poly_value = e_maxima_value[-len(e_maxima_poly_index):]
		e_minima_poly_value = e_minima_value[-len(e_minima_poly_index):]

		# for
		stacked_per_iter = np.stack((y_high, y_low))
		differed_per_iter = np.abs(np.diff(stacked_per_iter, axis=0)[0])

		for dot_left, dot_right in zip(e_maxima_index, e_maxima_index[1:]):
			if dot_left >= ees_index + after_ees_step_offset:
				dot_right += 1

				local_ind = np.argmin(differed_per_iter[dot_left:dot_right])
				global_ind = local_ind + dot_left

				if debugging:
					plt.plot([global_ind], [y_low[global_ind] + differed_per_iter[global_ind] / 2], '*', markersize=10, color='w')
					plt.plot([global_ind] * 2, [y_low[global_ind], y_high[global_ind]], linewidth=2, color='w')
					plt.fill_between(range(dot_left, dot_right), y_low[dot_left:dot_right], y_high[dot_left:dot_right])
					print(dot_left, dot_right, global_ind)


		plt.plot(y_high, color='k')
		plt.plot(y_low, color='k')

		plt.plot(median, color='k', linestyle='--')
		plt.plot(e_maxima_index, e_maxima_value, '.', color='r')
		plt.plot(e_minima_index, e_minima_value, '.', color='b')

		plt.axvline(x=ees_index, color='#FE9C1F', linewidth=5)
		plt.axvline(x=0, color='cyan', linewidth=3, linestyle='--')

		plt.plot(e_maxima_poly_index, e_maxima_poly_value, '.', color='r', markersize=15)
		plt.plot(e_minima_poly_index, e_minima_poly_value, '.', color='b', markersize=15)

		iii = min(len(e_maxima_poly_index), len(e_minima_poly_index))

		X = np.arange(0, len(slice_data) + 1, 4)
		# plt.xticks(X, X * bio_step)
		plt.grid(axis='x')

		plt.suptitle(f"Slice #{slice_index}, I=? II=? III={iii}")
		plt.show()
	raise Exception

	# build plot
	yticks = []
	shared_x = np.arange(25 / 0.25) * 0.25

	fig, ax = plt.subplots(figsize=(16, 9))

	for i, data in enumerate(splitted_per_slice):
		for i_da, da in enumerate(data):
			ax.scatter([i_da * 0.25] * len(da), [d + i * 6 for d in da], color='k', s=3)
			for t in da:
				ax.text(i_da * 0.25 + 0.02, t + i * 6, "{:.2f}".format(t), fontsize=5)

		data += i * 6

		ax.fill_between(shared_x, data[:, k_fliers_low], data[:, k_fliers_high], alpha=0.1, color='r')
		ax.fill_between(shared_x, data[:, k_whiskers_low], data[:, k_whiskers_high], alpha=0.3, color='r')
		ax.fill_between(shared_x, data[:, k_box_low], data[:, k_box_high], alpha=0.6, color='r')
		ax.plot(shared_x, data[:, k_median], color='k', linewidth=0.7)
		yticks.append(data[0, k_median])
	plt.show()

	splitted_per_slice = np.split(np.array([calc_boxplots(dot) for dot in bio_data.T]), slices_number)

	# build plot
	yticks = []
	shared_x = np.arange(25 / 0.25) * 0.25

	fig, ax = plt.subplots(figsize=(16, 9))

	for i, data in enumerate(splitted_per_slice):
		for i_da, da in enumerate(data):
			ax.scatter([i_da * 0.25] * len(da), [d + i * 6 for d in da], color='k', s=3)
			for t in da:
				ax.text(i_da * 0.25 + 0.02, t + i * 6, "{:.2f}".format(t), fontsize=5)

		data += i * 6
		ax.fill_between(shared_x, data[:, 6], data[:, 5], alpha=0.1, color='r')  # 6 f_low, 5 f_high
		ax.fill_between(shared_x, data[:, 4], data[:, 3], alpha=0.3, color='r')  # 4 w_low, 3 w_high
		ax.fill_between(shared_x, data[:, 2], data[:, 1], alpha=0.6, color='r')  # 2 b_low, 1 b_high
		ax.plot(shared_x, data[:, 0], color='k', linewidth=0.7)  # 0 med
		yticks.append(data[0, 0])
	plt.show()

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
		S = PolyArea(coords[hull.vertices, X], coords[hull.vertices, Y])
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
