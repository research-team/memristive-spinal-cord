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
	# return array
	return np.array([point for point, diff in zip(array, np.diff(array, n=1)) if diff > 1] + [array[-1]])


def smooth(y, box_pts):
	box = np.ones(box_pts) / box_pts
	return np.convolve(y, box, mode='same')


def run(with_mono=False, debugging=True):
	after_latencies = not with_mono
	# keys
	X = 0
	Y = 1
	# read all bio data (list of tests)
	bio_data = np.array(read_data(f"{data_folder}/bio_15.hdf5"))
	slices_number = int(len(bio_data[0]) / (25 / bio_step))

	splitted_per_slice = np.split(np.array([calc_boxplots(dot) for dot in bio_data.T]), slices_number)

	first_10ms = slice(0, int(10 / bio_step))

	shared_x = np.arange(25 / bio_step) * bio_step

	time_offset = int(10 / bio_step)

	k_low = 'low_Q1'
	k_high = 'high_Q3'
	k_minima = 'e_minima'
	k_maxima = 'e_maxima'
	k_indexes = 0
	k_values = 1

	global_lat_indexes = []

	# compute per slice data
	for slice_index, slice_data in enumerate(splitted_per_slice):
		extremuma = {k_low: {}, k_high: {}}


		y_original_high = slice_data[:, k_box_high]
		y_original_low = slice_data[:, k_box_low]

		y_high = smooth(slice_data[:, k_box_high], 2)
		y_low = smooth(slice_data[:, k_box_low], 2)

		# fix the last broken data after smoothing
		y_high[-2:] = y_original_high[-2:]
		y_low[-2:] = y_original_low[-2:]

		median = slice_data[:, k_median]

		# for high Q3
		ind = remove_flatters(argrelextrema(y_high, np.greater_equal)[0])
		extremuma[k_high][k_maxima] = np.stack((ind, y_high[ind]))
		ind = remove_flatters(argrelextrema(y_high, np.less_equal)[0])
		extremuma[k_high][k_minima] = np.stack((ind, y_high[ind]))


		# for low Q1
		ind = remove_flatters(argrelextrema(y_low, np.greater_equal)[0])
		extremuma[k_low][k_maxima] = np.stack((ind, y_low[ind]))
		ind = remove_flatters(argrelextrema(y_low, np.less_equal)[0])
		extremuma[k_low][k_minima] = np.stack((ind, y_low[ind]))

		differed_per_iter = np.abs(y_high - y_low)

		# [0, 5, 7]
		# [3, 6, 8]
		# [0, 3, 5, 6, 7, 8]
		a = [int(ind_dot) for ind_dot in extremuma[k_high][k_maxima][0, :] if ind_dot > time_offset]
		b = [int(ind_dot) for ind_dot in extremuma[k_high][k_minima][0, :] if ind_dot > time_offset]
		c = np.array(sorted(a + b))

		print(a)
		print(b)
		print(c)

		lat1 = []

		for dot_left, dot_right in zip(c, c[1:]):
			dot_left += 1
			if dot_right - dot_left == 0:
				global_ind = dot_right
			else:
				local_ind = np.argmin(differed_per_iter[dot_left:dot_right])
				global_ind = local_ind + dot_left
			lat1.append(global_ind)

		plt.plot(lat1, y_high[lat1], '.', markersize=20, color='orange', alpha=0.9)


		""" =============== """
		a = [int(ind_dot) for ind_dot in extremuma[k_low][k_maxima][0, :] if ind_dot > time_offset]
		b = [int(ind_dot) for ind_dot in extremuma[k_low][k_minima][0, :] if ind_dot > time_offset]
		c = np.array(sorted(a + b))

		lat2 = []

		for dot_left, dot_right in zip(c, c[1:]):
			dot_left += 1
			if dot_right - dot_left == 0:
				global_ind = dot_right
			else:
				local_ind = np.argmin(differed_per_iter[dot_left:dot_right])
				global_ind = local_ind + dot_left
			lat2.append(global_ind)

		plt.plot(lat2, y_low[lat2], '.', markersize=20, color='green', alpha=0.9)
		""" =============== """
		print(f"lat {slice_index + 1}")
		print(f"lat1 = {lat1}")
		print(f"lat2 = {lat2}")
		common = sorted(list(set(lat1) & set(lat2)))
		print(f"comm = {common}")
		# plot common found diffs
		plt.plot(common, [-1] * len(common), '.', markersize=30, color='#35A53F')

		smallest_diff_index = np.argmin(differed_per_iter[common])
		print(f"differed_per_iter = {differed_per_iter[common]}")
		smallest_diff_index += common[int(smallest_diff_index)]
		print(f"smallest_diff_index {smallest_diff_index}")

		smallest_diff_index = common[0]

		global_lat_indexes.append(smallest_diff_index)

		plt.plot([smallest_diff_index], [y_high[smallest_diff_index] + 2], '.', markersize=15,  color='k')


		plt.axvspan(xmin=0, xmax=time_offset, color='g', alpha=0.3)
		plt.suptitle(f"Slice #{slice_index + 1}")
		plt.plot(y_high, color='k')
		plt.plot(y_low, color='k')
		plt.plot(smooth(median, 5), linestyle='--', color='k')

		for q_data in extremuma.values():
			for name, extremuma in q_data.items():
				plt.plot(extremuma[k_indexes, :], extremuma[k_values, :], '.', color='r' if 'maxima' in name else 'b')
		plt.xticks(range(0, 101, 4), [int(x * bio_step) for x in range(0, 101, 4)])
		plt.tight_layout()
		plt.show()

		"""
		continue
		print(extremuma)
		print(extremuma[k_high][k_maxima][k_values, :])

		# finding EES
		max_diff_in_high_index = np.argmax(np.abs(e_maxima_value[first_10ms] - median[0]))
		max_diff_in_low_index = np.argmax(np.abs(e_minima_value[first_10ms] - median[0]))

		if e_maxima_value[max_diff_in_high_index] > e_minima_value[max_diff_in_low_index]:
			ees_index = e_maxima_index[max_diff_in_high_index]
		else:
			ees_index = e_maxima_index[max_diff_in_low_index]

		print_e("maxima", e_maxima_index, e_maxima_value)
		print_e("minima", e_minima_index, e_minima_value)

		# (III parameter) finding number of waves
		# exclude first 10 ms
		e_maxima_poly_index = np.compress(e_maxima_index >= time_offset, e_maxima_index)
		e_minima_poly_index = np.compress(e_minima_index >= time_offset, e_minima_index)
		e_maxima_poly_value = e_maxima_value[-len(e_maxima_poly_index):]
		e_minima_poly_value = e_minima_value[-len(e_minima_poly_index):]

		# get difference between
		differed_per_iter = np.abs(y_high - y_low)

		dot_index = 1
		waves = 0

		plt.figure(figsize=(16, 9))

		tmp_all_lat_per_slice = []
		tmp_all_lat_per_slice_minima = []

		max_poly = (0, 0)
		max_poly_minima = (0, 0)

		for dot_left, dot_right in zip(e_maxima_index, e_maxima_index[1:]):
			dot_left += 1

			if dot_right > time_offset:
				if dot_left < time_offset:
					dot_left = time_offset
				local_ind = np.argmin(differed_per_iter[dot_left:dot_right])
				global_ind = local_ind + dot_left

				# if waves == 0:
				# 	lat_per_slice.append((global_ind, y_high[dot_left:dot_right][local_ind]))

				local_MAX_ind = np.argmax(differed_per_iter[dot_left:dot_right])
				global_MAX_ind = local_MAX_ind + dot_left

				if differed_per_iter[local_MAX_ind] > max_poly[Y]:
					max_poly = (global_ind, y_high[global_ind])


				tmp_all_lat_per_slice.append((global_ind, y_high[dot_left:dot_right][local_ind]))

				if debugging:
					plt.plot([global_ind] * 2, [y_low[global_ind], 5], linewidth=2, color='k')
					plt.text(global_ind, 5, f"Lat {dot_index}", rotation=90)
					plt.fill_between(range(dot_left, dot_right), y_low[dot_left:dot_right], y_high[dot_left:dot_right], alpha=0.5)
					print(f"({dot_left}, {dot_right})")
				waves += 1
				dot_index += 1


		for dot_left, dot_right in zip(e_minima_index, e_minima_index[1:]):
			dot_left += 1

			if dot_right > time_offset:
				if dot_left < time_offset:
					dot_left = time_offset
				local_ind = np.argmin(differed_per_iter[dot_left:dot_right])
				global_ind = local_ind + dot_left

				local_MAX_ind = np.argmax(differed_per_iter[dot_left:dot_right])

				if differed_per_iter[local_MAX_ind] > max_poly[Y]:
					max_poly_minima = (global_ind, y_high[global_ind])

				tmp_all_lat_per_slice_minima.append((global_ind, y_high[dot_left:dot_right][local_ind]))


		lat_per_slice.append(max_poly)
		lat_per_slice_minima.append(max_poly_minima)

		all_lat_maxima.append(tmp_all_lat_per_slice)
		all_lat_minima.append(tmp_all_lat_per_slice_minima)



		plt.plot(y_original_high, color='r')
		plt.plot(y_original_low, color='r')
		plt.plot(y_high, color='k')
		plt.plot(y_low, color='k')

		plt.axvspan(0, time_offset, alpha=0.2, color='green')

		plt.plot(median, color='k', linestyle='--')
		plt.plot(e_maxima_index, e_maxima_value, '.', color='r')
		plt.plot(e_minima_index, e_minima_value, '.', color='b')

		plt.axvline(x=ees_index, color='#FE9C1F', linewidth=5)

		plt.plot(e_maxima_poly_index, e_maxima_poly_value, '.', color='r', markersize=15)
		plt.plot(e_minima_poly_index, e_minima_poly_value, '.', color='b', markersize=15)

		X = np.arange(0, len(slice_data) + 1, 4)
		plt.xticks(X, X * bio_step)
		plt.grid(axis='x')

		plt.suptitle(f"Slice #{slice_index + 1}, waves={waves}")
		# plt.tight_layout()
		# plt.savefig(f"/home/alex/{slice_index + 1}.png", format="png", dpi=120)
		plt.show()
		plt.close()

	# assert len(lat_per_slice) == slices_number
	print(len(lat_per_slice), lat_per_slice)

	# build plot
	"""
	plt.close('all')

	yticks = []

	plt.subplots(figsize=(16, 9))

	y_offset = 3
	slice_in_ms = 25

	bio_data = read_data(f"{data_folder}/bio_15.hdf5")
	slices_number = int(len(bio_data[0]) / (slice_in_ms / bio_step))

	splitted_per_slice = np.split(np.array(bio_data[0]), slices_number)
	shared_x = np.arange(slice_in_ms / bio_step) * bio_step

	for slice_index, data in enumerate(splitted_per_slice):
		data += slice_index * y_offset  # is a link (!)
		plt.plot(shared_x, data, color='r')
		yticks.append(data[0])

	dots = [15.044, 15, 14.32, 16.6, 16.9, 12.1, 16.2, 16.5, 19.7, 24.7, 24.7, 24.7]
	vals = [splitted_per_slice[ind][int(dot / bio_step)] for ind, dot in enumerate(dots)]

	plt.plot(dots, vals, '.', markersize=10, color='k')
	plt.plot(dots, vals, color='b', linewidth=3)

	print("GLOBAL: ", global_lat_indexes)

	lat_x = [x * bio_step for x in global_lat_indexes]
	lat_y = [splitted_per_slice[slice_index][lat] for slice_index, lat in enumerate(global_lat_indexes)]
	plt.plot(lat_x, lat_y, linestyle='--', color='g')

	# plt.xticks(range(100), [x * bio_step for x in range(100) if x % 4 == 0])
	plt.yticks(yticks, range(1, slices_number + 1))
	plt.xlim(0, 25)

	plt.show()


	raise Exception



	splitted_per_slice = np.split(np.array([calc_boxplots(dot) for dot in bio_data.T]), slices_number)


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
