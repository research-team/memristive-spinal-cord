import numpy as np
import pylab as plt
import h5py as hdf5
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy.signal import argrelextrema
from sklearn.preprocessing import normalize
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
		data_by_test = [-test_values[:] for test_values in file.values()]
	return data_by_test


def select_slices(path, begin, end):
	return [-data[begin:end] for data in read_data(path)]


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


def normalization(data, a=0, b=1, zero_relative=False):
	"""
	Normalization in [a, b] interval
	x` = (b - a) * (xi - min(x)) / (max(x) - min(x)) + a
	Args:
		data (list): data for normalization
		a (float or int): left interval a
		b (float or int): right interval b
		zero_relative (bool): if True -- recalculate data where 0 is the first element and -1 is min(EES)
	Returns:
		list: normalized data
	"""
	# checking on errors
	if a >= b:
		raise Exception("Left interval 'a' must be fewer than right interval 'b'")

	if zero_relative:
		first = data[0]
		minimal = abs(min(data))

		return [(volt - first) / minimal for volt in data]
	else:
		min_x = min(data)
		max_x = max(data)
		const = (b - a) / (max_x - min_x)

		return [(x - min_x) * const + a for x in data]


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


def straighten_data(points):
	"""

	Args:
		points:

	Returns:

	"""
	sim = select_slices(f"{data_folder}/gras_15.hdf5", 10000, 22000)[1][::10]
	PPP = np.stack((range(len(sim)), sim), axis=1)
	plt.plot(PPP[:,1])
	plt.show()

	P = np.stack((range(len(points)), points), axis=1)

	pca = PCA(n_components=2)
	pca.fit(P)

	##
	center = np.array(pca.mean_)  # get the center (mean value)

	# calc vectors
	vectors = []
	for v_length, vector in zip(pca.explained_variance_, pca.components_):
		y = vector * 3 * np.sqrt(v_length)
		vectors.append((center, center + y))

	def draw_vector(v0, v1, ax=None):
		ax = ax or plt.gca()
		arrowprops = dict(arrowstyle='->',
		                  linewidth=2,
		                  shrinkA=0, shrinkB=0)
		ax.annotate('', v1, v0, arrowprops=arrowprops)

	# plot data
	plt.scatter(P[:, 0], P[:, 1], alpha=0.2)
	plt.plot(P[:, 0], P[:, 1])

	for length, vector in zip(pca.explained_variance_, pca.components_):
		v = vector * 3 * np.sqrt(length)
		draw_vector(pca.mean_, pca.mean_ + v)

	plt.show()

	# pca.components_ : array, shape (n_components, n_features)
	# cos theta
	ct = pca.components_[0, 0]
	# sin theta
	st = pca.components_[0, 1]

	# One possible value of theta that lies in [0, pi]
	t = np.arccos(ct)

	# If t is in quadrant 1, rotate CLOCKwise by t
	if ct > 0 and st > 0:
		t *= -1
	# If t is in Q2, rotate COUNTERclockwise by the complement of theta
	elif ct < 0 and st > 0:
		t = np.pi - t
	# If t is in Q3, rotate CLOCKwise by the complement of theta
	elif ct < 0 and st < 0:
		t = -(np.pi - t)
	# If t is in Q4, rotate COUNTERclockwise by theta, i.e., do nothing
	elif ct > 0 and st < 0:
		pass

	# Manually build the ccw rotation matrix
	rotmat = np.array([[np.cos(t), -np.sin(t)],
	                   [np.sin(t), np.cos(t)]])
	# Apply rotation to each row of m (@ is a matrix multiplication)
	m2 = (rotmat @ P.T).T

	# Center the rotated point cloud at (0, 0)
	m2 -= m2.mean(axis=0)

	fig, ax = plt.subplots()
	plot_kws = {'alpha': '0.75',
	            'edgecolor': 'white',
	            'linewidths': 0.75}

	pca = PCA(n_components=2)
	pca.fit(PPP)
	# pca.components_ : array, shape (n_components, n_features)
	# cos theta
	ct = pca.components_[0, 0]
	# sin theta
	st = pca.components_[0, 1]

	# One possible value of theta that lies in [0, pi]
	t = np.arccos(ct)

	# If t is in quadrant 1, rotate CLOCKwise by t
	if ct > 0 and st > 0:
		t *= -1
	# If t is in Q2, rotate COUNTERclockwise by the complement of theta
	elif ct < 0 and st > 0:
		t = np.pi - t
	# If t is in Q3, rotate CLOCKwise by the complement of theta
	elif ct < 0 and st < 0:
		t = -(np.pi - t)
	# If t is in Q4, rotate COUNTERclockwise by theta, i.e., do nothing
	elif ct > 0 and st < 0:
		pass

	# Manually build the ccw rotation matrix
	rotmat = np.array([[np.cos(t), -np.sin(t)],
	                   [np.sin(t), np.cos(t)]])
	# Apply rotation to each row of m (@ is a matrix multiplication)
	sim_m2 = (rotmat @ PPP.T).T

	# Center the rotated point cloud at (0, 0)
	sim_m2 -= sim_m2.mean(axis=0)

	plot_kws = {'alpha': '0.75',
	            'edgecolor': 'white',
	            'linewidths': 0.75}



	# ax.scatter(range(len(P[:, 0])), m2[:, 1] * 5, **plot_kws)
	n_bio = normalization(np.append(m2[:, 1], 0))
	# ax.scatter(range(len(sim_m2[:, 1])), sim_m2[:, 1], **plot_kws)
	n_sim = normalization(np.append(sim_m2[:, 1], 0))

	ax.plot(range(len(n_bio)), n_bio - (n_bio[-1] - n_sim[-1]), color='orange', label="BIO", linewidth=3)
	ax.plot(range(len(n_sim)), n_sim, color='g', label="SIM", linewidth=3)

	# plt.axhline(y=n_bio[-1], color='orange', label="BIO 0", linestyle='--')
	plt.axhline(y=n_sim[-1], color='g', label="SIM 0", linestyle='--')
	plt.legend()
	plt.show()


def slice_metainfo(runs_data, ees_hz, debugging=True):
	"""
	ToDo fill the docstring
	Args:
		runs_data:
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

	original = np.array(runs_data[0]) - runs_data[0][0]
	normalized_old = normalization(list(original), zero_relative=True)


	plt.axhspan(ymin=-1, ymax=1, color='g', alpha=0.1)


	original_reversed = np.negative(original)
	normalized_lib = normalize(original_reversed.reshape(1, -1), norm="max")

	normalized_strange = (original - original.min(0)) / original.ptp(0)

	plt.plot(original, label='original')
	plt.plot(normalized_old, label='normalized_old', linewidth=3)
	plt.plot(np.negative(normalized_lib.T), label='normalized_lib')
	plt.plot(normalized_strange, label='???')
	plt.legend()
	plt.show()


	# set ees area, before that time we don't try to find latencies
	ees_zone_time = int(7 / bio_step)
	shared_x = np.arange(slice_in_ms / bio_step) * bio_step

	# read all bio data (list of tests)
	runs_data = np.array(runs_data)
	# get number of slices based on length of the bio data
	slices_number = int(len(runs_data[0]) / (slice_in_ms / bio_step))

	# split original data only for visualization
	splitted_per_slice_original = np.split(runs_data.T, slices_number)
	# calc boxplot per step and split it by number of slices
	splitted_per_slice_boxplots = np.split(np.array([calc_boxplots(dot) for dot in runs_data.T]), slices_number)

	# compute latency per slice
	for slice_index, slice_data in enumerate(splitted_per_slice_boxplots):
		'''[1] preparing data'''
		# get data by keys
		data_Q1 = slice_data[:, k_fliers_low]
		data_Q3 = slice_data[:, k_fliers_high]
		median = slice_data[:, k_median]
		# smooth the data to avoid micropeaks
		smoothed_Q1 = smooth(data_Q1, 2)
		smoothed_Q3 = smooth(data_Q3, 2)
		smoothed_median = smooth(median, 2)
		# fix the last broken data after smoothing (found by experimental way)
		smoothed_Q1[-2:] = data_Q1[-2:]
		smoothed_Q3[-2:] = data_Q3[-2:]
		smoothed_median[-2:] = median[-2:]
		# get a delta of NOT smoothed data
		delta_data = np.abs(data_Q1 - data_Q3)
		# get a delta of smoothed data
		delta_smoothed_data = np.abs(smoothed_Q1 - smoothed_Q3)

		'''[2] finding extremuma'''
		# find all Q1 extremuma indexes and values
		e_all_Q1_minima_indexes, e_all_Q1_minima_values = find_extremuma(smoothed_Q1, np.less_equal)
		e_all_Q1_maxima_indexes, e_all_Q1_maxima_values = find_extremuma(smoothed_Q1, np.greater_equal)
		# find all Q3 extremuma indexes and values
		e_all_Q3_minima_indexes, e_all_Q3_minima_values = find_extremuma(smoothed_Q3, np.less_equal)
		e_all_Q3_maxima_indexes, e_all_Q3_maxima_values = find_extremuma(smoothed_Q3, np.greater_equal)

		'''[3] finding EES'''
		# get the lowest (Q1) mono minima extremuma
		e_mono_Q1_minima_indexes = e_all_Q1_minima_indexes[e_all_Q1_minima_indexes < ees_zone_time].astype(int)
		# find an index of the biggest delta between median[0] and mono extremuma
		max_delta_Q1_index, _ = max_at(np.abs(smoothed_Q1[e_mono_Q1_minima_indexes] - median[0]))
		# get index of extremuma with the biggest delta
		ees_index = e_mono_Q1_minima_indexes[max_delta_Q1_index]

		'''[3] find a good poly area'''
		l_poly_border = ees_zone_time
		for index_Q1, index_Q3 in zip(e_all_Q1_maxima_indexes, e_all_Q3_maxima_indexes):
			index = index_Q1 if index_Q1 > index_Q3 else index_Q3
			if index > ees_index and index > ees_zone_time:
				l_poly_border = index
				break
		r_poly_border = int(slice_in_ms / bio_step)
		poly_area = slice(l_poly_border, r_poly_border)

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
		latencies_Q1 = []
		for e_left, e_right in zip(e_poly_Q1_indexes, e_poly_Q1_indexes[1:]):
			# if dots are too close
			e_left += 1
			e_right -= 1
			if e_right - e_left <= 0:
				continue
			if e_right - e_left == 1:
				latency_index = e_right
			# else find indexes of minimal variance index in [dot left, dot right) interval
			else:
				latency_index = e_left + min_at(delta_smoothed_data[e_left:e_right])[k_index]
			latencies_Q1.append(latency_index)

		# find latencies in Q3
		latencies_Q3 = []
		for e_left, e_right in zip(e_poly_Q3_indexes, e_poly_Q3_indexes[1:]):
			e_left += 1
			e_right -= 1
			if e_right - e_left <= 0:
				continue
			# if dots are too close
			if e_right - e_left == 1:
				latency_index = e_right
			# else find indexes of minimal variance index in [dot left, dot right) interval
			else:
				latency_index = e_left + min_at(delta_smoothed_data[e_left:e_right])[k_index]
			latencies_Q3.append(latency_index)

		'''[5] finding best borders by delta'''
		# find extremuma for deltas of Q1 and Q3
		e_delta_minima_indexes, e_delta_minima_values = find_extremuma(delta_data[poly_area], np.less_equal)
		e_delta_maxima_indexes, e_delta_maxima_values = find_extremuma(delta_data[poly_area], np.greater_equal)

		# prepare data for concatenating dots into one list (per parameter)
		common_lenght = len(e_delta_minima_indexes) + len(e_delta_maxima_indexes)
		merged_names = [None] * common_lenght
		merged_indexes = [None] * common_lenght
		merged_values = [None] * common_lenght

		# who located earlier -- max or min
		min_starts = 0 if e_delta_minima_indexes[0] < e_delta_maxima_indexes[0] else 1
		max_starts = 1 if e_delta_minima_indexes[0] < e_delta_maxima_indexes[0] else 0

		# fill minima lists based on the precedence
		merged_names[min_starts::2] = ['min'] * len(e_delta_minima_indexes)
		merged_indexes[min_starts::2] = e_delta_minima_indexes
		merged_values[min_starts::2] = e_delta_minima_values
		# the same for the maxima
		merged_names[max_starts::2] = ['max'] * len(e_delta_maxima_indexes)
		merged_indexes[max_starts::2] = e_delta_maxima_indexes
		merged_values[max_starts::2] = e_delta_maxima_values
		# convert them to the array for usability
		merged_names = np.array(merged_names)
		merged_values = np.array(merged_values)
		merged_indexes = np.array(merged_indexes)

		# get difference of merged indexes with step 1
		differed_indexes = np.abs(np.diff(merged_indexes, n=1))
		# filter closest indexes and add the True to the end, because the last dot doesn't have diff with next point
		is_index_ok = np.append(differed_indexes > min_index_interval, True)

		index = 0
		while index < len(is_index_ok):
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
		e_delta_minima_indexes = merged_indexes[min_mask]
		e_delta_maxima_indexes = merged_indexes[max_mask]

		# find the best right border
		i_min = 0
		r_best_border = 0
		l_best_border = e_delta_minima_indexes[0] + l_poly_border if e_delta_minima_indexes[0] > e_delta_maxima_indexes[0] else e_delta_minima_indexes[1] + l_poly_border
		is_border_found = False

		while not is_border_found and i_min < len(merged_names):
			if merged_names[i_min] == 'min' and is_index_ok[i_min]:
				for i_max in range(i_min + 1, len(merged_names)):
					if merged_names[i_max] == 'max' and is_index_ok[i_max] and abs(merged_values[i_max] - merged_values[i_min]) > allowed_diff:
						r_best_border = merged_indexes[i_max] + l_poly_border
						is_border_found = True
						break
			i_min += 1

		if r_best_border < l_best_border:
			l_best_border, r_best_border = r_best_border, l_best_border

		if not is_border_found:
			raise Exception("WHERE IS MAXIMAL BORDER???")

		'''[6] finding best latency in borders'''
		best_latency = (0, 0)
		# find the latency in Q1 and Q3 data (based on the maximal variance)
		while best_latency[k_value] == 0:
			for lat_Q1 in filter(lambda dot: l_best_border < dot < r_best_border, latencies_Q1):
				if delta_smoothed_data[lat_Q1] > best_latency[k_value]:
					best_latency = (lat_Q1, delta_smoothed_data[lat_Q1])
			for lat_Q3 in filter(lambda dot: l_best_border < dot < r_best_border, latencies_Q3):
				if delta_smoothed_data[lat_Q3] > best_latency[k_value]:
					best_latency = (lat_Q3, delta_smoothed_data[lat_Q3])
			if best_latency[k_value] == 0:
				r_best_border += 1

		# append found latency to the global list of the all latencies per slice
		global_lat_indexes.append(best_latency[k_index])

		# all debugging info
		if debugging:
			print("\n")
			print("- " * 20)
			print("{:^40}".format(f"SLICE {slice_index + 1}"))
			print("- " * 20)
			print(f"  EES was found at: {ees_index * bio_step}ms (index {ees_index})")
			print(f" minima indexes Q1: {e_poly_Q1_minima_indexes}")
			print(f" maxima indexes Q1: {e_poly_Q1_maxima_indexes}")
			print(f" merged indexes Q1: {e_poly_Q1_indexes}")
			print(f"found latencies Q1: {latencies_Q1}")
			print("- " * 20)
			print(f" minima indexes Q3: {e_poly_Q3_minima_indexes}")
			print(f" maxima indexes Q3: {e_poly_Q3_maxima_indexes}")
			print(f" merged indexes Q3: {e_poly_Q3_indexes}")
			print(f"found latencies Q3: {latencies_Q3}")
			print("- " * 20)
			print(f" poly answers area: [{l_poly_border * bio_step}, {r_poly_border * bio_step}]ms")
			print(f"  hist indexes min: {e_delta_minima_indexes}")
			print(f"  hist indexes max: {e_delta_maxima_indexes}")
			print(f"      merged names: {merged_names}")
			print(f"    merged indexes: {merged_indexes}")
			print(f"     merged values: {merged_values}")
			print(f"  differed_indexes: {differed_indexes}")
			print(f" indexes that okay: {is_index_ok}")
			print(f" best area between: {l_best_border * bio_step}ms and {r_best_border * bio_step}ms")
			print(f"        latency at: {best_latency[0] * bio_step}ms")

			gridsize = (3, 1)

			fig = plt.figure(figsize=(16, 9))
			ax1 = plt.subplot2grid(gridsize, (0, 0), rowspan=2)
			ax2 = plt.subplot2grid(gridsize, (2, 0), sharex=ax1)

			ax1.title.set_text(f"Bio data of slice #{slice_index + 1}")
			# plot an area of EES
			ax1.axvspan(xmin=0, xmax=l_poly_border, color='g', alpha=0.3, label="EES area")
			# plot EES
			ax1.axvline(x=ees_index, color='orange', linewidth=3, label="EES")
			# plot an area where we try to find a best latency
			ax1.axvspan(xmin=l_best_border, xmax=r_best_border, color='#175B99', alpha=0.3, label="best latency area")
			# plot original bio data per slice
			ax1.plot(splitted_per_slice_original[slice_index], linewidth=0.7)
			# plot Q1 and Q3 areas, and median
			ax1.plot(smoothed_Q1, color='k', linewidth=3.5, label="Q1/Q3 values")
			ax1.plot(smoothed_Q3, color='k', linewidth=3.5)
			ax1.plot(median, linestyle='--', color='k', label="median value")
			# plot extremuma
			ax1.plot(e_all_Q1_minima_indexes, e_all_Q1_minima_values, '.', color=min_color, label="minima extremuma")
			ax1.plot(e_all_Q1_maxima_indexes, e_all_Q1_maxima_values, '.', color=max_color, label="maxima extremuma")
			ax1.plot(e_all_Q3_minima_indexes, e_all_Q3_minima_values, '.', color=min_color)
			ax1.plot(e_all_Q3_maxima_indexes, e_all_Q3_maxima_values, '.', color=max_color)
			# plot latencies for Q1 and Q3
			ax1.plot(latencies_Q1, smoothed_Q1[latencies_Q1], '.', markersize=20, color='#227734', label="Q1 latencies")
			ax1.plot(latencies_Q3, smoothed_Q3[latencies_Q3], '.', markersize=20, color="#FF6600", label="Q3 latencies")
			# plot the best latency with guidline
			best_lat_x = best_latency[0]
			best_lat_y = smoothed_Q1[best_lat_x]
			ax1.plot([best_lat_x] * 2, [best_lat_y - 2, best_lat_y], color="k", linewidth=0.5)
			ax1.text(best_lat_x, best_lat_y - 2, best_lat_x * bio_step)
			ax1.set_xticks(range(0, 101, 4))
			ax1.set_xticklabels((np.arange(0, 101, 4) * bio_step).astype(int))
			ax1.set_xlim(0, 100)
			ax1.grid(axis='x', linestyle='--')
			ax1.legend()

			ax2.title.set_text("Delta of variance per iter")
			# plot an area where we try to find a best latency
			ax2.axvspan(xmin=l_best_border, xmax=r_best_border, color='#175B99', alpha=0.3, label="best latency area")
			# plot not filtered extremuma
			ax2.plot(merged_indexes + l_poly_border, merged_values + 0.2, '.', markersize=7, color='k')
			# plot bars (delta of variance) and colorize extremuma
			colors = ['#2B2B2B'] * len(delta_data)
			for i in e_delta_minima_indexes + l_poly_border:
				colors[i] = 'b'
			for i in e_delta_maxima_indexes + l_poly_border:
				colors[i] = 'r'
			ax2.bar(range(len(delta_data)), delta_data, width=0.4, color=colors, zorder=3)
			ax2.set_xticks(range(0, 101, 4))
			ax2.set_xticklabels((np.arange(0, 101, 4) * bio_step).astype(int))
			ax2.set_xlim(0, 100)
			ax2.grid(axis='x', linestyle='--', zorder=0)
			ax2.legend()

			plt.tight_layout()
			plt.show()

	# show latencies on the all slices (shadows)
	if debugging:
		# original data
		plt.subplots(figsize=(16, 9))
		for slice_index, slice_data in enumerate(splitted_per_slice_original):
			y_offset = slice_index * 10
			plt.plot(np.arange(len(slice_data)) * bio_step, slice_data * 1.5 + y_offset)
		plt.xticks(range(0, slice_in_ms + 1), range(0, slice_in_ms + 1))
		plt.grid(axis='x')
		plt.xlim(0, slice_in_ms)
		plt.show()

		# original data with latency
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
		plt.xticks(range(slice_in_ms + 1), range(slice_in_ms + 1))
		plt.grid(axis='x')

		lat_x = [x * bio_step for x in global_lat_indexes]
		lat_y = [splitted_per_slice_boxplots[slice_index][:, k_median][lat] for slice_index, lat in enumerate(global_lat_indexes)]
		plt.plot(lat_x, lat_y, linewidth=3, color='g')

		# plt.xticks(range(100), [x * bio_step for x in range(100) if x % 4 == 0])
		plt.yticks(yticks, range(1, slices_number + 1))
		plt.xlim(0, slice_in_ms)

		plt.show()

	return global_lat_indexes, global_amp_values, global_peaks_numbers


def plot_pca():
	for d in read_data(f"{data_folder}/bio_15.hdf5"):
		straighten_data(d)
	raise Exception
	bio_meta = slice_metainfo(read_data(f"{data_folder}/bio_15.hdf5"), ees_hz=40)

	raise Exception

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

