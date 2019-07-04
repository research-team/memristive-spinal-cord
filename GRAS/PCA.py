import numpy as np
import pylab as plt
import h5py as hdf5
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy.signal import argrelextrema

np.set_printoptions(threshold=np.inf)

data_folder = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/bio_data/"

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

min_color = "#00FFFF"
max_color = "#ED1B24"

percents = [25, 50, 75]


def read_data(filepath):
	with hdf5.File(filepath, 'r') as file:
		data_by_test = [test_values[:] for test_values in file.values()]
	return data_by_test


def select_slices(path, begin, end, sign=1):
	return [sign * data[begin:end][::10] for data in read_data(path)]


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
	Function for calculating boxplots from array of dots
	Args:
		dots (np.ndarray): array of dots
	Returns:
		tuple: [7 elements] median, Q3 and Q1 values, highest and lowest whiskers, highest and lowest fliers
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


def normalization(data, a=0, b=1, save_centering=False):
	"""
	Normalization in [a, b] interval or with saving centering
	x` = (b - a) * (xi - min(x)) / (max(x) - min(x)) + a
	Args:
		data (np.ndarray): data for normalization
		a (float): left interval
		b (float): right interval
		save_centering (bool): if True -- will save data centering and just normalize by lowest data
	Returns:
		np.ndarray: normalized data
	"""
	# checking on errors
	if a >= b:
		raise Exception("Left interval 'a' must be fewer than right interval 'b'")
	if save_centering:
		minimal = abs(min(data))
		return [volt / minimal for volt in data]
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
		array (np.ndarray): data array
		condition (np.ufunc): e.g. np.less (<), np.great_equal (>=) and etc.
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


def draw_vector(p0, p1, color=None):
	"""
	Small function for drawing vector with arrow by two points
	Args:
		p0 (np.ndarray): begin of the vector
		p1 (np.ndarray): end of the vector
		color (str): the color of vector
	"""
	ax = plt.gca()
	# this plot is fixing the problem of hiding annotations because of their not markers origin
	if color:
		ax.plot(p1[0], p1[1], '.', color=color)
	else:
		ax.plot(p1[0], p1[1], '.')
	ax.annotate('', p1, p0, arrowprops=dict(facecolor=color, linewidth=1.0))


def center_data_by_line(y_points, debugging=False):
	"""
	Straight the data and center the rotated points cloud at (0, 0)
	Args:
		y_points (list or np.ndarray): list of Y points value
		debugging (bool): True -- will print debugging info and plot figures
	Returns:
		np.ndarray: new straighten Y data
	"""
	X = 0
	Y = 1
	# prepare original data (convert to 2D ndarray)
	dots_2D = np.stack((range(len(y_points)), y_points), axis=1)
	# calc PCA for dots cloud
	pca = PCA(n_components=2)
	pca.fit(dots_2D)
	# get PCA components for finding an rotation angle
	cos_theta = pca.components_[0, 0]
	sin_theta = pca.components_[0, 1]
	# one possible value of Theta that lies in [0, pi]
	arccos = np.arccos(cos_theta)

	# if arccos is in Q1 (quadrant), rotate CLOCKwise by arccos
	if cos_theta > 0 and sin_theta > 0:
		arccos *= -1
	# elif arccos is in Q2, rotate COUNTERclockwise by the complement of theta
	elif cos_theta < 0 and sin_theta > 0:
		arccos = np.pi - arccos
	# elif arccos is in Q3, rotate CLOCKwise by the complement of theta
	elif cos_theta < 0 and sin_theta < 0:
		arccos = -(np.pi - arccos)
	# if arccos is in Q4, rotate COUNTERclockwise by theta, i.e. do nothing
	elif cos_theta > 0 and sin_theta < 0:
		pass

	# manually build the counter-clockwise rotation matrix
	rotation_matrix = np.array([[np.cos(arccos), -np.sin(arccos)],
	                            [np.sin(arccos), np.cos(arccos)]])
	# apply rotation to each row of 'array_dots' (@ is a matrix multiplication)
	rotated_dots_2D = (rotation_matrix @ dots_2D.T).T
	# center the rotated point cloud at (0, 0)
	rotated_dots_2D -= rotated_dots_2D.mean(axis=0)

	# plot debugging figures
	if debugging:
		plt.figure(figsize=(16, 9))
		plt.suptitle("PCA")
		# plot all dots and connect them
		plt.scatter(dots_2D[:, X], dots_2D[:, Y], alpha=0.2)
		plt.plot(dots_2D[:, X], dots_2D[:, Y])
		# plot vectors
		for v_length, vector in zip(pca.explained_variance_, pca.components_):
			v = vector * 3 * np.sqrt(v_length)
			draw_vector(pca.mean_, pca.mean_ + v)
		# figure properties
		plt.tight_layout()
		plt.show()
		plt.close()

		plt.figure(figsize=(16, 9))
		plt.suptitle("Centered data")
		# plot ogignal data on centered
		plt.plot(range(len(dots_2D)), dots_2D[:, Y], label='original')
		plt.plot(range(len(rotated_dots_2D)), rotated_dots_2D[:, Y], label='centered')
		plt.axhline(y=0, color='g', linestyle='--')
		# figure properties
		plt.tight_layout()
		plt.legend()
		plt.show()

	return rotated_dots_2D[:, 1]


def find_min_deltas(array, extremuma):
	"""
	Function for finding minimal scatter between two extremuma
	Args:
		array (list or np.ndarray): Y values
		extremuma (list or np.ndarray): indexes of extremuma
	Returns:
		list: indexes of minimal scatters
	"""
	min_scatters = []

	for e_left, e_right in zip(extremuma, extremuma[1:]):
		# if dots are too close (no place for minimal scatter) -- ignore them
		if e_right - e_left == 1:
			continue
		# elif distance is enough to place one dot
		elif e_right - e_left == 2:
			min_scatter_index = e_left + 1
		# else find indexes of minimal variance index in [dot left, dot right) interval
		else:
			e_left += 1
			e_right -= 1
			min_scatter_index = e_left + min_at(array[e_left:e_right])[k_index]
		min_scatters.append(min_scatter_index)

	return min_scatters


def merge_extremuma_arrays(minima_indexes, minima_values, maxima_indexes, maxima_values):
	"""

	Args:
		minima_indexes (np.ndarray):
		minima_values (np.ndarray):
		maxima_indexes (np.ndarray):
		maxima_values (np.ndarray):
	Returns:
		np.ndarray:
		np.ndarray:
		np.ndarray:
	"""
	# prepare data for concatenating dots into one list (per parameter)
	common_lenght = len(minima_indexes) + len(maxima_indexes)
	merged_names = [None] * common_lenght
	merged_indexes = [None] * common_lenght
	merged_values = [None] * common_lenght

	# who located earlier -- max or min
	min_starts = 0 if minima_indexes[0] < maxima_indexes[0] else 1
	max_starts = 1 if minima_indexes[0] < maxima_indexes[0] else 0

	# fill minima lists based on the precedence
	merged_names[min_starts::2] = ['min'] * len(minima_indexes)
	merged_indexes[min_starts::2] = minima_indexes
	merged_values[min_starts::2] = minima_values
	# the same for the maxima
	merged_names[max_starts::2] = ['max'] * len(maxima_indexes)
	merged_indexes[max_starts::2] = maxima_indexes
	merged_values[max_starts::2] = maxima_values
	# convert them to the array for usability
	merged_names = np.array(merged_names)
	merged_values = np.array(merged_values)
	merged_indexes = np.array(merged_indexes).astype(int)

	return merged_names, merged_indexes, merged_values


def filter_extremuma(merged_names, merged_indexes, merged_values, allowed_diff):
	"""
	Filtering extremuma by value (allowed_diff)
	Args:
		merged_names (np.ndarray): array of merged extremuma names
		merged_indexes (np.ndarray): array of merged extremuma indexes (X)
		merged_values (np.ndarray): array of merged extremuma values (Y)
		allowed_diff (float): remove arrays value which lower than that diff
	Returns:
		np.ndarray: filtered names
		np.ndarray: filtered indexes
		np.ndarray: filtered values
	"""
	# find good extremuma from the end
	filtered_mask_indexes = []
	i = len(merged_names) - 1
	next_i = len(merged_names) - 1

	while i >= 0 and next_i >= 0:
		next_i = i - 1
		while next_i >= 0:
			if abs(merged_values[i] - merged_values[next_i]) > allowed_diff:
				filtered_mask_indexes.append(next_i)
				i = next_i
				break
			next_i -= 1

	# reverse mask to ascending view
	filtered_mask_indexes = filtered_mask_indexes[::-1]

	# mask = (differed_indexes > 1) & (differed_values > allowed_diff)
	e_poly_Q1_names = merged_names[filtered_mask_indexes]
	e_poly_Q1_indexes = merged_indexes[filtered_mask_indexes]
	e_poly_Q1_values = merged_values[filtered_mask_indexes]

	return e_poly_Q1_names, e_poly_Q1_indexes, e_poly_Q1_values


def get_lat_amp(data_test_runs, ees_hz, data_step, debugging=False):
	"""
	Function for finding latencies at each slice in normalized (!) data
	Args:
		data_test_runs (list or np.ndarry): arrays of data
		ees_hz (int): EES value
		data_step (float): data step
		debugging (bool): True -- will print debugging info and plot figures
	Returns:
		list: latencies indexes
		list: amplitudes values
	"""
	global_lat_indexes = []
	global_amp_values = []

	data_test_runs = np.array(data_test_runs)

	# additional properties
	slice_in_ms = int(1000 / ees_hz)
	# set ees area, before that time we don't try to find latencies
	ees_zone_time = int(7 / data_step)
	shared_x = np.arange(slice_in_ms / data_step) * data_step
	# get number of slices based on length of the bio data
	slices_number = int(len(data_test_runs[0]) / (slice_in_ms / data_step))
	# split original data only for visualization
	splitted_per_slice_original = np.split(data_test_runs.T, slices_number)
	# calc boxplot per step and split it by number of slices
	boxplots_per_iter = np.array([calc_boxplots(dot) for dot in data_test_runs.T])
	splitted_per_slice_boxplots = np.split(boxplots_per_iter, slices_number)

	delta_poly = np.abs(boxplots_per_iter[:, k_fliers_low] - boxplots_per_iter[:, k_fliers_high])[ees_zone_time:]
	delta_poly_diff = np.abs(np.diff(delta_poly, n=1))

	Q1, median, Q3 = np.percentile(delta_poly_diff, percents)

	allowed_diff_for_extremuma = median
	allowed_diff_for_variance_delta = Q3

	# compute latency per slice
	for slice_index, slice_boxplot_data in enumerate(splitted_per_slice_boxplots):
		"""[1] prepare data"""
		# get data by keys
		data_Q1 = slice_boxplot_data[:, k_fliers_low]
		data_Q3 = slice_boxplot_data[:, k_fliers_high]
		median = slice_boxplot_data[:, k_median]
		# smooth a little the data to avoid micropeaks
		smoothed_Q1 = smooth(data_Q1, 2)
		smoothed_Q3 = smooth(data_Q3, 2)
		smoothed_median = smooth(median, 2)
		# fix the first and last ignored data after smoothing (found by experimental way)
		smoothed_Q1[:2] = data_Q1[:2]
		smoothed_Q1[-2:] = data_Q1[-2:]
		smoothed_Q3[:2] = data_Q3[:2]
		smoothed_Q3[-2:] = data_Q3[-2:]
		smoothed_median[:2] = median[:2]
		smoothed_median[-2:] = median[-2:]
		# get an absolute delta of smoothed data
		delta_smoothed_data = np.abs(smoothed_Q1 - smoothed_Q3)

		"""[2] find all extremuma"""
		# find all Q1 extremuma indexes and values
		e_all_Q1_minima_indexes, e_all_Q1_minima_values = find_extremuma(smoothed_Q1, np.less_equal)
		e_all_Q1_maxima_indexes, e_all_Q1_maxima_values = find_extremuma(smoothed_Q1, np.greater_equal)
		# find all Q3 extremuma indexes and values
		e_all_Q3_minima_indexes, e_all_Q3_minima_values = find_extremuma(smoothed_Q3, np.less_equal)
		e_all_Q3_maxima_indexes, e_all_Q3_maxima_values = find_extremuma(smoothed_Q3, np.greater_equal)

		'''[3] find mono answer'''
		# get the minima extremuma of mono answers (Q1 is lower than Q3, so take a Q1 data)
		e_mono_Q1_minima_indexes = e_all_Q1_minima_indexes[e_all_Q1_minima_indexes < ees_zone_time]
		# find an index of the biggest delta between first dot (median[0]) and mono extremuma
		max_delta_Q1_index, _ = max_at(np.abs(smoothed_Q1[e_mono_Q1_minima_indexes] - median[0]))
		# get index of extremuma with the biggest delta
		mono_answer_index = e_mono_Q1_minima_indexes[max_delta_Q1_index]

		"""[4] find a poly area"""
		l_poly_border = None
		# correct this border by the end of the mono answer peak
		for e_index_Q1, e_index_Q3 in zip(e_all_Q1_maxima_indexes, e_all_Q3_maxima_indexes):
			mono_answer_end_index = e_index_Q1 if e_index_Q1 > e_index_Q3 else e_index_Q3
			if mono_answer_end_index > ees_zone_time:
				l_poly_border = mono_answer_end_index
				break
		if l_poly_border > int(10 / data_step):
			l_poly_border = int(10 / data_step)

		"""[5] find extremuma of the poly area"""
		# get poly Q1 minima extremuma indexes and values
		e_poly_Q1_minima_indexes = e_all_Q1_minima_indexes[e_all_Q1_minima_indexes > l_poly_border]
		e_poly_Q1_minima_values = e_all_Q1_minima_values[e_all_Q1_minima_indexes > l_poly_border]
		# get poly Q1 maxima extremuma indexes and values
		e_poly_Q1_maxima_indexes = e_all_Q1_maxima_indexes[e_all_Q1_maxima_indexes > l_poly_border]
		e_poly_Q1_maxima_values = e_all_Q1_maxima_values[e_all_Q1_maxima_indexes > l_poly_border]
		# get poly Q3 minima extremuma indexes and values
		e_poly_Q3_minima_indexes = e_all_Q3_minima_indexes[e_all_Q3_minima_indexes > l_poly_border]
		e_poly_Q3_minima_values = e_all_Q3_minima_values[e_all_Q3_minima_indexes > l_poly_border]
		# get poly Q3 maxima extremuma indexes and values
		e_poly_Q3_maxima_indexes = e_all_Q3_maxima_indexes[e_all_Q3_maxima_indexes > l_poly_border]
		e_poly_Q3_maxima_values = e_all_Q3_maxima_values[e_all_Q3_maxima_indexes > l_poly_border]

		# merge Q1 poly extremuma indexes
		e_poly_Q1_names, e_poly_Q1_indexes, e_poly_Q1_values = merge_extremuma_arrays(e_poly_Q1_minima_indexes,
		                                                                              e_poly_Q1_minima_values,
		                                                                              e_poly_Q1_maxima_indexes,
		                                                                              e_poly_Q1_maxima_values)
		# filtering Q1 poly extremuma: remove micropeaks
		e_poly_Q1_names, e_poly_Q1_indexes, e_poly_Q1_values = filter_extremuma(e_poly_Q1_names,
		                                                                        e_poly_Q1_indexes,
		                                                                        e_poly_Q1_values,
		                                                                        allowed_diff=allowed_diff_for_extremuma)
		# find minimal deltas between pairs of extremuma
		min_deltas_Q1 = find_min_deltas(delta_smoothed_data, extremuma=e_poly_Q1_indexes)

		# merge Q3 poly extremuma indexes
		e_poly_Q3_names, e_poly_Q3_indexes, e_poly_Q3_values = merge_extremuma_arrays(e_poly_Q3_minima_indexes,
		                                                                              e_poly_Q3_minima_values,
		                                                                              e_poly_Q3_maxima_indexes,
		                                                                              e_poly_Q3_maxima_values)
		# filtering Q3 poly extremuma: remove micropeaks
		e_poly_Q3_names, e_poly_Q3_indexes, e_poly_Q3_values = filter_extremuma(e_poly_Q3_names,
		                                                                        e_poly_Q3_indexes,
		                                                                        e_poly_Q3_values,
		                                                                        allowed_diff=allowed_diff_for_extremuma)
		# find minimal deltas between pairs of extremuma
		min_deltas_Q3 = find_min_deltas(delta_smoothed_data, extremuma=e_poly_Q3_indexes)

		"""[6] find a best searching borders for latencies by delta"""
		# find extremuma for deltas of Q1 and Q3
		e_delta_minima_indexes, e_delta_minima_values = find_extremuma(delta_smoothed_data, np.less_equal)
		e_delta_maxima_indexes, e_delta_maxima_values = find_extremuma(delta_smoothed_data, np.greater_equal)

		# get only poly area
		e_delta_minima_values = e_delta_minima_values[e_delta_minima_indexes > l_poly_border]
		e_delta_minima_indexes = e_delta_minima_indexes[e_delta_minima_indexes > l_poly_border]

		e_delta_maxima_values = e_delta_maxima_values[e_delta_maxima_indexes > l_poly_border]
		e_delta_maxima_indexes = e_delta_maxima_indexes[e_delta_maxima_indexes > l_poly_border]

		# merge Q3 poly extremuma indexes
		merged_names, merged_indexes, merged_values = merge_extremuma_arrays(e_delta_minima_indexes,
		                                                                     e_delta_minima_values,
		                                                                     e_delta_maxima_indexes,
		                                                                     e_delta_maxima_values)

		# filter dots that are too close or have not enough differentiation in values
		mask = []
		for left, right in zip(range(len(merged_indexes)), range(len(merged_indexes))[1:]):
			is_index_ok = abs(merged_indexes[left] - merged_indexes[right]) > 1
			is_value_ok = abs(merged_values[left] - merged_values[right]) > allowed_diff_for_variance_delta

			if is_index_ok or is_value_ok:
				mask += [left, right]
		mask = [i in mask for i in range(len(merged_indexes))]

		# use filter
		e_merged_names = merged_names[mask]
		e_merged_indexes = merged_indexes[mask]
		e_merged_values = merged_values[mask]

		# calc peak that we recognie as good poly answer, not noise
		diff = np.abs(np.diff(merged_values, n=1))
		allowed_diff_peak = np.mean(diff)

		# the last filter for finding the best max min pair
		index = 0
		mask = []
		while index < len(e_merged_indexes):
			if e_merged_names[index] == 'min':
				mask += [True]
				index_right = index + 1
				while index_right < len(e_merged_indexes):
					if abs(e_merged_values[index] - e_merged_values[index_right]) > allowed_diff_peak:
						mask += [True]
						break
					else:
						mask += [False]
					index_right += 1
				index = index_right
			else:
				mask += [False]
			index += 1
		# use filter
		e_delta_names = e_merged_names[mask]
		e_delta_indexes = e_merged_indexes[mask]

		# calc best poly right border
		r_best_border = e_delta_indexes[e_delta_names == 'max'][0]
		# calc best poly left border
		min_indexes = e_merged_indexes[e_merged_names == 'min']
		l_best_border = min_indexes[min_indexes < r_best_border][-1]

		"""[7] find the best latency in borders"""
		best_latency = (None, -np.inf)
		# check on border length -- if small enough then find an max of extremuma delta dots
		if (r_best_border - l_best_border) * data_step < 3:
			# find the latency in Q1 and Q3 data (based on the maximal variance)
			for lat_Q1 in filter(lambda dot: l_best_border <= dot <= r_best_border, min_deltas_Q1):
				if delta_smoothed_data[lat_Q1] > best_latency[k_value]:
					best_latency = (lat_Q1, delta_smoothed_data[lat_Q1])
			if best_latency[k_index] is None:
				best_latency = (l_best_border, delta_smoothed_data[l_best_border])
		# else use a dichotomy to split a big interval and find the best one
		else:
			# set an initial borders
			left = l_best_border
			right = r_best_border
			borders_length = (right - left) // 2
			# dichotomy loop
			while borders_length > 5: # > 1.25 ms
				# first part
				a1 = left
				a2 = left + borders_length
				# second part
				b1 = right - borders_length
				b2 = right
				# get the difference in this areas
				diff_a = np.sum(np.abs(np.diff(delta_smoothed_data[a1:a2], n=1)))
				diff_b = np.sum(np.abs(np.diff(delta_smoothed_data[b1:b2], n=1)))
				# set the new border based on highest difference
				left = b1 if diff_b > diff_a else a1
				right = b2 if diff_b > diff_a else a2
				# split on two parts
				borders_length = (right - left) // 2

			# find the inflection
			ys = delta_smoothed_data[left:right]
			gradient_y = np.gradient(ys)
			max_index_gradient = np.argmax(gradient_y)
			# set the latency
			lat = left + max_index_gradient - 1
			best_latency = (lat, delta_smoothed_data[lat])

		# append found latency to the global list of the all latencies per slice
		latency_index = best_latency[k_index]
		global_lat_indexes.append(latency_index)

		"""[7] find amplitudes"""
		e_ampQ1_after_latency = np.abs(e_poly_Q1_values[e_poly_Q1_indexes > latency_index])
		e_ampQ3_after_latency = np.abs(e_poly_Q3_values[e_poly_Q3_indexes > latency_index])
		amp_sum = np.sum(e_ampQ1_after_latency) + np.sum(e_ampQ3_after_latency)

		global_amp_values.append(amp_sum)

		# all debugging info
		if debugging:
			print("\n")
			print("- " * 20)
			print("{:^40}".format(f"SLICE {slice_index + 1}"))
			print("- " * 20)
			print(f"  EES was found at: {mono_answer_index * data_step}ms (index {mono_answer_index})")
			print(f" minima indexes Q1: {e_poly_Q1_minima_indexes}")
			print(f" maxima indexes Q1: {e_poly_Q1_maxima_indexes}")
			print(f" merged indexes Q1: {e_poly_Q1_indexes}")
			print(f"found latencies Q1: {min_deltas_Q1}")
			print("- " * 20)
			print(f" minima indexes Q3: {e_poly_Q3_minima_indexes}")
			print(f" maxima indexes Q3: {e_poly_Q3_maxima_indexes}")
			print(f" merged indexes Q3: {e_poly_Q3_indexes}")
			print(f"found latencies Q3: {min_deltas_Q3}")
			print("- " * 20)
			print(f" poly answers area: [{l_poly_border * data_step}, {slice_in_ms}]ms")
			print(f"  hist indexes min: {e_delta_minima_indexes}")
			print(f"  hist indexes max: {e_delta_maxima_indexes}")
			print(f"      merged names: {merged_names}")
			print(f"    merged indexes: {merged_indexes}")
			print(f"     merged values: {merged_values}")
			print(f" best area between: {l_best_border * data_step}ms and {r_best_border * data_step}ms")
			print(f"        latency at: {best_latency[0] * data_step}ms")

			plt.figure(figsize=(16, 9))
			gridsize = (3, 1)
			ax1 = plt.subplot2grid(shape=gridsize, loc=(0, 0), rowspan=2)
			ax2 = plt.subplot2grid(shape=gridsize, loc=(2, 0), sharex=ax1)

			ax1.title.set_text(f"Slice #{slice_index + 1}/{slices_number}")
			# plot an area of EES
			ax1.axvspan(xmin=0, xmax=l_poly_border, color='g', alpha=0.3, label="EES area")
			# plot EES
			ax1.axvline(x=mono_answer_index, color='orange', linewidth=3, label="EES")
			# plot zero line
			ax1.axhline(y=0, color='k', linewidth=1, linestyle='dotted')
			# plot an area where we try to find a best latency
			ax1.axvspan(xmin=l_best_border, xmax=r_best_border, color='#175B99', alpha=0.3, label="best latency area")
			# plot original bio data per slice
			ax1.plot(splitted_per_slice_original[slice_index], linewidth=0.7)
			# plot latencies for Q1 and Q3
			ax1.plot(min_deltas_Q1, smoothed_Q1[min_deltas_Q1], '.', markersize=20, color='#227734', label="Q1 latencies")
			ax1.plot(min_deltas_Q3, smoothed_Q3[min_deltas_Q3], '.', markersize=20, color="#FF6600", label="Q3 latencies")
			# plot Q1 and Q3 areas, and median
			ax1.plot(smoothed_Q1, color='k', linewidth=3.5, label="Q1/Q3 values")
			ax1.plot(smoothed_Q3, color='k', linewidth=3.5)
			ax1.plot(median, linestyle='--', color='grey', label="median value")
			# plot extremuma
			ax1.plot(e_all_Q1_minima_indexes, e_all_Q1_minima_values, '.', markersize=3, color=min_color, label="minima MONO extremuma")
			ax1.plot(e_all_Q1_maxima_indexes, e_all_Q1_maxima_values, '.', markersize=3, color=max_color, label="maxima MONO extremuma")
			ax1.plot(e_all_Q3_minima_indexes, e_all_Q3_minima_values, '.', markersize=3, color=min_color)
			ax1.plot(e_all_Q3_maxima_indexes, e_all_Q3_maxima_values, '.', markersize=3, color=max_color)

			ax1.plot(e_poly_Q1_indexes[e_poly_Q1_names == 'min'], e_poly_Q1_values[e_poly_Q1_names == 'min'], '.',
			         markersize=10, color=min_color, label="minima POLY extremuma")
			ax1.plot(e_poly_Q1_indexes[e_poly_Q1_names == 'max'], e_poly_Q1_values[e_poly_Q1_names == 'max'], '.',
			         markersize=10, color=max_color, label="maxima POLY extremuma")
			ax1.plot(e_poly_Q3_indexes[e_poly_Q3_names == 'min'], e_poly_Q3_values[e_poly_Q3_names == 'min'], '.',
			         markersize=10, color=min_color)
			ax1.plot(e_poly_Q3_indexes[e_poly_Q3_names == 'max'], e_poly_Q3_values[e_poly_Q3_names == 'max'], '.',
			         markersize=10, color=max_color)

			# plot the best latency with guidline
			best_lat_x = best_latency[0]
			best_lat_y = smoothed_Q1[best_lat_x]
			ax1.plot([best_lat_x] * 2, [best_lat_y - 2, best_lat_y], color="k", linewidth=0.5)
			ax1.text(best_lat_x, best_lat_y - 2, best_lat_x * data_step)
			ax1.set_xticks(range(0, 101, 4))
			ax1.set_xticklabels((np.arange(0, int(slice_in_ms / data_step), int(1 / data_step)) * data_step).astype(int))
			ax1.set_xlim(0, int(slice_in_ms / data_step))
			ax1.grid(axis='x', linestyle='--')
			ax1.legend()

			ax2.title.set_text(f"Delta of variance per iter (allowed peak : {allowed_diff_peak:.4f})")
			# plot an area of EES
			ax2.axvspan(xmin=0, xmax=l_poly_border, color='g', alpha=0.3, label="EES area")
			# plot an area where we try to find a best latency
			ax2.axvspan(xmin=l_best_border, xmax=r_best_border, color='#175B99', alpha=0.3, label="best latency area")
			# plot bars (delta of variance) and colorize extremuma
			colors = ['#2B2B2B'] * len(delta_smoothed_data)
			print(e_delta_indexes)
			print(e_delta_names)
			for i in e_delta_indexes[e_delta_names == 'min']:
				colors[i] = 'b'
			for i in e_delta_indexes[e_delta_names == 'max']:
				colors[i] = 'r'
			ax2.plot(merged_indexes, merged_values + 0.02, '.')
			ax2.bar(range(len(delta_smoothed_data)), delta_smoothed_data, width=0.4, color=colors, zorder=3)
			ax2.set_xticks(range(0, int(slice_in_ms / data_step) + 1, int(1 / data_step)))
			ax2.set_xticklabels((np.arange(0, int(slice_in_ms / data_step) + 1, int(1 / data_step)) * data_step).astype(int))
			ax2.set_xlim(0, int(slice_in_ms / data_step))
			ax2.grid(axis='x', linestyle='--', zorder=0)
			ax2.legend()

			plt.tight_layout()
			plt.show()

	if debugging:
		# original data
		plt.subplots(figsize=(16, 9))
		for slice_index, slice_boxplot_data in enumerate(splitted_per_slice_original):
			y_offset = slice_index * 1
			plt.plot(np.arange(len(slice_boxplot_data)) * data_step, slice_boxplot_data + y_offset)
		plt.xticks(range(0, slice_in_ms + 1), range(0, slice_in_ms + 1))
		plt.grid(axis='x')
		plt.xlim(0, slice_in_ms)
		plt.show()

		# original data with latency
		plt.subplots(figsize=(16, 9))
		yticks = []
		y_offset = 1
		for slice_index, data in enumerate(splitted_per_slice_boxplots):
			data += slice_index * y_offset  # is a link (!)
			plt.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color='r', alpha=0.3, label="flier")
			plt.fill_between(shared_x, data[:, k_whiskers_high], data[:, k_whiskers_low], color='r', alpha=0.5, label="whisker")
			plt.fill_between(shared_x, data[:, k_box_Q3], data[:, k_box_Q1], color='r', alpha=0.7, label="box")
			plt.plot(shared_x, data[:, k_median], linestyle='--', color='k')
			yticks.append(data[:, k_median][0])
		plt.xticks(range(slice_in_ms + 1), range(slice_in_ms + 1))
		plt.grid(axis='x')

		lat_x = [x * data_step for x in global_lat_indexes]
		lat_y = [splitted_per_slice_boxplots[slice_index][:, k_median][lat] for slice_index, lat in enumerate(global_lat_indexes)]
		plt.plot(lat_x, lat_y, linewidth=3, color='g')

		# plt.xticks(range(100), [x * bio_step for x in range(100) if x % 4 == 0])
		plt.yticks(yticks, range(1, slices_number + 1))
		plt.xlim(0, slice_in_ms)

		plt.show()

	# ToDo replace a temporary 'global_amp_values'
	global_lat_indexes = np.array(global_lat_indexes) * data_step

	return global_lat_indexes, global_amp_values


def prepare_data(dataset):
	"""
	Centering -> Normalizing -> returning
	Args:
		dataset (list or np.ndarray): original dataset
	Returns:
		list: prepared dataset per test
	"""
	prepared_data = []
	for data_per_test in dataset:
		centered_data = center_data_by_line(data_per_test)
		normalized_data = normalization(centered_data, save_centering=True)
		prepared_data.append(normalized_data)
	return prepared_data


def plot_pca(debugging=False):
	"""
	Preparing data and drawing PCA for them
	Args:
		debugging:
	"""
	X = 0
	Y = 1
	# process bio dataset
	dataset = read_data(f"{data_folder}/bio/bio_control_E_15cms_40Hz_i100_2pedal_no5ht_T_2017-09-05.hdf5")
	lat_per_slice, amp_per_slice = get_lat_amp(prepare_data(dataset), ees_hz=40, data_step=0.25)
	bio_pack = (np.stack((amp_per_slice, lat_per_slice), axis=1), '#a6261d', 'bio')

	# process GRAS dataset
	dataset = select_slices(f"{data_folder}/gras_15.hdf5", 10000, 22000)
	lat_per_slice, amp_per_slice = get_lat_amp(prepare_data(dataset), ees_hz=40, data_step=0.25)
	gras_pack = (np.stack((amp_per_slice, lat_per_slice), axis=1), '#287a72', 'gras')

	# process NEURON dataset
	dataset = select_slices(f"{data_folder}/neuron_15.hdf5", 0, 12000, sign=-1)
	lat_per_slice, amp_per_slice = get_lat_amp(prepare_data(dataset), ees_hz=40, data_step=0.25)
	neuron_pack = (np.stack((amp_per_slice, lat_per_slice), axis=1), '#f2aa2e', 'neuron')

	# start plotting
	fig, ax = plt.subplots()

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
		sign = -1 if first_vector[X] > vertical[X] else 1
		# calculate ellipse size
		ellipse_width = length(vectors[1][1] - center) * 2
		ellipse_height = length(vectors[0][1] - center) * 2

		# plot vectors
		for vector in vectors:
			draw_vector(vector[0], vector[1], color=color)
		# plot dots
		ax.scatter(coords[:, X], coords[:, Y], color=color, label=label, s=80)
		# plot ellipse
		ellipse = Ellipse(xy=tuple(center), width=ellipse_width, height=ellipse_height, angle=sign * angle_degrees)
		ellipse.set_edgecolor(hex2rgb(color))
		ellipse.set_fill(False)
		ax.add_artist(ellipse)
		# fill convex figure
		hull = ConvexHull(coords)
		ax.fill(coords[hull.vertices, X], coords[hull.vertices, Y], color=color, alpha=0.3)
		# calc an area of the poly figure
		poly_area = poly_area_by_coords(coords[hull.vertices, X], coords[hull.vertices, Y])

		if debugging:
			for index, x, y in zip(range(len(coords[:, X])), coords[:, X], coords[:, Y]):
				ax.text(x, y, index + 1)

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
