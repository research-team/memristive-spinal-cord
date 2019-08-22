import logging
import numpy as np
import pylab as plt
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

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

min_color = "#00FFFF"
max_color = "#ED1B24"
percents = [25, 50, 75]


class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
		FancyArrowPatch.draw(self, renderer)


def form_ellipse(P):
	""" Form the ellipsoid based on all points
	Here, P is a numpy array of points:
	P = [[x1,y1,z1],
		 . . .
		 [xn,yn,zn]]
	Returns:
		np.ndarray: radii values
		np.ndarray: rotation matrix
	"""
	# get P shape information
	points_number, dimension = P.shape
	# auxiliary matrix
	u = (1 / points_number) * np.ones(points_number)
	# center of the P points
	center = P.T @ u
	# form the algebraic form of the ellipsoid by computing the (multiplicative) inverse of a matrix . . .
	A = np.linalg.inv(P.T @ (np.diag(u) @ P) - np.outer(center, center)) / dimension
	# get singular value decomposition data of the matrix A
	_, size, rotation = np.linalg.svd(A)
	radiuses = 1 / np.sqrt(size)
	return radiuses, rotation


def plot_ellipsoid(center, radii, rotation, plot_axes=False, color='b', alpha=0.2):
	"""
	Plot an ellipsoid
	Args:
		center (np.ndarray): center of the ellipsoid
		radii (np.ndarray): radius per axis
		rotation (np.ndarray): rotation matrix
		ax (axes): current axes of the figure
		plot_axes (bool): plot the axis of ellipsoid if need
		color (str): color in matlab forms (hex, name of color, first char of color)
		alpha (float): opacity value
	"""
	ax = plt.gca()

	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	# cartesian coordinates that correspond to the spherical angles
	x = radii[0] * np.outer(np.cos(u), np.sin(v))
	y = radii[1] * np.outer(np.sin(u), np.sin(v))
	z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
	# rotate accordingly
	for i in range(len(x)):
		for j in range(len(x)):
			x[i, j], y[i, j], z[i, j] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
	# additional visualization for debugging
	if plot_axes:
		# matrix of axes
		axes = np.array([[radii[0], 0.0, 0.0],
		                 [0.0, radii[1], 0.0],
		                 [0.0, 0.0, radii[2]]])
		# rotate accordingly
		for i in range(len(axes)):
			axes[i] = np.dot(axes[i], rotation)
		# plot axes
		for point in axes:
			X_axis = np.linspace(-point[0], point[0], 5) + center[0]
			Y_axis = np.linspace(-point[1], point[1], 5) + center[1]
			Z_axis = np.linspace(-point[2], point[2], 5) + center[2]
			ax.plot(X_axis, Y_axis, Z_axis, color='g')
	# plot ellipsoid
	stride = 4
	# print("-----------")
	# for xc in x:
	# 	print("x= ", len(xc), xc)
	# for yc in y:
	# 	print("y = ", len(yc), yc)
	# for zc in z:
	# 	print("z= ", len(zc), zc)
	# print("len(z) = ", len(z[0]))
	# print("-----------")
	ax.plot_wireframe(x, y, z, rstride=stride, cstride=stride, color=color, alpha=0.3)
	ax.plot_surface(x, y, z, rstride=stride, cstride=stride, alpha=0.1, color=color)

	return x, y, z


def hex2rgb(hex_color):
	hex_color = hex_color.lstrip('#')
	return [int("".join(gr), 16) / 256 for gr in zip(*[iter(hex_color)] * 2)]


def length(p0, p1):
	return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def unit_vector(vector):
	return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def split_by_slices(data, slice_length):
	"""
	TODO: add docstring
	Args:
		data (np.ndarray): data array
		slice_length (int): slice length in steps
	Returns:
		np.ndarray: sliced data
	"""
	slices_begin_indexes = range(0, len(data) + 1, slice_length)
	splitted_per_slice = [data[beg:beg + slice_length] for beg in slices_begin_indexes]
	print(len(splitted_per_slice))
	# remove tails
	if len(splitted_per_slice[0]) != len(splitted_per_slice[-1]):
		del splitted_per_slice[-1]
	return splitted_per_slice


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


def find_extrema(array, condition):
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

	# who located earlier -- max or min
	min_starts = 0 if minima_indexes[0] < maxima_indexes[0] else 1
	max_starts = 1 if minima_indexes[0] < maxima_indexes[0] else 0

	common_length = len(minima_indexes) + len(maxima_indexes)

	merged_names = [None] * common_length
	merged_indexes = [None] * common_length
	merged_values = [None] * common_length

	# be sure that size of [min_starts::2] be enough for filling
	if len(merged_indexes[min_starts::2]) < len(minima_indexes):
		minima_indexes = minima_indexes[:-1]
		minima_values = minima_values[:-1]

	if len(merged_indexes[min_starts::2]) > len(minima_indexes):
		minima_indexes = np.append(minima_indexes, minima_indexes[-1])
		minima_values = np.append(minima_values, minima_values[-1])

	if len(merged_indexes[max_starts::2]) < len(maxima_indexes):
		maxima_indexes = maxima_indexes[:-1]
		maxima_values = maxima_values[:-1]

	if len(merged_indexes[max_starts::2]) > len(maxima_indexes):
		maxima_indexes = np.append(maxima_indexes, maxima_indexes[-1])
		maxima_values = np.append(maxima_values, maxima_values[-1])

	# fill minima lists based on the precedence
	merged_names[min_starts::2] = ['min'] * len(minima_indexes)
	merged_indexes[min_starts::2] = minima_indexes
	merged_values[min_starts::2] = minima_values
	# the same for the maxima
	merged_names[max_starts::2] = ['max'] * len(maxima_indexes)
	merged_indexes[max_starts::2] = maxima_indexes
	merged_values[max_starts::2] = maxima_values

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


def get_lat_amp(data_test_runs, ees_hz, step_size, debugging=False):
	"""
	Function for finding latencies at each slice in normalized (!) data
	Args:
		data_test_runs (list or np.ndarry): arrays of data
		ees_hz (int): EES value
		step_size (float): data step
		debugging (bool): True -- will print debugging info and plot figures
	Returns:
		list: latencies indexes
		list: amplitudes values
	"""
	global_lat_indexes = []
	global_mono_indexes = []
	global_amp_values = []

	data_test_runs = np.array(data_test_runs, dtype=float)
	# additional properties
	slice_in_ms = 1000 / ees_hz
	slices_number = int(len(data_test_runs[0]) / (slice_in_ms / step_size))
	slice_in_steps = int(slice_in_ms / step_size)
	# set ees area, before that time we don't try to find latencies
	ees_zone_time = int(7 / step_size)
	shared_x = np.arange(slice_in_steps) * step_size
	original_data = data_test_runs.T
	# calc boxplot per dot
	boxplots_per_iter = np.array([calc_boxplots(dot) for dot in original_data])
	# split data by slices
	splitted_per_slice_boxplots = split_by_slices(boxplots_per_iter, slice_in_steps)
	splitted_per_slice_original = split_by_slices(original_data, slice_in_steps)

	# compute latency per slice
	for slice_index, slice_boxplot_data in enumerate(splitted_per_slice_boxplots):
		log.info(f"{slice_index + 1:=^20}")
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
		log.info(f"data length: {len(delta_smoothed_data)}")

		"""[2] find all extremuma"""
		# find all Q1 extremuma indexes and values
		e_all_Q1_minima_indexes, e_all_Q1_minima_values = find_extrema(smoothed_Q1, np.less_equal)
		e_all_Q1_maxima_indexes, e_all_Q1_maxima_values = find_extrema(smoothed_Q1, np.greater_equal)
		log.info(f"all Q1 minima extrema: {e_all_Q1_minima_indexes}")
		log.info(f"all Q1 maxima extrema: {e_all_Q1_maxima_indexes}")
		# find all Q3 extremuma indexes and values
		e_all_Q3_minima_indexes, e_all_Q3_minima_values = find_extrema(smoothed_Q3, np.less_equal)
		e_all_Q3_maxima_indexes, e_all_Q3_maxima_values = find_extrema(smoothed_Q3, np.greater_equal)
		log.info(f"all Q3 minima extrema: {e_all_Q3_minima_indexes}")
		log.info(f"all Q3 maxima extrema: {e_all_Q3_maxima_indexes}")

		'''[3] find mono answer'''
		# get the minima extremuma of mono answers (Q1 is lower than Q3, so take a Q1 data)
		e_mono_Q1_minima_indexes = e_all_Q1_minima_indexes[e_all_Q1_minima_indexes < ees_zone_time]
		# find an index of the biggest delta between first dot (median[0]) and mono extremuma

		if len(e_mono_Q1_minima_indexes) == 0:
			global_amp_values.append(0)
			global_lat_indexes.append(len(data_Q1) - 1)
			global_mono_indexes.append(3)
			continue

		max_delta_Q1_index = max_at(np.abs(smoothed_Q1[e_mono_Q1_minima_indexes] - median[0]))[0]
		# get index of extremuma with the biggest delta
		mono_answer_index = e_mono_Q1_minima_indexes[max_delta_Q1_index]
		log.info(f"Mono answer: {mono_answer_index * step_size}ms")

		"""[4] find a poly area"""
		l_poly_border = ees_zone_time
		# correct this border by the end of the mono answer peak
		for e_index_Q1, e_index_Q3 in zip(e_all_Q1_maxima_indexes, e_all_Q3_maxima_indexes):
			mono_answer_end_index = e_index_Q1 if e_index_Q1 > e_index_Q3 else e_index_Q3
			if mono_answer_end_index > ees_zone_time:
				l_poly_border = mono_answer_end_index
				break
		if l_poly_border > int(10 / step_size):
			l_poly_border = int(10 / step_size)

		global_mono_indexes.append(l_poly_border)

		log.info(f"Poly area: {l_poly_border * step_size} - {slice_in_ms}")

		"""[5] find a latency"""
		delta_poly_diff = np.abs(np.diff(delta_smoothed_data[l_poly_border:], n=1))

		if not len(delta_poly_diff):
			global_amp_values.append(0)
			global_lat_indexes.append(len(data_Q1) - 1)
			global_mono_indexes.append(mono_answer_index)
			continue

		diff_Q1, diff_median, diff_Q3 = np.percentile(delta_poly_diff, percents)
		allowed_diff_for_extremuma = diff_median

		gradient = np.gradient(delta_smoothed_data)
		poly_gradient = gradient[l_poly_border:]

		poly_gradient_diff = np.abs(np.diff(poly_gradient, n=1))
		poly_gradient_diff = np.append(poly_gradient_diff, poly_gradient_diff[-1])
		Q1, med, Q3 = np.percentile(poly_gradient_diff, percents)

		if all(delta_poly_diff < 0.012):
			global_amp_values.append(0)
			global_lat_indexes.append(len(data_Q1) - 1)
			global_mono_indexes.append(mono_answer_index)
			continue

		if not len(poly_gradient[poly_gradient_diff > Q3]):
			global_amp_values.append(0)
			global_lat_indexes.append(len(data_Q1) - 1)
			global_mono_indexes.append(mono_answer_index)
			continue

		gradient_Q1, gradient_med, gradient_Q3 = np.percentile(poly_gradient[poly_gradient_diff > Q3], percents)

		# fine the index of the cross between gradient and Q3
		for index, diff in enumerate(gradient[l_poly_border:], l_poly_border):
			if diff > gradient_Q3:
				latency_index = index
				break
		# if not found -- the last index
		else:
			latency_index = len(gradient) - 1
		log.info(f"Latency: {latency_index * step_size}")

		# append found latency to the global list of the all latencies per slice
		global_lat_indexes.append(latency_index)

		"""[6] find extremuma of the poly area"""
		# get poly Q1 minima extremuma indexes and values
		e_poly_Q1_minima_indexes = e_all_Q1_minima_indexes[e_all_Q1_minima_indexes > l_poly_border]
		e_poly_Q1_minima_values = e_all_Q1_minima_values[e_all_Q1_minima_indexes > l_poly_border]
		# get poly Q1 maxima extremuma indexes and values
		e_poly_Q1_maxima_indexes = e_all_Q1_maxima_indexes[e_all_Q1_maxima_indexes > l_poly_border]
		e_poly_Q1_maxima_values = e_all_Q1_maxima_values[e_all_Q1_maxima_indexes > l_poly_border]
		log.info(f"poly Q1 minima extrema: {e_poly_Q1_minima_indexes}")
		log.info(f"poly Q1 maxima extrema: {e_poly_Q1_maxima_indexes}")
		# get poly Q3 minima extremuma indexes and values
		e_poly_Q3_minima_indexes = e_all_Q3_minima_indexes[e_all_Q3_minima_indexes > l_poly_border]
		e_poly_Q3_minima_values = e_all_Q3_minima_values[e_all_Q3_minima_indexes > l_poly_border]
		# get poly Q3 maxima extremuma indexes and values
		e_poly_Q3_maxima_indexes = e_all_Q3_maxima_indexes[e_all_Q3_maxima_indexes > l_poly_border]
		e_poly_Q3_maxima_values = e_all_Q3_maxima_values[e_all_Q3_maxima_indexes > l_poly_border]
		log.info(f"poly Q3 minima extrema: {e_poly_Q3_minima_indexes}")
		log.info(f"poly Q3 maxima extrema: {e_poly_Q3_maxima_indexes}")

		"""[7] find amplitudes"""
		amp_sum = 0

		Q1_is_ok = len(e_poly_Q1_minima_indexes) and len(e_poly_Q1_maxima_indexes)
		Q3_is_ok = len(e_poly_Q3_minima_indexes) and len(e_poly_Q3_maxima_indexes)

		if Q1_is_ok:
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
			amp_sum += sum(np.abs(e_poly_Q1_values[e_poly_Q1_indexes > latency_index]))

		if Q3_is_ok:
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
			amp_sum += sum(np.abs(e_poly_Q3_values[e_poly_Q3_indexes > latency_index]))

		log.info(f"Amplitude sum: {amp_sum}")
		# append sum of amplitudes to the global list of the all amplitudes per slice
		global_amp_values.append(amp_sum)

		# all debugging info
		if debugging:
			plt.figure(figsize=(16, 9))
			gridsize = (3, 1)
			ax1 = plt.subplot2grid(shape=gridsize, loc=(0, 0), rowspan=1)
			ax2 = plt.subplot2grid(shape=gridsize, loc=(1, 0), sharex=ax1)
			ax3 = plt.subplot2grid(shape=gridsize, loc=(2, 0), sharex=ax1)

			xticks = np.arange(0, int(slice_in_ms / step_size) + 1, int(1 / step_size))
			xticklabels = (xticks * step_size).astype(int)

			ax1.title.set_text(f"Slice #{slice_index + 1}/{slices_number}")
			# plot an area of EES
			ax1.axvspan(xmin=0, xmax=l_poly_border, color='g', alpha=0.3, label="EES area")
			# plot EES
			ax1.axvline(x=mono_answer_index, color='orange', linewidth=3, label="EES")
			# plot zero line
			ax1.axhline(y=0, color='k', linewidth=1, linestyle='dotted')
			# plot original bio data per slice
			ax1.plot(splitted_per_slice_original[slice_index], linewidth=0.7)
			# plot Q1 and Q3 areas, and median
			ax1.plot(smoothed_Q1, color='k', linewidth=3.5, label="Q1/Q3 values")
			ax1.plot(smoothed_Q3, color='k', linewidth=3.5)
			ax1.plot(median, linestyle='--', color='grey', label="median value")
			# plot extremuma
			ax1.plot(e_all_Q1_minima_indexes, e_all_Q1_minima_values, '.', markersize=3, color=min_color,
			         label="minima extremuma")
			ax1.plot(e_all_Q1_maxima_indexes, e_all_Q1_maxima_values, '.', markersize=3, color=max_color,
			         label="maxima extremuma")
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
			ax1.axvline(x=latency_index, color='k')

			ax1.set_xticks(xticks)
			ax1.set_xticklabels(xticklabels)
			ax1.set_xlim(0, int(slice_in_ms / step_size))
			ax1.set_ylim(-1, 1)
			ax1.grid(axis='x', linestyle='--')
			ax1.legend()

			ax2.title.set_text("Delta of variance per iter")
			# plot an area of EES
			ax2.axvline(x=latency_index, color='k')
			ax2.axvspan(xmin=0, xmax=l_poly_border, color='g', alpha=0.3, label="EES area")
			ax2.bar(range(len(delta_smoothed_data)), delta_smoothed_data, width=0.4, zorder=3, color='k', label='delta')

			ax2.set_xticks(xticks)
			ax2.set_xticklabels(xticklabels)
			ax2.set_xlim(0, int(slice_in_ms / step_size))
			ax2.grid(axis='x', linestyle='--', zorder=0)
			ax2.legend()

			ax3.title.set_text("Gradient")
			ax3.axvspan(xmin=0, xmax=l_poly_border, color='g', alpha=0.3, label="EES area")
			ax3.axhline(y=0, linestyle='--', color='k')
			ax3.axvline(x=latency_index, color='k')
			ax3.axhline(y=gradient_Q3, color='g', label="positive gradient mean", linewidth=2)
			ax3.plot(gradient, label='delta gradient', linewidth=2)
			ax3.plot(latency_index, gradient[latency_index], '.', markersize=15, label="latency", color='k')
			ax3.scatter(range(len(gradient)), gradient, s=10)

			ax3.set_xticks(xticks)
			ax3.set_xticklabels(xticklabels)
			ax3.set_xlim(0, int(slice_in_ms / step_size))
			ax3.grid(axis='x', linestyle='--', zorder=0)
			ax3.legend()

			plt.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.04, hspace=0.2)
			plt.show()

	if debugging:
		# original data
		plt.subplots(figsize=(16, 9))
		for slice_index, slice_boxplot_data in enumerate(splitted_per_slice_original):
			y_offset = slice_index * 1
			plt.plot(np.arange(len(slice_boxplot_data)) * step_size, slice_boxplot_data + y_offset)
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
			plt.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color='r', alpha=0.3,
			                 label="flier")
			plt.fill_between(shared_x, data[:, k_whiskers_high], data[:, k_whiskers_low], color='r', alpha=0.5,
			                 label="whisker")
			plt.fill_between(shared_x, data[:, k_box_Q3], data[:, k_box_Q1], color='r', alpha=0.7, label="box")
			plt.plot(shared_x, data[:, k_median], linestyle='--', color='k')
			yticks.append(data[:, k_median][0])
		plt.xticks(range(slice_in_ms + 1), range(slice_in_ms + 1))
		plt.grid(axis='x')

		lat_x = [x * step_size for x in global_lat_indexes]
		lat_y = [splitted_per_slice_boxplots[slice_index][:, k_median][lat] for slice_index, lat
		         in enumerate(global_lat_indexes)]
		plt.plot(lat_x, lat_y, linewidth=3, color='g')

		# plt.xticks(range(100), [x * bio_step for x in range(100) if x % 4 == 0])
		plt.yticks(yticks, range(1, slices_number + 1))
		plt.xlim(0, slice_in_ms)

		plt.show()

	global_lat_indexes = np.array(global_lat_indexes) * step_size
	global_amp_values = np.array(global_amp_values)
	global_mono_indexes = np.array(global_mono_indexes) * step_size

	return global_lat_indexes, global_amp_values, global_mono_indexes


def get_peaks(data_runs, latencies, ees_hz, step_size):
	"""
	TODO: add docstring
	Args:
		data_runs:
		latencies:
		ees_hz:
		step_size:
	Returns:
		list: number of peaks per slice
	"""
	max_times_amp = []
	max_values_amp = []
	min_times_amp = []
	min_values_amp = []
	max_peaks = []
	min_peaks = []

	slice_in_ms = 1000 / ees_hz
	slice_in_steps = int(slice_in_ms / step_size)

	#
	for data in data_runs:
		# smooth data to remove micro-peaks
		smoothed_data = smooth(data, 7)
		# split data by slice based on EES (slice length)
		splitted_per_slice = split_by_slices(smoothed_data, slice_in_steps)
		# calc extrema per slice data
		for poly_border, slice_data in zip(latencies, splitted_per_slice):
			poly_border = int(poly_border / step_size)
			e_all_minima_indexes, e_all_minima_values = find_extrema(slice_data, np.less_equal)
			e_all_maxima_indexes, e_all_maxima_values = find_extrema(slice_data, np.greater_equal)

			e_poly_minima_indexes = e_all_minima_indexes[e_all_minima_indexes > poly_border]
			e_poly_minima_values = e_all_minima_values[e_all_minima_indexes > poly_border]

			e_poly_maxima_indexes = e_all_maxima_indexes[e_all_maxima_indexes > poly_border]
			e_poly_maxima_values = e_all_maxima_values[e_all_maxima_indexes > poly_border]

			min_peaks.append(len(e_poly_minima_indexes))
			max_peaks.append(len(e_poly_maxima_indexes))

			min_times_amp.append(e_poly_minima_indexes)
			min_values_amp.append(e_poly_minima_values)

			max_times_amp.append(e_poly_maxima_indexes)
			max_values_amp.append(e_poly_maxima_values)

	min_peaks = np.split(np.array(min_peaks), len(data_runs))
	max_peaks = np.split(np.array(max_peaks), len(data_runs))

	min_times_amp = np.split(np.array(min_times_amp), len(data_runs))
	min_values_amp = np.split(np.array(min_values_amp), len(data_runs))

	max_times_amp = np.split(np.array(max_times_amp), len(data_runs))
	max_values_amp = np.split(np.array(max_values_amp), len(data_runs))

	sum_peaks = []
	for i in range(len(min_peaks)):
		for j in range(len(min_peaks[i])):
			sum_peaks.append(max_peaks[i][j] + min_peaks[i][j])
	sum_peaks = sum(sum_peaks) / len(data_runs)

	sum_peaks_for_plot = []
	for j in range(len(max_peaks)):
		sum_peaks_for_plot_tmp = []
		for i in range(len(max_peaks[j])):
			sum_peaks_for_plot_tmp.append(max_peaks[j][i] + min_peaks[j][i])
		sum_peaks_for_plot.append(sum_peaks_for_plot_tmp)

	avg_sum_peaks_per_slice = list(map(sum, np.array(sum_peaks_for_plot).T))
	avg_sum_peaks_per_slice = [a / len(data_runs) for a in avg_sum_peaks_per_slice]
	for a in range(len(avg_sum_peaks_per_slice)):
		avg_sum_peaks_per_slice[a] = round(avg_sum_peaks_per_slice[a], 1)

	all_peaks_sum = []
	for i in range(len(sum_peaks_for_plot)):
		all_peaks_sum.append(sum(sum_peaks_for_plot[i]))

	return avg_sum_peaks_per_slice


def plot_3D_PCA(data_pack, save_to):
	"""
	TODO: add docstring
	Args:
		data_pack (list of tuple): special structure to easily work with (coords, color and label)
		save_to (str): save folder path
	"""
	x1 = []
	y1 = []
	z1 = []

	for elev, azim, title in (0, -90.1, "Lat Peak"), (0.1, 0.1, "Amp Peak"), (89.9, -90.1, "Lat Amp"):
		# init 3D projection figure
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')
		# plot each data pack
		# coords is a matrix of coordinates, stacked as [[x1, y1, z1], [x2, y2, z2] ...]
		for coords, color, label in data_pack:
			# create PCA instance and fit the model with coords
			pca = PCA(n_components=3)
			pca.fit(coords)
			# get the center (mean value of points cloud)
			center = pca.mean_
			# print("center = ", center)
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
			# print("radii = ", radii)
			# print("rotation = ", rotation)
			# print("vectors_points = ", vectors_points)
			# plot PCA vectors
			for point_head in vectors_points:
				arrow = Arrow3D(*zip(center.T, point_head.T), mutation_scale=20, lw=3, arrowstyle="-|>", color=color)
				ax.add_artist(arrow)
			# plot cloud of points
			ax.scatter(*coords.T, alpha=0.5, s=30, color=color, label=label)
			# plot ellipsoid
			x, y, z = plot_ellipsoid(center, radii, rotation, plot_axes=False, color=color, alpha=0.1)

			# # plot by combinations (lat, peaks) (amp, peaks) (lat, amp)
			# for comb_A, comb_B in (0, 2), (1, 2), (0, 1):
			# 	p0 = vectors_points[comb_A][[comb_A, comb_B]]
			# 	p1 = vectors_points[comb_B][[comb_A, comb_B]]
			# 	c = center[[comb_A, comb_B]]
			# 	a = length(p0, c)
			# 	b = length(p1, c)
			# 	area = a * b * np.pi
			# 	angle = angle_between(p0 - c, p1 - c)
			# 	dataset_meta[label].append((f"{labels[comb_A]}/{labels[comb_B]}", area, angle, c))

			x1.append(x)
			y1.append(y)
			z1.append(z)

		# figure properties
		ax.set_xticklabels(ax.get_xticks().astype(float), fontsize=35, rotation=90)
		ax.set_yticklabels(ax.get_yticks().astype(float), fontsize=35, rotation=90)
		ax.set_zticklabels(ax.get_zticks().astype(float), fontsize=35)

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
		plt.savefig(f"{save_to}/{title}.pdf", dpi=250, format="pdf")
		plt.show()
		plt.close(fig)
	return x1, y1, z1