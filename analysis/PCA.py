import numpy as np
import pylab as plt
import h5py as hdf5
import logging as log
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

np.set_printoptions(suppress=True)

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
	ax.plot_wireframe(x, y, z, rstride=stride, cstride=stride, color=color, alpha=alpha / 2)
	ax.plot_surface(x, y, z, rstride=stride, cstride=stride, alpha=alpha, color=color)


def read_data(filepath):
	with hdf5.File(filepath, 'r') as file:
		data_by_test = [test_values[:] for test_values in file.values()]
	return data_by_test


def select_slices(path, begin, end, sign=1):
	return [sign * data[begin:end][::10] for data in read_data(path)]


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


def draw_vector(p0, p1, color):
	"""
	Small function for drawing vector with arrow by two points
	Args:
		p0 (np.ndarray): begin of the vector
		p1 (np.ndarray): end of the vector
		color (str): the color of vector
	"""
	ax = plt.gca()
	# this plot is fixing the problem of hiding annotations because of their not markers origin
	ax.plot(p1[0], p1[1], '.', alpha=0)
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
		e_all_Q1_minima_indexes, e_all_Q1_minima_values = find_extremuma(smoothed_Q1, np.less_equal)
		e_all_Q1_maxima_indexes, e_all_Q1_maxima_values = find_extremuma(smoothed_Q1, np.greater_equal)
		log.info(f"all Q1 minima extrema: {e_all_Q1_minima_indexes}")
		log.info(f"all Q1 maxima extrema: {e_all_Q1_maxima_indexes}")
		# find all Q3 extremuma indexes and values
		e_all_Q3_minima_indexes, e_all_Q3_minima_values = find_extremuma(smoothed_Q3, np.less_equal)
		e_all_Q3_maxima_indexes, e_all_Q3_maxima_values = find_extremuma(smoothed_Q3, np.greater_equal)
		log.info(f"all Q3 minima extrema: {e_all_Q3_minima_indexes}")
		log.info(f"all Q3 maxima extrema: {e_all_Q3_maxima_indexes}")

		'''[3] find mono answer'''
		# get the minima extremuma of mono answers (Q1 is lower than Q3, so take a Q1 data)
		e_mono_Q1_minima_indexes = e_all_Q1_minima_indexes[e_all_Q1_minima_indexes < ees_zone_time]
		# find an index of the biggest delta between first dot (median[0]) and mono extremuma
		max_delta_Q1_index = max_at(np.abs(smoothed_Q1[e_mono_Q1_minima_indexes] - median[0]))[0]
		# get index of extremuma with the biggest delta
		mono_answer_index = e_mono_Q1_minima_indexes[max_delta_Q1_index]
		log.info(f"Mono answer: {mono_answer_index * data_step}ms")

		"""[4] find a poly area"""
		l_poly_border = ees_zone_time
		# correct this border by the end of the mono answer peak
		for e_index_Q1, e_index_Q3 in zip(e_all_Q1_maxima_indexes, e_all_Q3_maxima_indexes):
			mono_answer_end_index = e_index_Q1 if e_index_Q1 > e_index_Q3 else e_index_Q3
			if mono_answer_end_index > ees_zone_time:
				l_poly_border = mono_answer_end_index
				break
		if l_poly_border > int(10 / data_step):
			l_poly_border = int(10 / data_step)
		log.info(f"Poly area: {l_poly_border * data_step} - {slice_in_ms}")

		"""[5] find a latency"""
		delta_poly_diff = np.abs(np.diff(delta_smoothed_data[l_poly_border:], n=1))
		diff_Q1, diff_median, diff_Q3 = np.percentile(delta_poly_diff, percents)
		allowed_diff_for_extremuma = diff_median

		delta_diff_gradient = np.gradient(delta_smoothed_data)
		delta_diff_poly_gradient = delta_diff_gradient[l_poly_border:]

		gradient_mean = np.mean(delta_diff_poly_gradient[delta_diff_poly_gradient > 0])

		# fine the index of the cross between gradient and Q3
		for index, diff in enumerate(delta_diff_gradient[l_poly_border:], l_poly_border):
			if diff > gradient_mean:
				latency_index = index
				break
		# if not found -- the last index
		else:
			latency_index = len(delta_diff_gradient) - 1
		log.info(f"Latency: {latency_index * data_step}")

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

			xticks = np.arange(0, int(slice_in_ms / data_step) + 1, int(1 / data_step))
			xticklabels = (xticks * data_step).astype(int)

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
			ax1.set_xlim(0, int(slice_in_ms / data_step))
			ax1.set_ylim(-1, 0.5)
			ax1.grid(axis='x', linestyle='--')
			ax1.legend()

			ax2.title.set_text("Delta of variance per iter")
			# plot an area of EES
			ax2.axvline(x=latency_index, color='k')
			ax2.axvspan(xmin=0, xmax=l_poly_border, color='g', alpha=0.3, label="EES area")
			ax2.bar(range(len(delta_smoothed_data)), delta_smoothed_data, width=0.4, zorder=3, color='k', label='delta')

			ax2.set_xticks(xticks)
			ax2.set_xticklabels(xticklabels)
			ax2.set_xlim(0, int(slice_in_ms / data_step))
			ax2.grid(axis='x', linestyle='--', zorder=0)
			ax2.legend()

			ax3.title.set_text("Gradient")
			ax3.axvspan(xmin=0, xmax=l_poly_border, color='g', alpha=0.3, label="EES area")
			ax3.axhline(y=0, linestyle='--', color='k')
			ax3.axvline(x=latency_index, color='k')
			ax3.axhline(y=gradient_mean, color='g', label="positive gradient mean", linewidth=2)
			ax3.plot(delta_diff_gradient, label='delta gradient', linewidth=2)
			ax3.plot(latency_index, delta_diff_gradient[latency_index], '.', markersize=15, label="latency", color='k')
			ax3.scatter(range(len(delta_diff_gradient)), delta_diff_gradient, s=10)

			ax3.set_xticks(xticks)
			ax3.set_xticklabels(xticklabels)
			ax3.set_xlim(0, int(slice_in_ms / data_step))
			ax3.grid(axis='x', linestyle='--', zorder=0)
			ax3.legend()

			plt.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.04, hspace=0.2)
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
			plt.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low], color='r', alpha=0.3,
			                 label="flier")
			plt.fill_between(shared_x, data[:, k_whiskers_high], data[:, k_whiskers_low], color='r', alpha=0.5,
			                 label="whisker")
			plt.fill_between(shared_x, data[:, k_box_Q3], data[:, k_box_Q1], color='r', alpha=0.7, label="box")
			plt.plot(shared_x, data[:, k_median], linestyle='--', color='k')
			yticks.append(data[:, k_median][0])
		plt.xticks(range(slice_in_ms + 1), range(slice_in_ms + 1))
		plt.grid(axis='x')

		lat_x = [x * data_step for x in global_lat_indexes]
		lat_y = [splitted_per_slice_boxplots[slice_index][:, k_median][lat] for slice_index, lat
		         in enumerate(global_lat_indexes)]
		plt.plot(lat_x, lat_y, linewidth=3, color='g')

		# plt.xticks(range(100), [x * bio_step for x in range(100) if x % 4 == 0])
		plt.yticks(yticks, range(1, slices_number + 1))
		plt.xlim(0, slice_in_ms)

		plt.show()

	global_lat_indexes = np.array(global_lat_indexes) * data_step
	global_amp_values = np.array(global_amp_values)

	return global_lat_indexes, global_amp_values


def get_peaks(data, herz, step):
	"""

	Args:
		data:
		herz:
		step:

	Returns:

	"""
	max_times_amp = []
	max_values_amp = []
	min_times_amp = []
	min_values_amp = []
	max_peaks = []
	min_peaks = []

	l_poly_border = int(10 / step)
	slice_length = int(1000 / herz / step)
	slices_number = len(data[0]) // slice_length
	latencies, amplitudes = get_lat_amp(data, herz, step)

	for i, run_data in enumerate(data):
		for sliced_data in np.split(smooth(run_data, 7), slices_number):
			e_all_minima_indexes, e_all_minima_values = find_extremuma(sliced_data, np.less_equal)
			e_all_maxima_indexes, e_all_maxima_values = find_extremuma(sliced_data, np.greater_equal)

			e_poly_minima_indexes = e_all_minima_indexes[e_all_minima_indexes > l_poly_border]
			e_poly_minima_values = e_all_minima_values[e_all_minima_indexes > l_poly_border]

			e_poly_maxima_indexes = e_all_maxima_indexes[e_all_maxima_indexes > l_poly_border]
			e_poly_maxima_values = e_all_maxima_values[e_all_maxima_indexes > l_poly_border]

			min_peaks.append(len(e_poly_minima_indexes))
			max_peaks.append(len(e_poly_maxima_indexes))

			min_times_amp.append(e_poly_minima_indexes)
			min_values_amp.append(e_poly_minima_values)

			max_times_amp.append(e_poly_maxima_indexes)
			max_values_amp.append(e_poly_maxima_values)

	min_peaks = np.split(np.array(min_peaks), len(data))
	max_peaks = np.split(np.array(max_peaks), len(data))

	min_times_amp = np.split(np.array(min_times_amp), len(data))
	min_values_amp = np.split(np.array(min_values_amp), len(data))

	max_times_amp = np.split(np.array(max_times_amp), len(data))
	max_values_amp = np.split(np.array(max_values_amp), len(data))

	sum_peaks = []
	for i in range(len(min_peaks)):
		for j in range(len(min_peaks[i])):
			sum_peaks.append(max_peaks[i][j] + min_peaks[i][j])
	sum_peaks = sum(sum_peaks) / len(data)

	sum_peaks_for_plot = []
	for j in range(len(max_peaks)):
		sum_peaks_for_plot_tmp = []
		for i in range(len(max_peaks[j])):
			sum_peaks_for_plot_tmp.append(max_peaks[j][i] + min_peaks[j][i])
		sum_peaks_for_plot.append(sum_peaks_for_plot_tmp)

	avg_sum_peaks_in_sl = list(map(sum, np.array(sum_peaks_for_plot).T))
	avg_sum_peaks_in_sl = [a / len(data) for a in avg_sum_peaks_in_sl]
	for a in range(len(avg_sum_peaks_in_sl)):
		avg_sum_peaks_in_sl[a] = round(avg_sum_peaks_in_sl[a], 1)

	all_peaks_sum = []
	for i in range(len(sum_peaks_for_plot)):
		all_peaks_sum.append(sum(sum_peaks_for_plot[i]))

	return latencies, max_times_amp, min_times_amp, max_values_amp, min_values_amp, amplitudes, \
	       sum_peaks_for_plot, avg_sum_peaks_in_sl, all_peaks_sum, sum_peaks


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


def plot_pca(debugging=False, plot_3d=False):
	"""
	Preparing data and drawing PCA for them
	Args:
		debugging:
		plot_3d:
	"""
	if debugging:
		log.basicConfig(level=log.INFO)


	bio_path = '/home/alex/GitHub/memristive-spinal-cord/bio-data/hdf5/bio_control_E_21cms_40Hz_i100_2pedal_no5ht_T_2017-09-05.hdf5'
	gras_path = '/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/dat/21 cms 2pedal/MN_E.hdf5'
	neuron_path = '/home/alex/Downloads/Telegram Desktop/mn_E25tests_nr.hdf5'

	# process BIO dataset
	dataset = read_data(bio_path)
	prepared_data = prepare_data(dataset)
	lat_per_slice, amp_per_slice = get_lat_amp(prepared_data, ees_hz=40, data_step=0.25, debugging=True)
	peaks_per_slice = get_peaks(prepared_data, 40, 0.25)[7]
	# form data pack
	bio_pack = [np.stack((lat_per_slice, amp_per_slice, peaks_per_slice), axis=1), "#a6261d", "bio"]

	# process GRAS dataset
	dataset = select_slices(gras_path, 5000, 11000, sign=1)
	prepared_data = prepare_data(dataset)
	lat_per_slice, amp_per_slice = get_lat_amp(prepared_data, ees_hz=40, data_step=0.25, debugging=True)
	peaks_per_slice = get_peaks(prepared_data, 40, 0.25)[7]
	# form data pack
	gras_pack = [np.stack((lat_per_slice, amp_per_slice, peaks_per_slice), axis=1), "#287a72", "gras"]

	# process NEURON dataset
	dataset = select_slices(neuron_path, 0, 6000, sign=-1)
	prepared_data = prepare_data(dataset)
	lat_per_slice, amp_per_slice = get_lat_amp(prepared_data, ees_hz=40, data_step=0.25, debugging=True)
	peaks_per_slice = get_peaks(prepared_data, 40, 0.25)[7]
	# form data pack
	neuron_pack = [np.stack((lat_per_slice, amp_per_slice, peaks_per_slice), axis=1), "#f2aa2e", "neuron"]

	axis_labels = ["Latencies", "Amplitudes", "Peaks"]

	if plot_3d:
		# init 3D projection figure
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		# plot each data pack
		# coords is a matrix of coordinates, stacked as [[x1, y1, z1], [x2, y2, z2] ...]
		for coords, color, label in bio_pack, neuron_pack, gras_pack:
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
			plot_ellipsoid(center, radii, rotation, plot_axes=debugging, color=color, alpha=0.1)
		# figure properties
		ax.set_xlabel(axis_labels[0])
		ax.set_ylabel(axis_labels[1])
		ax.set_zlabel(axis_labels[2])
		plt.legend()
		plt.show()
		plt.close(fig)
	else:
		X = 0
		Y = 1
		# plot by combinations (lat, amp) (amp, peaks) (lat, peaks)
		for comb_A, comb_B in (0, 1), (1, 2), (0, 2):
			# start plotting
			fig, ax = plt.subplots()
			# plot per pack
			for coords, color, title in bio_pack, gras_pack, neuron_pack:
				# get only necessary coords
				coords = coords[:, [comb_A, comb_B]]
				# create PCA instance and fit the model with coords
				pca = PCA(n_components=2)
				pca.fit(coords)
				# get the center (mean value of points cloud)
				center = pca.mean_
				# get PCA vectors' head points (semi axis)
				vectors_points = [3 * np.sqrt(val) * vec for val, vec in zip(pca.explained_variance_, pca.components_)]
				vectors_points = np.array(vectors_points) + center
				# calc an angle between first vector and vertical vector from the center
				vertical = np.array([center[X], center[Y] + 10])
				angle = angle_between(vertical - center, vectors_points[0] - center)
				# check on angle sign
				sign = -1 if vectors_points[0][X] > center[X] else 1
				# calculate ellipse size
				e_height = length(center, vectors_points[0]) * 2
				e_width = length(center, vectors_points[1]) * 2
				# plot PCA vectors
				for point_head in vectors_points:
					draw_vector(center, point_head, color=color)
				# plot dots
				ax.scatter(coords[:, X], coords[:, Y], color=color, label=title, s=80)
				# plot ellipse
				ellipse = Ellipse(xy=tuple(center), width=e_width, height=e_height, angle=sign * angle)
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
			plt.xlabel(axis_labels[comb_A], fontsize=28)
			plt.ylabel(axis_labels[comb_B], fontsize=28)
			plt.legend()
			plt.show()


def run():
	plot_pca()


if __name__ == "__main__":
	run()
