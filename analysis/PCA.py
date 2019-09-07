import logging
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()

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
	# vector containing the center of the ellipsoid
	center = P.T @ u
	# this matrix contains all the information regarding the shape of the ellipsoid
	matrixA = np.linalg.inv(P.T @ (np.diag(u) @ P) - np.outer(center, center)) / dimension
	# to get the radii and orientation of the ellipsoid take the SVD of the output matrix A
	_, size, rotation = np.linalg.svd(matrixA)
	# the radii are given by
	radiuses = 1 / np.sqrt(size)
	# rotation matrix gives the orientation of the ellipsoid
	return radiuses, rotation, matrixA


def plot_ellipsoid(center, radii, rotation, plot_axes=False, color='b'):
	"""
	Plot an ellipsoid
	Args:
		center (np.ndarray): center of the ellipsoid
		radii (np.ndarray): radius per axis
		rotation (np.ndarray): rotation matrix
		plot_axes (bool): plot the axis of ellipsoid if need
		color (str): color in matlab forms (hex, name of color, first char of color)
	"""
	# (for plotting) set the number of grid for plotting surface
	stride = 4
	#
	phi = np.linspace(0, np.pi, 100)
	theta = np.linspace(0, 2 * np.pi, 100)
	# cartesian coordinates that correspond to the spherical angles
	x = radii[0] * np.outer(np.cos(theta), np.sin(phi))
	y = radii[1] * np.outer(np.sin(theta), np.sin(phi))
	z = radii[2] * np.outer(np.ones_like(theta), np.cos(phi))
	# rotate accordingly
	for i in range(len(x)):
		for j in range(len(x)):
			x[i, j], y[i, j], z[i, j] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

	ax = plt.gca()
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
	# plot ellipsoid with wireframe
	ax.plot_wireframe(x, y, z, rstride=stride, cstride=stride, color=color, alpha=0.1)
	ax.plot_surface(x, y, z, rstride=stride, cstride=stride, color=color, alpha=0.05)


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
	# remove tails
	if len(splitted_per_slice[0]) != len(splitted_per_slice[-1]):
		del splitted_per_slice[-1]
	return splitted_per_slice


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


def find_extrema(array, condition):
	"""
	Advanced wrapper of numpy.argrelextrema
	Args:
		array (np.ndarray): data array
		condition (np.ufunc): e.g. np.less (<), np.great_equal (>=) and etc.
	Returns:
		np.ndarray: indexes of extrema
		np.ndarray: values of extrema
	"""
	# get indexes of extrema
	indexes = argrelextrema(array, condition)[0]
	# in case where data line is horisontal and doesn't have any extrema -- return None
	if len(indexes) == 0:
		return None, None
	# get values based on found indexes
	values = array[indexes]
	# calc the difference between nearby extrema values
	diff_nearby_extrema = np.abs(np.diff(values, n=1))
	# form indexes where no twin extrema (the case when data line is horisontal and have two extrema on borders)
	indexes = np.array([index for index, diff in zip(indexes, diff_nearby_extrema) if diff > 0] + [indexes[-1]])
	# get values based on filtered indexes
	values = array[indexes]

	return indexes, values


def merge_extrema(minima_indexes, minima_values, maxima_indexes, maxima_values):
	"""
	ToDo add info
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


def filter_extrema(merged_names, merged_indexes, merged_values, allowed_diff):
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
	i = 0
	next_i = 0

	while i < len(merged_names) and next_i < len(merged_names):
		next_i = i + 1
		while next_i < len(merged_names):
			if abs(merged_values[i] - merged_values[next_i]) > allowed_diff:
				filtered_mask_indexes.append(next_i)
				i = next_i
				break
			next_i += 1
	filtered_mask_indexes = np.append(0, filtered_mask_indexes).astype(int)
	e_poly_names = merged_names[filtered_mask_indexes]
	e_poly_indexes = merged_indexes[filtered_mask_indexes]
	e_poly_values = merged_values[filtered_mask_indexes]

	return e_poly_names, e_poly_indexes, e_poly_values


def get_lat_per_exp(sliced_datasets, step_size, debugging=False):
	"""
	Function for finding latencies at each slice in normalized (!) data
	Args:
		sliced_datasets (np.ndarry): arrays of data
		                      data per slice
		               [[...], [...], [...], [...],
		dataset number  [...], [...], [...], [...],
		                [...], [...], [...], [...]]
		step_size (float): data step
		debugging (bool): True -- will print debugging info and plot figures
	Returns:
		np.ndarray: latencies indexes
	"""
	if type(sliced_datasets) is not np.ndarray:
		raise TypeError("Non valid type of data - use only np.ndarray")

	global_lat_indexes = []
	micro_border = 0.005
	l_poly_border = int(10 / step_size)
	# or use sliced_datasets.reshape(-1, sliced_datasets.shape[2])
	for slices_per_experiment in sliced_datasets:
		for slice_data in slices_per_experiment:
			# smooth data to avoid micro peaks and noise
			smoothed_data = smooth(slice_data, 2)
			smoothed_data[:2] = slice_data[:2]
			smoothed_data[-2:] = slice_data[-2:]

			# I. find latencies (begining of poly answer)
			gradient = np.gradient(smoothed_data)
			assert len(gradient) == len(smoothed_data)
			# get only poly area data (exclude mono answer and activity before it)
			poly_gradient = gradient[l_poly_border:]
			# get positive gradient X Y data
			pos_gradient_x = np.argwhere(poly_gradient > 0).flatten()
			pos_gradient_y = poly_gradient[pos_gradient_x].flatten()
			# get negative gradient X Y data
			negative_gradient_x = np.argwhere(poly_gradient < 0).flatten()
			negative_gradient_y = poly_gradient[negative_gradient_x].flatten()
			# calc the median, Q1, and Q3 values of dots
			if len(pos_gradient_y):
				pos_gradient_Q1, pos_gradient_med, pos_gradient_Q3 = np.percentile(pos_gradient_y, [20, 50, 80])
			else:
				pos_gradient_Q1, pos_gradient_med, pos_gradient_Q3 = np.inf, np.inf, np.inf
			if len(negative_gradient_y):
				neg_gradient_Q1, neg_gradient_med, neg_gradient_Q3 = np.percentile(negative_gradient_y, [20, 50, 80])
			else:
				neg_gradient_Q1, neg_gradient_med, neg_gradient_Q3 = -np.inf, -np.inf, -np.inf
			# find the index of latency by the cross between gradient and negative grad Q1/positive grad Q3
			for index, grad in enumerate(gradient[l_poly_border:]):
				if (grad > pos_gradient_Q3 or grad < neg_gradient_Q1) and (grad > micro_border or grad < -micro_border):
					latency_index = index + l_poly_border
					break
			# if not found -- take the last index
			else:
				latency_index = len(gradient) - 1
			# collect found item
			global_lat_indexes.append(latency_index)

			if debugging:
				plt.figure(figsize=(16, 9))
				plt.axhline(y=pos_gradient_Q3, color='r', linestyle='dotted')
				plt.axhline(y=neg_gradient_Q1, color='b', linestyle='dotted')
				plt.fill_between(np.arange(len(poly_gradient)) + l_poly_border,
				                 poly_gradient, [0] * len(poly_gradient), color='r', alpha=0.6)
				plt.fill_between(range(len(gradient)),
				                 [0] * len(gradient), [-1] * len(gradient), color='w')
				plt.fill_between(np.arange(len(poly_gradient)) + l_poly_border,
				                 poly_gradient, [0] * len(poly_gradient), color='b', alpha=0.2)
				plt.axhline(y=0, color='k', linestyle='--')
				plt.plot(smoothed_data, color='b', label="slice data")
				plt.plot(np.arange(len(gradient)), gradient, color='r', label="gradient")
				plt.plot(np.arange(len(gradient)), gradient, '.', color='k', markersize=1)
				plt.plot([latency_index], [smoothed_data[latency_index]], '.', color='k', markersize=15)
				plt.xlim(0, len(smoothed_data))
				plt.xticks(range(len(smoothed_data)),
				           [x * step_size if x % 25 == 0 else None for x in range(len(smoothed_data) + 1)])
				plt.legend()
				plt.show()

	return np.array(global_lat_indexes) * step_size


def get_amp_per_exp(sliced_datasets, step_size):
	"""
	Function for finding latencies at each slice in normalized (!) data
	Args:
		sliced_datasets (np.ndarry): arrays of data
		                      data per slice
		               [[...], [...], [...], [...],
		dataset number  [...], [...], [...], [...],
		                [...], [...], [...], [...]]
		step_size (float): data step
	Returns:
		np.ndarray: amplitudes values
	"""
	if type(sliced_datasets) is not np.ndarray:
		raise TypeError("Non valid type of data - use only np.ndarray")

	global_amp_values = []
	l_poly_border = int(10 / step_size)

	# or use sliced_datasets.reshape(-1, sliced_datasets.shape[2])
	for slices_per_experiment in sliced_datasets:
		for slice_data in slices_per_experiment:
			# smooth data to avoid micro peaks and noise
			smoothed_data = smooth(slice_data, 2)
			smoothed_data[:2] = slice_data[:2]
			smoothed_data[-2:] = slice_data[-2:]

			# II. find the sum of amplitudes (integral area)
			amplitude_sum = np.sum(np.abs(smoothed_data[l_poly_border:]))
			global_amp_values.append(amplitude_sum)

	return np.array(global_amp_values)


def get_peak_per_exp(sliced_datasets, step_size, split_by_intervals=False, debugging=False):
	"""
	Function for finding latencies at each slice in normalized (!) data
	Args:
		sliced_datasets (np.ndarry): arrays of data
		                      data per slice
		               [[...], [...], [...], [...],
		dataset number  [...], [...], [...], [...],
		                [...], [...], [...], [...]]
		step_size (float): data step
		split_by_intervals (bool):
		debugging (bool): True -- will print debugging info and plot figures
	Returns:
		np.ndarray: peaks number or intervals
	"""
	if type(sliced_datasets) is not np.ndarray:
		raise TypeError("Non valid type of data - use only np.ndarray")

	global_peaks_number = []
	# only for non-interval's algorithm
	l_poly_border = int(10 / step_size)
	dataset_size = len(sliced_datasets)
	slices_number = len(sliced_datasets[0])
	# static borders (convert by division to steps)
	intervals = np.array([[0, 3], [7, 10], [10, 15], [15, 20], [20, 25]]) / step_size
	# prepare array for fitting data
	peaks_per_interval = np.zeros((slices_number, len(intervals)))
	#
	mono_end = int(7 / step_size)
	mono_start = int(3 / step_size)
	# nested loop for processong each slice in datasets
	# or use sliced_datasets.reshape(-1, sliced_datasets.shape[2]) to make it 1D with saving order
	for slices_per_experiment in sliced_datasets:
		for slice_index, slice_data in enumerate(slices_per_experiment):
			# smooth data (small value for smoothing only micro-peaks)
			smoothed_data = smooth(slice_data, 2)
			# optional variant to calculate peaks in specific intervals
			# get all extrema
			e_maxima_indexes, e_maxima_values = find_extrema(smoothed_data, np.greater)
			e_minima_indexes, e_minima_values = find_extrema(smoothed_data, np.less)

			if split_by_intervals:
				# merge extrema
				e_poly_names, e_poly_indexes, e_poly_values = merge_extrema(e_minima_indexes, e_minima_values,
				                                                            e_maxima_indexes, e_maxima_values)
				# remove extrema included in mono area
				mask = (e_poly_indexes < mono_start) | (e_poly_indexes > mono_end)
				e_poly_names = e_poly_names[mask]
				e_poly_indexes = e_poly_indexes[mask]
				e_poly_values = e_poly_values[mask]

				# check if where are no extrema (no activity)
				if len(e_poly_indexes) == 0:
					e_poly_names, e_poly_indexes, e_poly_values = [], [], []
				else:
					# calc the values which corresponds to the percentiles
					diff_Q1, diff_median, diff_Q3 = np.percentile(e_poly_values, (15, 50, 85))
					# filter extrema: remove micropeaks by Q3 percentile value
					e_poly_names, e_poly_indexes, e_poly_values = filter_extrema(e_poly_names, e_poly_indexes,
					                                                             e_poly_values,
					                                                             allowed_diff=diff_Q3)
				# fill array by intervals
				for interval_index, interval in enumerate(intervals):
					peaks_in_interval = filter(lambda x: interval[0] <= x < interval[1], e_poly_indexes)
					peaks_per_interval[slice_index][interval_index] += len(list(peaks_in_interval))
			else:
				# get maxima/minima extrema
				e_maxima_indexes, e_maxima_values = find_extrema(smoothed_data[l_poly_border:], np.greater)
				e_minima_indexes, e_minima_values = find_extrema(smoothed_data[l_poly_border:], np.less)

				if e_maxima_indexes is None or e_minima_indexes is None:
					global_peaks_number.append(0)
					continue

				e_poly_names, e_poly_indexes, e_poly_values = merge_extrema(e_minima_indexes, e_minima_values,
				                                                            e_maxima_indexes, e_maxima_values)
				# check if where are no extrema (no activity)
				if len(e_poly_indexes) == 0:
					e_poly_names, e_poly_indexes, e_poly_values = [], [], []
				else:
					# calc the values which corresponds to the percentiles
					diff_Q1, diff_median, diff_Q3 = np.percentile(e_poly_values, (15, 50, 85))
					# filter extrema: remove micropeaks by Q3 percentile value
					e_poly_names, e_poly_indexes, e_poly_values = filter_extrema(e_poly_names, e_poly_indexes,
					                                                             e_poly_values,
					                                                             allowed_diff=diff_Q3)
				#
				global_peaks_number.append(len(e_poly_indexes))

			if debugging:
				plt.axvspan(xmin=3 / step_size, xmax=7 / step_size, color='r', alpha=0.3)
				plt.plot(slice_data, color='g', label="original")
				plt.plot(smoothed_data, color='k', label="smoothed")
				x_shift = 0 if split_by_intervals else l_poly_border
				plt.plot(e_poly_indexes + x_shift, e_poly_values, '.', color='k', markersize=20, alpha=0.8)
				plt.plot(e_maxima_indexes + l_poly_border, e_maxima_values, '.', color='r', markersize=10)
				plt.plot(e_minima_indexes + l_poly_border, e_minima_values, '.', color='cyan', markersize=10)
				plt.legend()
				plt.show()

	if split_by_intervals:
		peaks_per_interval = peaks_per_interval / dataset_size
		# reshape
		c = peaks_per_interval[:, 0].copy()
		peaks_per_interval[:, 0: -1] = peaks_per_interval[:, 1:]
		peaks_per_interval[:, -1] = c
		peaks_per_interval[:, -1] = np.append(peaks_per_interval[1:, -1], 0)

		return peaks_per_interval

	return np.array(global_peaks_number)


def plot_3D_PCA(data_pack, save_to, correlation=False):
	"""
	TODO: add docstring
	Args:
		data_pack (list of tuple): special structure to easily work with (coords, color and label)
		save_to (str): save folder path
		correlation (bool): enable or disable corelation calculating
	"""
	# plot PCA at different point of view
	for elev, azim, title in (0, -90.1, "Lat Peak"), (0.1, 0.1, "Amp Peak"), (89.9, -90.1, "Lat Amp"):
		volume_sum = 0
		data_pack_xyz = []
		new_filename = str(title).lower().replace(" ", "_")
		# init 3D projection figure
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')
		# plot each data pack
		for coords, color, filename in data_pack:
			# create PCA instance and fit the model with coords
			pca = PCA(n_components=3)
			# coords is a matrix of coordinates, stacked as [[x1, y1, z1], ... , [xN, yN, zN]]
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
			radii, rotation, matrixA = form_ellipse(axis_points)
			# choose -- calc correlaion or just plot PCA
			if correlation:
				# start calculus of points intersection
				volume = (4 / 3) * np.pi * radii[0] * radii[1] * radii[2]
				volume_sum += volume
				log.info(f"V: {volume}, {filename}")
				# keep ellipsoid surface dots, A matrix, center
				phi = np.linspace(0, np.pi, 600)
				theta = np.linspace(0, 2 * np.pi, 600)
				# cartesian coordinates that correspond to the spherical angles
				x = radii[0] * np.outer(np.cos(theta), np.sin(phi))
				y = radii[1] * np.outer(np.sin(theta), np.sin(phi))
				z = radii[2] * np.outer(np.ones_like(theta), np.cos(phi))
				# rotate accordingly
				for i in range(len(x)):
					for j in range(len(x)):
						x[i, j], y[i, j], z[i, j] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
				data_pack_xyz.append((matrixA, center, x.flatten(), y.flatten(), z.flatten()))
			else:
				# plot PCA vectors
				for point_head in vectors_points:
					arrow = Arrow3D(*zip(center.T, point_head.T), mutation_scale=20, lw=3, arrowstyle="-|>", color=color)
					ax.add_artist(arrow)
				# plot cloud of points
				ax.scatter(*coords.T, alpha=0.2, s=30, color=color, label=filename)
				# plot ellipsoid
				plot_ellipsoid(center, radii, rotation, plot_axes=False, color=color)
		if correlation:
			# collect all intersect point
			points_inside = []
			# get data of two ellipsoids: A matrix, center and points coordinates
			A1, C1, x1, y1, z1 = data_pack_xyz[0]
			A2, C2, x2, y2, z2 = data_pack_xyz[1]
			# based on stackoverflow.com/a/34385879/5891876 solution with own modernization
			# the equation for the surface of an ellipsoid is (x-c)TA(x-c)=1.
			# all we need to check is whether (x-c)TA(x-c) is less than 1 for each of points
			for coord in np.stack((x1, y1, z1), axis=1):
				if np.sum(np.dot(coord - C2, A2 * (coord - C2))) <= 1:
					points_inside.append(coord)
			# do the same for another ellipsoid
			for coord in np.stack((x2, y2, z2), axis=1):
				if np.sum(np.dot(coord - C1, A1 * (coord - C1))) <= 1:
					points_inside.append(coord)
			points_inside = np.array(points_inside)

			if not len(points_inside):
				log.info("NO INTERSECTIONS: 0 correlation")
				return

			# form convex hull of 3D surface
			hull = ConvexHull(points_inside)
			# get a volume of this surface
			v_intersection = hull.volume
			# calc correlation value
			pca_similarity = v_intersection / (volume_sum - v_intersection)
			log.info(f"PCA similarity: {pca_similarity}")
			# debugging plotting
			# ax.scatter(*points_in.T, alpha=0.2, s=1, color='r')
		else:
			# figure properties
			ax.xaxis._axinfo['tick']['inward_factor'] = 0
			ax.yaxis._axinfo['tick']['inward_factor'] = 0
			ax.zaxis._axinfo['tick']['inward_factor'] = 0

			ax.tick_params(which='major', length=10, width=3, labelsize=50)
			ax.tick_params(which='minor', length=4, width=2, labelsize=50)

			# remove one of the plane ticks to make output pdf more readable
			if "Lat" not in title:
				ax.set_xticks([])
				ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=len(ax.get_yticks()), integer=True))
				ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=len(ax.get_zticks()), integer=True))
			if "Amp" not in title:
				ax.set_yticks([])
				ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=len(ax.get_xticks()), integer=True))
				ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=len(ax.get_zticks()), integer=True))
			if "Peak" not in title:
				ax.set_zticks([])
				ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=len(ax.get_xticks()), integer=True))
				ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=len(ax.get_yticks()), integer=True))

			plt.legend()
			ax.view_init(elev=elev, azim=azim)
			plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
			plt.savefig(f"{save_to}/{new_filename}.pdf", dpi=250, format="pdf")
			plt.savefig(f"{save_to}/{new_filename}.png", dpi=250, format="png")
			plt.close(fig)
