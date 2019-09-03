import logging
import numpy as np
import pylab as plt
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
	ax.plot_wireframe(x, y, z, rstride=stride, cstride=stride, color=color, alpha=0.2)
	ax.plot_surface(x, y, z, rstride=stride, cstride=stride, color=color, alpha=0.1)


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


def get_lat_amp_peak_per_exp(sliced_datasets, step_size, debugging=False):
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
		np.ndarray: amplitudes values
		np.ndarray: peaks number
	"""
	if type(sliced_datasets) is not np.ndarray:
		raise TypeError("Non valid type of data - use only np.ndarray")

	global_amp_values = []
	global_lat_indexes = []
	global_peaks_number = []
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
				pos_gradient_Q1, pos_gradient_med, pos_gradient_Q3 = np.percentile(pos_gradient_y, [25, 50, 75])
			else:
				pos_gradient_Q1, pos_gradient_med, pos_gradient_Q3 = np.inf, np.inf, np.inf
			if len(negative_gradient_y):
				neg_gradient_Q1, neg_gradient_med, neg_gradient_Q3 = np.percentile(negative_gradient_y, [25, 50, 75])
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

			# II. find the sum of amplitudes (integral area)
			amplitude_sum = np.sum(np.abs(smoothed_data[l_poly_border:]))
			global_amp_values.append(amplitude_sum)

			# III. find peaks number by strong smoothing
			smoothed_data = smooth(slice_data, 7)
			smoothed_data[:2] = slice_data[:2]
			smoothed_data[-2:] = slice_data[-2:]
			# get maxima/minima extrema
			e_maxima_indexes, e_maxima_values = find_extrema(smoothed_data[l_poly_border:], np.greater)
			e_minima_indexes, e_minima_values = find_extrema(smoothed_data[l_poly_border:], np.less)
			# sum all found peaks
			peaks_sum = 0
			if e_maxima_indexes is not None:
				peaks_sum += len(e_maxima_indexes)
			if e_minima_indexes is not None:
				peaks_sum += len(e_minima_indexes)
			global_peaks_number.append(peaks_sum)

			if debugging:
				plt.plot(np.gradient(smoothed_data), color='r')
				plt.plot(slice_data, color='g')
				plt.plot(smoothed_data, color='k')
				plt.plot(e_maxima_indexes + l_poly_border, e_maxima_values, '.', color='r')
				plt.plot(e_minima_indexes + l_poly_border, e_minima_values, '.', color='b')
				plt.show()

	return np.array(global_lat_indexes) * step_size, np.array(global_amp_values), np.array(global_peaks_number)


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
		title = str(title).lower().replace(" ", "_")
		# init 3D projection figure
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')
		# plot each data pack
		for coords, color, label in data_pack:
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
				log.info(f"V: {volume}, {label}")
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
				ax.scatter(*coords.T, alpha=0.5, s=30, color=color, label=label)
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
			# remove one of the plane ticks to make output pdf more readable
			if "Lat" not in title:
				ax.set_xticks([])
				ax.set_yticklabels(ax.get_yticks().astype(float), fontsize=35, rotation=90)
				ax.set_zticklabels(ax.get_zticks().astype(float), fontsize=35)
			if "Amp" not in title:
				ax.set_yticks([])
				ax.set_xticklabels(ax.get_xticks().astype(float), fontsize=35, rotation=90)
				ax.set_zticklabels(ax.get_zticks().astype(float), fontsize=35)
			if "Peak" not in title:
				ax.set_zticks([])
				ax.set_xticklabels(ax.get_xticks().astype(float), fontsize=35, rotation=90)
				ax.set_yticklabels(ax.get_yticks().astype(float), fontsize=35)

			plt.legend()
			ax.view_init(elev=elev, azim=azim)
			plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
			plt.savefig(f"{save_to}/{title}.pdf", dpi=250, format="pdf")
			plt.close(fig)
