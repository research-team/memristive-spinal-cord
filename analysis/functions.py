import csv
import logging
import numpy as np
import h5py as hdf5
import pylab as plt
import matplotlib.transforms as transforms

from scipy.stats import kstwobign
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse


logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()


def normalization(data, dmin=0, dmax=1, save_centering=False):
	"""
	Normalization in [a, b] interval or with saving centering
	x` = (b - a) * (xi - min(x)) / (max(x) - min(x)) + a
	Args:
		data (np.ndarray): data for normalization
		dmin (float): left interval
		dmax (float): right interval
		save_centering (bool): if True -- will save data centering and just normalize by lowest data
	Returns:
		np.ndarray: normalized data
	"""
	# checking on errors
	if dmin >= dmax:
		raise Exception("Left interval 'dmin' must be fewer than right interval 'dmax'")
	if save_centering:
		return data / abs(min(data))
	else:
		min_x = min(data)
		max_x = max(data)
		return (data - min_x) * (dmax - dmin) / (max_x - min_x) + dmin


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
	"""
	Create a plot of the covariance confidence ellipse of *x* and *y*.
	Args:
		x (np.ndarray): array-like, shape (n, )
		y (np.ndarray): array-like, shape (n, )
		ax (matplotlib.axes.Axes): the axes object to draw the ellipse into.
		n_std (float): the number of standard deviations to determine the ellipse's radiuses
		facecolor:
		**kwargs (~matplotlib.patches.Patch): properties
	Returns:
		matplotlib.patches.Ellipse: plot
	"""
	if x.size != y.size:
		raise ValueError("x and y must be the same size")

	cov = np.cov(x, y)
	pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
	# Using a special case to obtain the eigenvalues of this
	# two-dimensionl dataset.
	ell_radius_x = np.sqrt(1 + pearson)
	ell_radius_y = np.sqrt(1 - pearson)
	ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

	# Calculating the stdandard deviation of x from
	# the squareroot of the variance and multiplying
	# with the given number of standard deviations.
	scale_x = np.sqrt(cov[0, 0]) * n_std
	mean_x = np.mean(x)

	# calculating the stdandard deviation of y ...
	scale_y = np.sqrt(cov[1, 1]) * n_std
	mean_y = np.mean(y)

	transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

	ellipse.set_transform(transf + ax.transData)
	return ax.add_patch(ellipse)


def read_hdf5(filepath, rat):
	"""
	ToDo add info
	Args:
		filepath (str): path to the file
		rat (int): rat id
	Returns:
		np.ndarray: data
	"""
	data_by_test = []
	with hdf5.File(filepath, 'r') as file:
		for test_names, test_values in file.items():
			if f"#{rat}" in test_names and len(test_values[:]) != 0:
				data_by_test.append(test_values[:])
	if len(data_by_test):
		log.info(f"{len(data_by_test)} packs in rat #{rat} inside of {filepath}")
		return data_by_test
	return None

def read_bio_data(path):
	"""
	Function for reading of bio data from txt file
	Args:
		path: string
			path to file

	Returns:
		data_RMG :list
			readed data from the first till the last stimulation,
		shifted_indexes: list
			stimulations from the zero
	"""
	with open(path) as file:
		# skipping headers of the file
		for i in range(6):
			file.readline()
		reader = csv.reader(file, delimiter='\t')
		# group elements by column (zipping)
		grouped_elements_by_column = list(zip(*reader))
		# avoid of NaN data
		raw_data_RMG = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[2]]
		# FixMe use 5 if new data else 7
		data_stim = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[7]]
	# preprocessing: finding minimal extrema an their indexes
	mins, indexes = find_mins(data_stim)
	# remove raw data before the first EES and after the last (slicing)
	data_RMG = raw_data_RMG[indexes[0]:indexes[-1]]
	# shift indexes to be normalized with data RMG (because a data was sliced) by value of the first EES
	shifted_indexes = [d - indexes[0] for d in indexes]

	return data_RMG, shifted_indexes


def subsampling(dataset, dstep_from, dstep_to):
	# to convert
	if dstep_from == dstep_to or not dstep_from or not dstep_to:
		return dataset
	sub_step = int(dstep_to / dstep_from)
	return dataset[:, ::sub_step]


def trim_data(dataset, beg=None, end=None):
	"""
	Args:
		dataset (np.ndarray):
		beg (None or int):
		end (None or int):
	Returns:
		np.ndarray: extracted data
	"""
	if beg is None:
		beg = 0
	if end is None:
		end = int(10e6)

	return [d[beg:end] for d in dataset]


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
		y_points (np.ndarray): list of Y points value
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
	cc = rotated_dots_2D.mean(axis=0)
	rotated_dots_2D -= cc

	# plot debugging figures
	if debugging:
		plt.figure(figsize=(16, 9))
		# plot all dots and connect them
		plt.plot(dots_2D[:, X], dots_2D[:, Y], lw=2)
		# plt.plot(dots_2D[:, X], dots_2D[:, Y], '.', alpha=0.5)
		# plot vectors
		for v_length, vector in zip(pca.explained_variance_, pca.components_):
			v = vector * 3 * np.sqrt(v_length)
			draw_vector(pca.mean_, pca.mean_ + v, color='k')
		# figure properties
		plt.axhline(y=pca.mean_[1], c='k', lw=2, ls='--')

		plt.xticks(fontsize=30)
		plt.yticks(fontsize=30)
		plt.tight_layout()
		plt.show()
		plt.close()

		plt.figure(figsize=(16, 9))
		# plot ogignal data on centered
		plt.plot(range(len(dots_2D)), dots_2D[:, Y] - cc[1], label='original', lw=2)
		plt.plot(range(len(rotated_dots_2D)), rotated_dots_2D[:, Y], label='centered', lw=2)
		plt.axhline(y=0, c='k', ls='--', lw=2)
		# figure properties
		plt.xticks(fontsize=30)
		plt.yticks(fontsize=30)
		plt.tight_layout()
		plt.legend(fontsize=20)
		plt.show()

	return rotated_dots_2D[:, 1]


def calc_boxplots(dots, percents=(25, 50, 75)):
	"""
	Function for calculating boxplots from array of dots
	Args:
		dots (np.ndarray): array of dots
		percents (tuple):
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


def get_boxplots(sliced_datasets):
	"""
	ToDo
	Args:
		sliced_datasets (np.ndarray):
	Returns:
		np.ndarray: boxplots per slice per dataset
	"""
	if type(sliced_datasets) is not np.ndarray:
		raise TypeError("Non valid type of data - use only np.ndarray")
	# additional properties
	slices_number = len(sliced_datasets[0])
	# calc boxplot per dot in envolved dataset
	envolved_dataset = np.array([d.flatten() for d in sliced_datasets]).T
	boxplots = np.array([calc_boxplots(dots_per_iter) for dots_per_iter in envolved_dataset])
	splitted_boxplots = np.split(boxplots, slices_number)

	return splitted_boxplots


def calibrate_data(dataset, source):
	"""
	Center the data set, then normalize it and return
	Args:
		dataset (np.ndarray): original dataset
		source:
	Returns:
		np.ndarray: prepared dataset per test
	"""
	prepared_data = []

	for data_per_test in dataset:
		data_per_test = np.array(data_per_test)
		if source == "bio":
			centered_data = center_data_by_line(data_per_test)
		elif source == "nest":
			centered_data = data_per_test
		elif source == "neuron":
			centered_data = data_per_test + 0.57
		elif source == "gras":
			centered_data = data_per_test - data_per_test[0]
		else:
			raise Exception(f"Cannot recognize '{source}' source")
		normalized_data = normalization(centered_data, save_centering=True)
		prepared_data.append(normalized_data.tolist())

	return prepared_data


def parse_filename(filename):
	"""
	Example filename: bio_E_PLT_13.5cms_40Hz_2pedal_0.1step

	Args:
		filename (str):

	Returns:

	"""
	meta = filename.split("_")
	source = meta[0]
	if source not in ['bio', 'neuron', 'gras', 'nest']:
		raise Exception("Cannot recognize source")

	muscle = meta[1]
	if muscle not in ['E', 'F']:
		raise Exception("Cannot recognize muscle name")

	mode = meta[2]
	if mode not in ['AIR', 'TOE', 'PLT', 'QPZ', 'STR']:
		raise Exception("Cannot recognize mode")

	speed = meta[3].replace("cms", "")
	if speed not in ['21', '13.5', '6']:
		raise Exception("Cannot recognize speed")

	rate = int(meta[4].replace("Hz", ""))
	if rate not in range(5, 201, 5):
		raise Exception("Cannot recognize Hz rate")

	pedal = meta[5].replace("pedal", "")
	if pedal not in ['2', '4']:
		raise Exception("Cannot recognize pedal")

	stepsize = float(meta[6].replace("step.hdf5", ""))
	if stepsize not in (0.025, 0.1, 0.25):
		raise Exception("Step size not in allowed list")

	return source, muscle, mode, speed, rate, pedal, stepsize


def optimized_peacock2(xx, yy):
	n1 = xx.shape[0]
	n2 = yy.shape[0]

	# greatest common divisor: dd
	dd = np.gcd(n1, n2)
	# least common multiple: L
	L = n1 / dd * n2
	d1 = L / n1
	d2 = L / n2

	assert xx.shape[1] == 2
	assert yy.shape[1] == 2

	def combine(a, b):
		return np.append(a, b)

	xy1 = combine(xx[:, 0], yy[:, 0])
	xy2 = combine(xx[:, 1], yy[:, 1])

	I1 = np.argsort(xy1)
	I2 = np.argsort(xy2)

	I2_lt_n1 = np.where(I2 < n1, d1, -d2)

	max_hnn_M, max_hpn_M, max_hnp_M, max_hpp_M = 0, 0, 0, 0

	for zu in xy1[I1]:
		# hnn
		hnn_M = np.where(xy1[I2] <= zu, I2_lt_n1, 0)
		max_hnn_M = max(np.max((np.abs(np.cumsum(hnn_M)), np.abs(hnn_M))), max_hnn_M)
		# hpn
		hpn_M = np.where(xy1[I2] > zu, I2_lt_n1, 0)
		max_hpn_M = max(np.max((np.abs(np.cumsum(hpn_M)), np.abs(hpn_M))), max_hpn_M)
		# hnp
		hnp_M = np.where(xy1[I2][::-1] <= zu, I2_lt_n1, 0)
		max_hnp_M = max(np.max((np.abs(np.cumsum(hnp_M)), np.abs(hnp_M))), max_hnp_M)
		# hpp
		hpp_M = np.where(xy1[I2][::-1] > zu, I2_lt_n1, 0)
		max_hpp_M = max(np.max((np.abs(np.cumsum(hpp_M)), np.abs(hpp_M))), max_hpp_M)

	D = max([max_hnn_M, max_hpn_M, max_hnp_M, max_hpp_M]) / L

	return NotImplemented


def peacock2(xx, yy):
	n1 = xx.shape[0]
	n2 = yy.shape[0]

	# greatest common divisor: dd
	dd = np.gcd(n1, n2)
	# least common multiple: L
	L = n1 / dd * n2
	d1 = L / n1
	d2 = L / n2

	assert xx.shape[1] == 2
	assert yy.shape[1] == 2

	def combine(a, b):
		return np.append(a, b)

	xy1 = combine(xx[:, 0], yy[:, 0])
	xy2 = combine(xx[:, 1], yy[:, 1])

	I1 = np.argsort(xy1)
	I2 = np.argsort(xy2)

	max_hnn, max_hpn, max_hnp, max_hpp = [0] * 4

	for zu in xy1[I1]:
		hnn = 0
		hpn = 0

		for v in I2:
			if xy1[v] <= zu:
				hnn += d1 if v < n1 else -d2
				max_hnn = max(max_hnn, abs(hnn))
			else:
				hpn += d1 if v < n1 else -d2
				max_hpn = max(max_hpn, abs(hpn))

		hnp = 0
		hpp = 0

		for v in I2[::-1]:
			if xy1[v] <= zu:
				hnp += d1 if v < n1 else -d2
				max_hnp = max(max_hnp, abs(hnp))
			else:
				hpp += d1 if v < n1 else -d2
				max_hpp = max(max_hpp, abs(hpp))

	D = max([max_hnn, max_hpn, max_hnp, max_hpp]) / L

	# p-value calculating
	en = np.sqrt(n1 * n2 / (n1 + n2))
	prob = kstwobign.sf(en * D)

	return D, prob
