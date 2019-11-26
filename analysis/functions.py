import csv
import logging
import numpy as np
import h5py as hdf5
import pylab as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import kstwobign


logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()


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
		return data / minimal
	else:
		min_x = min(data)
		max_x = max(data)
		const = (b - a) / (max_x - min_x)

		return (data - min_x) * const + a


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


def read_data(filepath):
	"""
	ToDo add info
	Args:
		filepath (str): path to the file
	Returns:
		np.ndarray: data
	"""
	data_by_test = []
	with hdf5.File(filepath, 'r') as file:
		for test_names, test_values in file.items():
			#if "#8_112309_quip" not in test_names: # not in ["#8_112309_quip_BIPEDAL_burst10_Ton_21.fig", "#8_112309_quip_BIPEDAL_burst3_Ton_14.fig", "#8_112309_quip_BIPEDAL_burst4_Ton_15.fig",  "#8_112309_quip_BIPEDAL_burst6_Ton_17.fig",  "#8_112309_quip_BIPEDAL_burst7_Ton_18.fig",  "#8_112309_quip_BIPEDAL_burst9_Ton_20.fig"]:
			data_by_test.append(test_values[:])
			# data_by_test = [test_values[:] for test_values in file.values()]
		if not all(map(len, data_by_test)):
			raise Exception("hdf5 has an empty data!")
	return np.array(data_by_test)


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
		sub_step = 1
	else:
		sub_step = int(dstep_to / dstep_from)

	subsampled_dataset = [data[::sub_step] for data in dataset]

	return np.array(subsampled_dataset)


def extract_data(path, beg=None, end=None):
	"""
	ToDo add info
	Args:
		path:
		beg:
		end:
	Returns:
		np.ndarray: extracted data
	"""
	if beg is None:
		beg = 0
	if end is None:
		end = int(10e6)

	array = [data[beg:end] for data in read_data(path)]

	return array


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


def prepare_data(dataset):
	"""
	Center the data set, then normalize it and return
	Args:
		dataset (np.ndarray): original dataset
	Returns:
		np.ndarray: prepared dataset per test
	"""
	prepared_data = []
	for data_per_test in dataset:
		centered_data = center_data_by_line(data_per_test)
		normalized_data = normalization(centered_data, save_centering=True)
		prepared_data.append(normalized_data)
	return np.array(prepared_data)


def split_by_slices(data, slice_in_steps):
	"""
	TODO: add docstring
	Args:
		data (np.ndarray): data array
		slice_in_steps (int): slice length in steps
	Returns:
		np.ndarray: sliced data
	"""
	slices_begin_indexes = range(0, len(data) + 1, slice_in_steps)
	splitted_per_slice = [data[beg:beg + slice_in_steps] for beg in slices_begin_indexes]
	# remove tails
	if len(splitted_per_slice[0]) != len(splitted_per_slice[-1]):
		del splitted_per_slice[-1]
	return np.array(splitted_per_slice)


def parse_filename(filename):
	"""
	Example filename: bio_E_PLT_13.5cms_40Hz_2pedal_0.1step

	Args:
		filename (str):

	Returns:

	"""
	meta = filename.split("_")
	source = meta[0]
	muscle = meta[1]
	mode = meta[2]
	speed = meta[3].replace("cms", "")
	rate = int(meta[4].replace("Hz", ""))
	pedal = meta[5].replace("pedal", "")
	stepsize = float(meta[6].replace("step.hdf5", ""))

	return source, muscle, mode, speed, rate, pedal, stepsize


def auto_prepare_data(folder, filename, dstep_to, debugging=False):
	"""
	ToDo add info
	Args:
		folder (str):
		filename (str):
		dstep_to (float):
	Returns:
		np.ndarray: prepared data for analysis
		int: EES frequency in Hz
	"""
	log.info(f"prepare {filename}")

	slices_number_dict = {
		("PLT", '21'): (6, 5),
		("PLT", '13.5'): (12, 5),
		("PLT", '6'): (30, 5),
		("TOE", '21'): (4, 4),
		("TOE", '13.5'): (8, 4),
		("AIR", '13.5'): (5, 4),
		("QPZ", '13.5'): (12, 5),
		("STR", '21'): (6, 5),
		("STR", '13.5'): (12, 5),
		("STR", '6'): (30, 5),
	}

	# extract common meta info from the filename
	source, muscle, mode, speed, ees_hz, pedal, dstep = parse_filename(filename)

	if ees_hz not in [5] + list(range(10, 101, 10)):
		raise Exception("EES frequency not in allowed list")

	if speed not in ("6", "13.5", "21"):
		raise Exception("Speed not in allowed list")

	if dstep not in (0.025, 0.1, 0.25):
		raise Exception("Step size not in allowed list")

	e_slices_number, f_slices_number = slices_number_dict[(mode, speed)]
	slice_in_ms = int(1000 / ees_hz)
	slice_in_steps = int(slice_in_ms / dstep_to)
	standard_slice_length_in_steps = int(25 / dstep)
	abs_filepath = f"{folder}/{filename}"

	# extract data of extensor
	if muscle == "E":
		full_size = int(e_slices_number * 25 / dstep_to)
		# extract dataset based on slice numbers (except biological data)
		if source == "bio":
			dataset = extract_data(abs_filepath)
		else:
			e_begin = 0
			e_end = e_begin + standard_slice_length_in_steps * e_slices_number
			# use native funcion for get needful data
			dataset = extract_data(abs_filepath, e_begin, e_end)
	# extract data of flexor
	elif muscle == "F":
		full_size = int(f_slices_number * 25 / dstep_to)
		# preapre flexor data
		if source == "bio":
			dataset = extract_data(abs_filepath)
		else:
			f_begin = standard_slice_length_in_steps * e_slices_number
			f_end = f_begin + (7 if pedal == "4" else 5) * standard_slice_length_in_steps
			# use native funcion for get needful data
			dataset = extract_data(abs_filepath, f_begin, f_end)
	# in another case
	else:
		raise Exception("Couldn't parse filename and extract muscle name")

	# subsampling data to the new data step
	dataset = subsampling(dataset, dstep_from=dstep, dstep_to=dstep_to)
	# prepare data and fill zeroes where not enough slices
	prepared_data = np.array([np.append(d, [0] * (full_size - len(d))) for d in prepare_data(dataset)])

	# mean
	# for d in np.array([split_by_slices(d, slice_in_steps) for d in prepared_data]):
	# 	for i, s in enumerate(d):
	# 		plt.plot(s + i * 0.2, color='gray')
	#
	# prepared_data = np.mean(prepared_data, axis=0)
	# splitted_per_slice = np.array([split_by_slices(prepared_data, slice_in_steps)])
	#
	# for i, d in enumerate(splitted_per_slice[0]):
	# 	plt.plot(d + i * 0.2, color='r', linewidth='2')
	# plt.xlim(0, len(splitted_per_slice[0][0]))
	# plt.show()

	# split datatest into the slices
	splitted_per_slice = np.array([split_by_slices(d, slice_in_steps) for d in prepared_data])

	return splitted_per_slice


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
