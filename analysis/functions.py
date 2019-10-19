import csv
import logging
import numpy as np
import h5py as hdf5
import pylab as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import kstwobign, pearsonr


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
	with hdf5.File(filepath, 'r') as file:
		data_by_test = [test_values[:] for test_values in file.values()]
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

	# check if all lengths are equal
	if len(set(map(len, subsampled_dataset))) <= 1:
		return np.array(subsampled_dataset)
	raise Exception("Length of slices not equal!")


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
	'''
	FixMe: temporary debugging
	a = read_data(path)[:, beg:end]
	print(beg, end)
	slice_length = int(int(1000 / 100) / 0.025)
	for k in a:
		plt.plot(range(beg, end), k)
	for i in range(beg, end + 1, slice_length):
		plt.axvline(x=i)
	plt.xticks(range(beg, end + 1, slice_length))
	plt.show()
	'''
	return read_data(path)[:, beg:end]


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


def auto_prepare_data(folder, filename, dstep_to=None):
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

	# map of cms <-> number of slices
	e_slices_number = {'6': 30, '15': 12, '13.5': 12, '21': 6}

	# extract common meta info from the filename
	ees_hz = int(filename[:filename.find('Hz')].split('_')[-1])
	if ees_hz not in [5] + list(range(10, 101, 10)):
		raise Exception("EES frequency not in allowed list")

	speed = filename[:filename.find('cms')].split('_')[-1]
	if speed not in ("6", "13.5", "15", "21"):
		raise Exception("Speed not in allowed list")

	step_size = float(filename[:filename.find('step')].split('_')[-1])
	if step_size not in (0.025, 0.1, 0.25):
		raise Exception("Step size not in allowed list")

	standard_slice_length_in_steps = int(25 / step_size)
	filepath = f"{folder}/{filename}"

	# extract data of extensor
	if '_E_' in filename:
		# extract dataset based on slice numbers (except biological data)
		if 'bio_' in filename:
			dataset = extract_data(filepath)
		else:
			e_begin = 0
			e_end = e_begin + standard_slice_length_in_steps * e_slices_number[speed]
			# use native funcion for get needful data
			dataset = extract_data(filepath, e_begin, e_end)
	# extract data of flexor
	elif '_F_' in filename:
		# preapre flexor data
		if 'bio_' in filename:
			dataset = extract_data(filepath)
		else:
			f_begin = standard_slice_length_in_steps * e_slices_number[speed]
			f_end = f_begin + (7 if '4pedal' in filename else 5) * standard_slice_length_in_steps
			# use native funcion for get needful data
			dataset = extract_data(filepath, f_begin, f_end)
	# in another case
	else:
		raise Exception("Couldn't parse filename and extract muscle name")
	# subsampling data to the new data step
	dataset = subsampling(dataset, dstep_from=step_size, dstep_to=dstep_to)
	#
	slice_in_ms = int(1000 / ees_hz)
	slice_in_steps = int(slice_in_ms / dstep_to)
	# split datatest into the slices
	splitted_per_slice = np.array([split_by_slices(d, slice_in_steps) for d in prepare_data(dataset)])

	return splitted_per_slice


def peacock2(xx, yy):
	n1 = xx.shape[0]
	n2 = yy.shape[0]
	n = n1 + n2

	# geatest common divisor: dd
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

	# SF - Survival function (also defined as 1 - cdf, but sf is sometimes more accurate)
	en = np.sqrt(n1 * n2 / (n1 + n2))
	p = kstwobign.sf(en * D)

	return D, p


def ks2d2s(x1, y1, x2, y2):
	'''Two-dimensional Kolmogorov-Smirnov test on two samples.
	Parameters
	----------
	x1, y1 : ndarray, shape (n1, )
		Data of sample 1.
	x2, y2 : ndarray, shape (n2, )
		Data of sample 2. Size of two samples can be different.
	extra: bool, optional
		If True, KS statistic is also returned. Default is False.
	Returns
	-------
	p : float
		Two-tailed p-value.
	D : float, optional
		KS statistic. Returned if keyword `extra` is True.
	Notes
	-----
	This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)
	References
	----------
	Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
	Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
	Press, W.H. et al. 2007, Numerical Recipes, section 14.8
	'''
	assert (len(x1) == len(y1)) and (len(x2) == len(y2))
	n1, n2 = len(x1), len(x2)
	D = avgmaxdist(x1, y1, x2, y2)

	sqen = np.sqrt(n1 * n2 / (n1 + n2))
	r1 = pearsonr(x1, y1)[0]    # get the linear correlation coefficient for each sample
	r2 = pearsonr(x2, y2)[0]
	r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
	d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
	p = kstwobign.sf(d)

	return D, p


def avgmaxdist(x1, y1, x2, y2):
	D1 = maxdist(x1, y1, x2, y2)
	D2 = maxdist(x2, y2, x1, y1)
	return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
	n1 = len(x1)
	D1 = np.empty((n1, 4))
	for i in range(n1):
		a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
		a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
		D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

	# re-assign the point to maximize difference,
	# the discrepancy is significant for N < ~50
	D1[:, 0] -= 1 / n1

	dmin, dmax = -D1.min(), D1.max() + 1 / n1
	return max(dmin, dmax)


def quadct(x, y, xx, yy):
	n = len(xx)
	ix1, ix2 = xx <= x, yy <= y
	a = np.sum(ix1 & ix2) / n
	b = np.sum(ix1 & ~ix2) / n
	c = np.sum(~ix1 & ix2) / n
	d = 1 - a - b - c
	return a, b, c, d
