import csv
import logging
import numpy as np
import h5py as hdf5
import pylab as plt
from sklearn.decomposition import PCA


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


def read_data(filepath):
	"""
	ToDo add info
	Args:
		filepath:

	Returns:

	"""
	with hdf5.File(filepath, 'r') as file:
		data_by_test = [test_values[:] for test_values in file.values()]
		if not all(map(len, data_by_test)):
			raise Exception("hdf5 has an empty data!")
	return data_by_test

def find_mins(data_array): # matching_criteria was None
	"""
	Function for finding the minimal extrema in the data
	Args:
		data_array (list):
			data what is needed to find mins in
		matching_criteria (int or float or None):
			number less than which min peak should be to be considered as the start of a new slice
	Returns:
		tuple: min_elems -- values of the starts of new slice
		       indexes -- indexes of the starts of new slice
	"""
	indexes = []
	min_elems = []
	bio_step = 0.25

	# FixMe taken from the old function find_mins_without_criteria. Why -0.5 (?)
	ms_pause = 0
	data_array = [abs(d) for d in data_array]

	for index_elem in range(1, len(data_array) - 1):
		if (data_array[index_elem - 1] < data_array[index_elem] >= data_array[index_elem + 1]) \
				and ms_pause <= 0 \
				and data_array[index_elem] >= 0.2:
			min_elems.append(data_array[index_elem])
			indexes.append(index_elem)
			ms_pause = int(3 / bio_step)
		ms_pause -= 1

	return min_elems, indexes


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
		data_stim = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[5]]
	# preprocessing: finding minimal extrema an their indexes
	mins, indexes = find_mins(data_stim)
	# remove raw data before the first EES and after the last (slicing)
	data_RMG = raw_data_RMG[indexes[0]:indexes[-1]]
	# shift indexes to be normalized with data RMG (because a data was sliced) by value of the first EES
	shifted_indexes = [d - indexes[0] for d in indexes]

	return data_RMG, shifted_indexes


def extract_data(path, beg=None, end=None, step_from=None, step_to=None, ees_hz=None):
	"""
	ToDo add info
	Args:
		path:
		beg:
		end:
		step_from:
		step_to:

	Returns:
		np.ndarray: sliced data
	"""
	#
	if beg is None:
		beg = 0
	if end is None:
		end = int(10e6)
	# to convert
	if step_from == step_to or not step_from or not step_to:
		shrink_step = 1
	else:
		shrink_step = int(step_to / step_from)

	# slice data and shrink if need
	shrinked_data = [data[beg:end][::shrink_step] for data in read_data(path)]

	# check if all lengths are equal
	if len(set(map(len, shrinked_data))) <= 1:
		return np.array(shrinked_data)
	raise Exception("Length of slices not equal!")


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


def split_by_slices(dataset, slice_len_steps):
	"""
	TODO: add docstring
	Args:
		dataset (np.ndarray): dataset (N experimental runs)
		slice_len_steps (int): slice length in steps
	Returns:
		np.ndarray: sliced data
	"""
	sliced_dataset = []
	for data in dataset:
		slices_begin_indexes = range(0, len(data) + 1, slice_len_steps)
		splitted_per_slice = [data[beg:beg + slice_len_steps] for beg in slices_begin_indexes]
		# remove tail
		if len(splitted_per_slice[0]) != len(splitted_per_slice[-1]):
			del splitted_per_slice[-1]
		sliced_dataset.append(splitted_per_slice)

	return np.array(sliced_dataset)


def auto_prepare_data(folder, filename, step_size_to=None):
	"""
	ToDo add info
	Args:
		folder (str):
		filename (str):
		step_size_to (float):
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


	if "toe" in folder:
		pass

	filepath = f"{folder}/{filename}"
	slice_in_steps = int(1000 / ees_hz / step_size)
	wtf = int(25 / step_size)
	# extract data of extensor
	if '_E_' in filename:
		# extract dataset based on slice numbers (except biological data)
		if 'bio_' in filename:
			dataset = extract_data(filepath, step_from=step_size, step_to=step_size_to, ees_hz=ees_hz)
		else:
			e_begin = 0
			e_end = wtf * e_slices_number[speed]

			if "air" in folder and "13.5cms" in filename:
				e_end = wtf * 5
			if "toe" in folder and "13.5cms" in filename:
				e_end = wtf * 8
			if "toe" in folder and "21cms" in filename:
				e_end = wtf * 4

			# use native funcion for get needful data
			dataset = extract_data(filepath, e_begin, e_end, step_from=step_size, step_to=step_size_to, ees_hz=ees_hz)
	# extract data of flexor
	elif '_F_' in filename:
		# preapre flexor data
		if 'bio_' in filename:
			dataset = extract_data(filepath, step_from=step_size, step_to=step_size_to, ees_hz=ees_hz)
		else:
			f_begin = wtf * e_slices_number[speed]
			f_end = f_begin + (7 if '4pedal' in filename else 5) * wtf

			if any(d in folder for d in ("air", "toe")):
				f_end = f_begin + 4 * wtf

			# use native funcion for get needful data
			dataset = extract_data(filepath, f_begin, f_end, step_from=step_size, step_to=step_size_to, ees_hz=ees_hz)
	# in another case
	else:
		raise Exception("Couldn't parse filename and extract muscle name")

	# centering and normalizing data
	prepared_dataset = prepare_data(dataset)

	return prepared_dataset, ees_hz, step_size
