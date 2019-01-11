import csv
import h5py as hdf5


def normalization(data, a=0, b=1, zero_relative=False):
	"""
	Normalization in [a, b] interval
	x` = (b - a) * (xi - min(x)) / (max(x) - min(x)) + a
	Args:
		data (list):
			data for normalization
		a (float or int):
			left interval a
		b (float or int):
			right interval b
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
		# prepare the constans
		min_x = min(data)
		max_x = max(data)
		const = (b - a) / (max_x - min_x)
		return [(x - min_x) * const + a for x in data]


def find_latencies(mins_maxes, step, norm_to_ms=False):
	"""
	Function for autonomous finding the latencies in slices by bordering and finding minimals
	Args:
		mins_maxes (list of list):
			0 max times by slices
			1 max values by slices
			2 min times by slices
			3 min values by slices
		step (float or int):
			step of data recording (e.g. step=0.025 means 40 recorders in 1 ms)
		norm_to_ms (bool):
			if True -- convert steps to ms, else return steps
	Returns:
		list: includes latencies for each slice
	"""
	latencies = []
	slice_numbers = len(mins_maxes[2])
	slice_indexes = range(slice_numbers)

	slices_index_interval = lambda a, b: slice_indexes[int(slice_numbers / 6 * a):int(slice_numbers / 6 * (b + 1))]
	step_to_ms = lambda current_step: current_step * step

	# find latencies per slice
	for slice_index in slice_indexes:
		additional_border = 0
		slice_times = mins_maxes[2][slice_index]
		slice_values = mins_maxes[3][slice_index]
		# while minimal value isn't found -- find with extended borders [left, right]
		while True:
			if slice_index in slices_index_interval(0, 1): # [0,1]
				left = 11 - additional_border
				right = 16 + additional_border
			elif slice_index in slices_index_interval(2, 2): # [2]
				left = 11 - additional_border
				right = 17 + additional_border
			elif slice_index in slices_index_interval(3, 4): # [3, 4]
				left = 13 - additional_border
				right = 21 + additional_border
			elif slice_index in slices_index_interval(5, 6): # [5, 6]
				left = 15 - additional_border
				right = 24 + additional_border
			else:
				raise Exception("Error in the slice index catching")

			if left < 0:
				left = 0
			if right > 25:
				right = 25

			found_points = [v for i, v in enumerate(slice_values) if left <= step_to_ms(slice_times[i]) <= right]

			# save index of the minimal element in founded points
			if len(found_points):
				minimal_val = min(found_points)
				index_of_minimal = slice_values.index(minimal_val)
				latencies.append(slice_times[index_of_minimal])
				break
			else:
				additional_border += 1

			if additional_border > 25:
				print("KEK", slice_index, left, right, slice_times)
				print(mins_maxes[2:])
				latencies.append(-10)
				break

	# checking on errors
	if len(latencies) != slice_numbers:
		raise Exception("Latency list length is not equal to number of slices!")

	if norm_to_ms:
		return [lat * step for lat in latencies]
	return latencies


def find_mins(data_array, matching_criteria=None):
	"""
	Function for finding the minimal extrema in the data
	Args:
		data_array (list):
			data what is needed to find mins in
		matching_criteria (int or float or None):
			number less than which min peak should be to be considered as the start of a new slice
	Returns:
		list: min_elems -- values of the starts of new slice
		list: indexes -- indexes of the starts of new slice
	"""
	indexes = []
	min_elems = []
	# FixMe taken from the old function find_mins_without_criteria. Why -0.5 (?)
	if matching_criteria is None:
		matching_criteria = -0.5
	for index_elem in range(1, len(data_array) - 1):
		if (data_array[index_elem - 1] > data_array[index_elem] <= data_array[index_elem + 1]) \
				and data_array[index_elem] < matching_criteria:
			min_elems.append(data_array[index_elem])
			indexes.append(index_elem)
	return min_elems, indexes


def read_neuron_data(path):
	"""
	Reading hdf5 data for Neuron simulator
	Args:
		path (str):
			path to the file
	Returns:
		list: data from the file
	"""
	with hdf5.File(path) as file:
		neuron_means = [data[:] for data in file.values()]
	return neuron_means

def read_nest_data(path):
	"""
	FixMe merge with read_neuron_data (!),
	 For Alex: use a negative voltage data writing (like as extracellular)
	Reading hdf5 data for NEST simulator
	Args:
		path (str):
			path to the file
	Returns:
		list: data from the file
	"""
	nest_data = []
	with hdf5.File(path) as file:
		for test_data in file.values():
			first = -test_data[0]
			# transforming to extracellular form
			nest_data.append([-d - first for d in test_data[:]])
	return nest_data


def read_bio_data(path):
	with open(path) as file:
		# skipping headers of the file
		for i in range(6):
			file.readline()
		reader = csv.reader(file, delimiter='\t')
		# group elements by column (zipping)
		grouped_elements_by_column = list(zip(*reader))
		# avoid of NaN data
		raw_data_RMG = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[2]]
		data_stim = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[7]]
	# preprocessing: finding minimal extrema an their indexes
	mins, indexes = find_mins(data_stim)
	# remove raw data before the first EES and after the last (slicing)
	data_RMG = raw_data_RMG[indexes[0]:indexes[-1]]
	# shift indexes to be normalized with data RMG (because a data was sliced) by value of the first EES
	shifted_indexes = [d - indexes[0] for d in indexes]

	return data_RMG, shifted_indexes
