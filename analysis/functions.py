import csv
import h5py as hdf5
import pylab as plt
from analysis.namespaces import *


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
		zero_relative (bool):
			if True -- recalculate data where 0 is the first element and -1 is min(EES)
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


def calc_max_min(slices_start_time, test_data, step, remove_micropeaks=False, stim_corr=None):
	"""
	Function for finding min/max extrema
	Args:
		slices_start_time (list or range):
			list of slices start times
		test_data (list):
			list of data for processing
		step (float):
			step size of data recording
		remove_micropeaks (optional or bool):
			True - if need to remove micro peaks (<0.02 mV of normalized data)
	Returns:
		(list): slices_max_time
		(list): slices_max_value
		(list): slices_min_time
		(list): slices_min_value
	"""

	slices_max_time = []
	slices_max_value = []
	slices_min_time = []
	slices_min_value = []

	for slice_index in range(1, len(slices_start_time) + 1):
		tmp_max_time = []
		tmp_min_time = []
		tmp_max_value = []
		tmp_min_value = []

		if stim_corr:
			offset = slices_start_time[slice_index - 1] - stim_corr[slice_index - 1]
		else:
			offset = 0

		start = slices_start_time[slice_index - 1]
		if slice_index == len(slices_start_time):
			end = len(test_data)
		else:
			end = slices_start_time[slice_index]

		sliced_values = test_data[start:end]
		datas_times = range(end - start)
		# compare points
		for i in range(1, len(sliced_values) - 1):
			if sliced_values[i - 1] < sliced_values[i] >= sliced_values[i + 1]:
				tmp_max_time.append(datas_times[i] + offset)
				tmp_max_value.append(sliced_values[i])
			if sliced_values[i - 1] > sliced_values[i] <= sliced_values[i + 1]:
				tmp_min_time.append(datas_times[i] + offset)
				tmp_min_value.append(sliced_values[i])
		# append found points per slice to the 'main' lists
		slices_max_time.append(tmp_max_time)
		slices_max_value.append(tmp_max_value)
		slices_min_time.append(tmp_min_time)
		slices_min_value.append(tmp_min_value)

	# small realization of ommiting data marked as False
	remove_micropeaks_func = lambda datas, booleans: [data for data, boolean in zip(datas, booleans) if boolean]

	# realization of removing micro-peaks from the min/max points
	if remove_micropeaks:
		diff = 0.02 # the lowest difference between two points value which means micro-changing
		# per slice
		for slice_index in range(len(slices_min_value)):
			max_i = 0
			min_i = 0
			len_max = len(slices_max_time[slice_index])
			len_min = len(slices_min_time[slice_index])
			# init by bool the tmp lists for marking points
			maxes_bool = [True] * len_max
			mins_bool = [True] * len_min
			# just simplification
			maxes_val = slices_max_value[slice_index]
			mins_val = slices_min_value[slice_index]
			maxes_time = slices_max_time[slice_index]
			mins_time = slices_min_time[slice_index]

			while (max_i < len_max - 1) and (min_i < len_min - 1):
				# if points have small differnece mark them as False
				if abs(maxes_val[max_i] - mins_val[min_i]) < diff:
					maxes_bool[max_i] = False
					mins_bool[min_i] = False
				# but if the current points has the 3ms difference with the next point, remark the current as True
				if abs(mins_time[min_i + 1] - mins_time[min_i]) > (3 / step):
					mins_bool[min_i] = True
				if abs(maxes_time[max_i + 1] - maxes_time[max_i]) > (3 / step):
					maxes_bool[max_i] = True
				# change indexes (walking by pair: min-max, max-min, min-max...)
				if max_i == min_i:
					max_i += 1
				else:
					min_i += 1
			# ommit the data marked as False
			slices_max_value[slice_index] = remove_micropeaks_func(maxes_val, maxes_bool)
			slices_max_time[slice_index] = remove_micropeaks_func(maxes_time, maxes_bool)
			slices_min_value[slice_index] = remove_micropeaks_func(mins_val, mins_bool)
			slices_min_time[slice_index] = remove_micropeaks_func(mins_time, mins_bool)

	return slices_max_time, slices_max_value, slices_min_time, slices_min_value


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
	slice_numbers = len(mins_maxes[0])
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
				raise Exception("Error, out of borders")

	# checking on errors
	if len(latencies) != slice_numbers:
		raise Exception("Latency list length is not equal to number of slices!")

	if norm_to_ms:
		return [lat * step for lat in latencies]
	return latencies


def find_ees_indexes(stim_indexes, datas):
	"""
	Function for finding the indexes of the EES mono-answer in borders formed by stimulations time
	Args:
		stim_indexes (list):
			indexes of the EES stimulations
		datas (list of list):
			includes min/max times and min/max values
	Returns:
		list: global indexes of the EES mono-answers
	"""
	ees_indexes = []
	for slice_index in range(len(stim_indexes)):
		min_values = datas[k_min_val][slice_index]
		min_times = datas[k_min_time][slice_index]
		# EES peak is the minimal one
		ees_value_index = min_values.index(min(min_values))
		# calculate the EES answer as the sum of the local time of the found EES peak (of the layer)
		# and global time of stimulation for this layer
		ees_indexes.append(stim_indexes[slice_index] + min_times[ees_value_index])
	return ees_indexes


def calc_amplitudes(datas, latencies):
	"""
	Function for calculating amplitudes
	Args:
		datas (list of list):
			includes min/max time min/max value for each slice
		latencies (list):
			latencies pr slice for calculating only after the first poly-answer
	Returns:
		list: amplitudes per slice
	"""
	amplitudes = []
	slice_numbers = len(datas[0])

	for slice_index in range(slice_numbers):
		maxes_v = datas[k_max_val][slice_index]
		maxes_t = datas[k_max_time][slice_index]
		mins_v = datas[k_min_val][slice_index]
		mins_t = datas[k_min_time][slice_index]

		max_amp_in_maxes = max([abs(m) for index, m in enumerate(maxes_v) if maxes_t[index] >= latencies[slice_index]])
		max_amp_in_mins = max([abs(m) for index, m in enumerate(mins_v) if mins_t[index] >= latencies[slice_index]])

		amplitudes.append(max([max_amp_in_maxes, max_amp_in_mins]))

	if len(amplitudes) != slice_numbers:
		raise Exception("Length of amplitudes must be equal to slice numbers!")

	return amplitudes


def debug(voltages, datas, stim_indexes, ees_indexes, latencies, amplitudes, step):
	"""
	Temporal function for visualization of preprocessed data
	Args:
		voltages (list):
			voltage data
		datas (list of list):
			includes min/max time min/max value for each slice
		stim_indexes (list):
			indexes of EES stimlations
		ees_indexes (list):
			indexes of EES answers (mono-answer)
		latencies (list):
			latencies of the first poly-answers per slice
		amplitudes (list):
			amplitudes per slice
		step (float):
			 step size of the data
	"""
	amplitudes_y = []

	slice_indexes = range(len(ees_indexes))

	show_text = True
	show_amplitudes = True
	show_points = True
	show_axvlines = True

	# the 1st subplot demonstrates a voltage data, ees answers, ees stimulations and found latencies
	ax = plt.subplot(2, 1, 1)
	# plot the voltage data
	norm_voltages = normalization(voltages, zero_relative=True)

	plt.plot([t * step for t in range(len(norm_voltages))], norm_voltages, color='grey', linewidth=1)
	# standartization to the step size
	for slice_index in slice_indexes:
		datas[k_max_time][slice_index] = [d * step for d in datas[0][slice_index]]
	for slice_index in slice_indexes:
		datas[k_min_time][slice_index] = [d * step for d in datas[2][slice_index]]

	stim_indexes = [index * step for index in stim_indexes]
	ees_indexes = [index * step for index in ees_indexes]

	# plot the EES stimulation
	for i in stim_indexes:
		if show_axvlines:
			plt.axvline(x=i, linestyle='--', color='k', alpha=0.35, linewidth=1)
		if show_text:
			plt.annotate("Stim\n{:.2f}".format(i), xy=(i-1.3, 0), textcoords='data', color='k')
	# plot the EES answers
	for index, v in enumerate(ees_indexes):
		if show_axvlines:
			plt.axvline(x=v, color='r', alpha=0.5, linewidth=5)
		if show_text:
			plt.annotate("EES\n{:.2f}".format(v - stim_indexes[index]), xy=(v + 1, -0.2), textcoords='data', color='r')
	# plot the latencies
	for index, lat in enumerate(latencies):
		lat_x = stim_indexes[index] + lat
		if show_axvlines:
			plt.axvline(x=lat_x, color='g', alpha=0.7, linewidth=2)
		if show_text:
			plt.annotate("Lat: {:.2f}".format(lat), xy=(lat_x + 0.2, -0.4), textcoords='data', color='g')
	# plot min/max points for each slice and calculate their amplitudes
	plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
	plt.axhline(y=-1, color='r', linestyle='--', linewidth=1)

	for slice_index in slice_indexes:
		min_times = datas[k_min_time][slice_index]
		min_values = datas[k_min_val][slice_index]
		max_times = datas[k_max_time][slice_index]
		max_values = datas[k_max_val][slice_index]
		amplitudes_y.append(amplitudes[slice_index])
		# plot them
		if show_points:
			plt.plot([t + stim_indexes[slice_index] for t in min_times], min_values, '.', color='b', markersize=5)
			plt.plot([t + stim_indexes[slice_index] for t in max_times], max_values, '.', color='r', markersize=5)
	plt.legend()

	# plot the amplitudes with shared x-axis
	plt.subplot(2, 1, 2, sharex=ax)
	# plot the EES answers
	if show_amplitudes:
		for i in ees_indexes:
			plt.axvline(x=i, color='r')
		# plot amplitudes by the horizontal line
		plt.bar([ees_index + ees_indexes[0] for ees_index in ees_indexes], amplitudes, width=5, color=color_lat, alpha=0.7, zorder=2)
		for slice_index in slice_indexes:
			x = ees_indexes[slice_index] + ees_indexes[0] - 5 / 2
			y = amplitudes[slice_index]
			plt.annotate("{:.2f}".format(y), xy=(x, y + 0.01), textcoords='data')
		plt.ylim(0, 0.8)
	else:
		plt.plot([t * step for t in range(len(voltages))], voltages, color='grey', linewidth=1)
	plt.xlim(0, 150)
	plt.show()
	plt.close()



def bio_process(bio, slice_numbers, debugging=False):
	# get bio voltages and EES stimulations from the argument
	bio_stim_indexes = bio[k_bio_stim][:slice_numbers + 1]
	# remove unescesary data
	bio_voltages = bio[k_bio_volt][:bio_stim_indexes[-1]]
	# get the min/max extrema based on stimulation indexes
	bio_datas = calc_max_min(bio_stim_indexes, bio_voltages, bio_step)
	# find EES answers basing on min/max extrema
	bio_ees_indexes = find_ees_indexes(bio_stim_indexes[:-1], bio_datas)
	# normalize data
	norm_bio_voltages = normalization(bio_voltages, zero_relative=True)
	# get the min/max extrema based on EES answers indexes (because we need the data after 25ms of the slice)
	bio_datas = calc_max_min(bio_ees_indexes, norm_bio_voltages, bio_step, stim_corr=bio_stim_indexes)
	# get the latencies and amplitudes based on min/max extrema
	bio_lat = find_latencies(bio_datas, bio_step, norm_to_ms=True)
	bio_amp = calc_amplitudes(bio_datas, bio_lat)

	if debugging:
		debug(bio_voltages, bio_datas, bio_stim_indexes, bio_ees_indexes, bio_lat, bio_amp, bio_step)

	return bio_lat, bio_amp


def sim_process(data, debugging=False):
	sim_stim_indexes = list(range(0, len(data), int(25 / sim_step)))
	mins_maxes = calc_max_min(sim_stim_indexes, data, sim_step)
	sim_ees_indexes = find_ees_indexes(sim_stim_indexes, mins_maxes)
	norm_nest_means = normalization(data, zero_relative=True)
	mins_maxes = calc_max_min(sim_ees_indexes, norm_nest_means, sim_step, stim_corr=sim_stim_indexes)
	sim_lat = find_latencies(mins_maxes, sim_step, norm_to_ms=True)
	sim_amp = calc_amplitudes(mins_maxes, sim_lat)

	if debugging:
		debug(data, mins_maxes, sim_stim_indexes, sim_ees_indexes, sim_lat, sim_amp, sim_step)

	return sim_lat, sim_amp


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
