import csv
import numpy as np
import h5py as hdf5
import pylab as plt
from analysis.namespaces import *
from sklearn.linear_model import LinearRegression
import statistics
import copy
#from analysis.patterns_in_bio_data import bio_data_runs


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
		# prepare the constans
		first = data[0]
		minimal = abs(min(data))

		return [(volt - first) / minimal for volt in data]
	else:
		min_x = min(data)
		max_x = max(data)
		const = (b - a) / (max_x - min_x)

		return [(x - min_x) * const + a for x in data]


def calc_linear(x, y):
	model = LinearRegression(fit_intercept=True)
	model.fit(x[:, np.newaxis], y)
	xfit = np.linspace(0, 25, 10)
	yfit = model.predict(X=xfit[:, np.newaxis])

	return xfit, yfit


def calc_max_min(slices_start_time, test_data, remove_micropeaks=False, stim_corr=None, find_EES=False):
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
		stim_corr (list):
			EES stimulation indexes for correction the time of found min/max points (to be relative from EES stim)
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
		border = len(sliced_values) / 3 if find_EES else len(sliced_values)
		datas_times = range(end - start)
		# compare points
		for i in range(1, len(sliced_values) - 1):
			if sliced_values[i - 1] < sliced_values[i] >= sliced_values[i + 1] and i < border:
				tmp_max_time.append(datas_times[i] + offset)
				tmp_max_value.append(sliced_values[i])
			if sliced_values[i - 1] > sliced_values[i] <= sliced_values[i + 1] and i < border:
				tmp_min_time.append(datas_times[i] + offset)
				tmp_min_value.append(sliced_values[i])
			if not tmp_max_time or not tmp_max_value or not tmp_min_time or not tmp_min_value:
				border += 1

		# append found points per slice to the 'main' lists
		slices_max_time.append(tmp_max_time)
		slices_max_value.append(tmp_max_value)
		slices_min_time.append(tmp_min_time)
		slices_min_value.append(tmp_min_value)

	# FixMe remove this functionality in future
	if remove_micropeaks:
		raise Warning("This functionality is deprecated and will be removed soon")
		# small realization of ommiting data marked as False
		remove_micropeaks_func = lambda datas, booleans: [data for data, boolean in zip(datas, booleans) if boolean]

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
				if abs(mins_time[min_i + 1] - mins_time[min_i]) > (1 / sim_step):
					mins_bool[min_i] = True
				if abs(maxes_time[max_i + 1] - maxes_time[max_i]) > (1 / sim_step):
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


def find_latencies(mins_maxes, step, norm_to_ms=False, reversed_data=False, inhibition_zero=False): # , thresholds
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
	times_latencies = []
	slice_numbers = len(mins_maxes[0])
	slice_indexes = range(slice_numbers)

	slices_index_interval = lambda a, b: slice_indexes[int(slice_numbers / 6 * a):int(slice_numbers / 6 * (b + 1))]
	step_to_ms = lambda current_step: current_step * step
	count = 0
	# find latencies per slice
	for slice_index in slice_indexes:
		additional_border = 0
		slice_times = mins_maxes[0][slice_index]    # was 2
		for time in mins_maxes[2][slice_index]:
			slice_times.append(time)
		# print("slice_times = ", slice_times)
		slice_values = mins_maxes[1][slice_index]   # was 3
		for value in mins_maxes[3][slice_index]:
			slice_values.append(value)
		# print("slice_values = ", slice_values)
		slice_times, slice_values = (list(x) for x in zip(*sorted(zip(slice_times, slice_values))))
		# print("slice_times = ", [s / 4 for s in slice_times])
		# print("slice_values = ", slice_values)
		# while minimal value isn't found -- find with extended borders [left, right]

		while True:
			if inhibition_zero:
				left = 11 - additional_border   # 11
				right = 25 + additional_border
				# print('inhibition zero')
			else:
				# raise Exception
				if slice_index in slices_index_interval(0, 1): # [0,1]
					if reversed_data:
						left = 15 - additional_border
						right = 24 + additional_border
					else:
						left = 12 - additional_border
						right = 18 + additional_border
				elif slice_index in slices_index_interval(2, 2): # [2]
					if reversed_data:
						left = 13 - additional_border
						right = 21 + additional_border
					else:
						left = 15 - additional_border
						right = 17 + additional_border
				elif slice_index in slices_index_interval(3, 4): # [3, 4]
					if reversed_data:
						left = 10 - additional_border
						right = 17 + additional_border
					else:
						left = 15 - additional_border
						right = 21 + additional_border
				elif slice_index in slices_index_interval(5, 6): # [5, 6]
					if reversed_data:
						left = 11 - additional_border
						right = 16 + additional_border
					else:
						left = 13 - additional_border
						right = 24 + additional_border
				else:
					raise Exception("Error in the slice index catching")

			if left < 0:
				left = 0
			if right > 25:
				right = 25

			found_points = [v for i, v in enumerate(slice_values) if left <= step_to_ms(slice_times[i]) <= right]
			# print(slice_index, found_points)
			# print("left = ", left)
			# print("right = ", right)
			# print("found_points = ", found_points)
			# for f in found_points:
				# print("count = ", count)
				# print("found_points values = ", slice_values[slice_values.index(f)])
				# print("found_points times = ", slice_times[slice_values.index(f)] * 0.025)
				# print()
				# if slice_values[slice_values.index(f) > thresholds[count]]:
					# latencies.append(slice_times[slice_values.index(f)])

			# print("---")
			# save index of the minimal element in founded points
			if len(found_points):
				# print("found_points = ", found_points)
				# for f in range(len(found_points)):
			# 		if slice_values[slice_values.index(found_points[f])] > thresholds[count]:
			# 			print("slice_values.index(found_points[{}]) = ".format(f), slice_values.index(found_points[f]))
						# print("slice_times.index(found_points[{}]) = ".format(f), slice_times.index(found_points[f]))
						# print("thresholds[{}] = ".format(count), thresholds[count])
						# latencies.append(slice_times[slice_values.index(found_points[f])])
						# print("latencies = ", latencies)
						# count += 1
						# print("count = ", count)
						# break
				# else:
				# 		f += 1
				# for i in range(len(found_points)):
					# print("found_points = ", slice_times[slice_values.index(found_points[i])] * 0.25)
					# if found_points[i] <= found_points[i + 1]:
						# minimal_val = found_points[i]
						# print("i = ", i)
						# print("minimal_val = ", minimal_val)
						# break
				# print("found_points = ", found_points)
				minimal_val = found_points[0] if inhibition_zero else max(found_points) # found_points[0]
				index_of_minimal = slice_values.index(minimal_val)
				# print("slice_values[{}] = ".format(index_of_minimal), slice_values[index_of_minimal])
				latencies.append(slice_times[index_of_minimal])
				break

			else:
				additional_border += 1
			if additional_border > 25:
				# FixMe
				latencies.append(-999)
				break
				# FixMe raise Exception("Error, out of borders")
	# checking on errors
	if len(latencies) != slice_numbers:
		raise Exception("Latency list length is not equal to number of slices!")

	if norm_to_ms:
		return [lat * step for lat in latencies]
	return latencies


def find_ees_indexes(stim_indexes, datas, reverse_ees=False):
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

	if reverse_ees:
		for slice_index in range(len(stim_indexes)):
			max_values = datas[k_max_val][slice_index]
			max_times = datas[k_max_time][slice_index]
			# EES peak is the minimal one
			ees_value_index = max_values.index(max(max_values))
			# calculate the EES answer as the sum of the local time of the found EES peak (of the layer)
			# and global time of stimulation for this layer
			ees_indexes.append(stim_indexes[slice_index] + max_times[ees_value_index])
	else:
		for slice_index in range(len(stim_indexes)):
			min_values = datas[k_min_val][slice_index]
			min_times = datas[k_min_time][slice_index]
			# EES peak is the minimal one
			ees_value_index = min_values.index(min(min_values))
			# calculate the EES answer as the sum of the local time of the found EES peak (of the layer)
			# and global time of stimulation for this layer
			ees_indexes.append(stim_indexes[slice_index] + min_times[ees_value_index])
	return ees_indexes


def calc_amplitudes(datas, latencies, step, after_latencies=False):
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
	# for data in datas:
		# print("datas = ", len(data), data)
	amplitudes = []
	slice_numbers = len(datas[0])

	# print("latencies = ", latencies)

	for l in range(len(latencies)):
		latencies[l] /= step
		latencies[l] = int(latencies[l])

	# print("---")
	# print("latencies = ", latencies)
	# print("---")

	max_times = datas[0]
	max_values = datas[1]
	min_times = datas[2]
	min_values = datas[3]

	# print("max_times = ", len(max_times[0]), max_times)
	# print("max_values = ", len(max_values[0]), max_values)
	# print("min_times = ", len(min_times[0]), min_times)
	# print("min_values = ", len(min_values[0]), min_values)

	if after_latencies:
		for l in latencies:
			to_delete = []
			for s in range(len(max_times)):
				# print("len(max_times[{}]) = ".format(s), len(max_times[s]))
				# print("max_times[{}] = ".format(s), max_times[s])
				to_delete_slice = []
				for d in range(len(max_times[s])):
					if max_times[s][d] < l:
						to_delete_slice.append(max_times[s][d])

		to_delete_value_max = []
		for slice in range(len(to_delete)):
			to_delete_value_max_tmp = []

			# print("slice = ", slice)
			# print("to_delete[{}] = ".format(slice), len(to_delete[slice]), to_delete[slice])
			for slice_of_all_data in range(len(max_times[slice]) - 1, -1, -1):
				for dot in to_delete[slice]:
					# print("max_times[{}][{}] = ".format(slice, slice_of_all_data), max_times[slice][slice_of_all_data])
					# print("dot = ", dot)
					if max_times[slice][slice_of_all_data] == dot:
						del max_times[slice][slice_of_all_data]
						to_delete_value_max_tmp.append(slice_of_all_data)
				to_delete_value_max.append(to_delete_value_max_tmp)

		for br in range(len(to_delete_value_max) - 1, 0, -1):
			# print("to_delete_value_max[{}] = ".format(br), to_delete_value_max[br])
			# print("to_delete_value_max[{}] = ".format(br - 1), to_delete_value_max[br - 1])
			if to_delete_value_max[br] == to_delete_value_max[br - 1]:
				del to_delete_value_max[br]
		# del max_values[slice][dot]
		# del min_times[slice][dot]
		# del min_values[slice][dot]
		# print("to_delete_value_max = ", to_delete_value_max)

		for s in range(len(to_delete_value_max)):
			for d in to_delete_value_max[s]:
				del max_values[s][d]
		for l in latencies:
			to_delete_mins = []
			for s in range(len(min_times)):
				to_delete_mins_slice = []
				for d in range(len(min_times[s])):
					if min_times[s][d] < l:
						to_delete_mins_slice.append(min_times[s][d])
				to_delete_mins.append(to_delete_mins_slice)

		to_delete_value_min = []
		for slice in range(len(to_delete_mins)):
			to_delete_value_min_tmp = []
			for slice_of_all_data in range(len(min_times[slice]) - 1, -1, -1):
				for dot in to_delete_mins[slice]:
					try:
						if min_times[slice][slice_of_all_data] == dot:
							del min_times[slice][slice_of_all_data]
							to_delete_value_min_tmp.append(slice_of_all_data)
					except IndexError:
						continue
				to_delete_value_min.append(to_delete_value_min_tmp)
		for br in range(len(to_delete_value_min) - 1, 0, -1):
			# print("to_delete_value_max[{}] = ".format(br), to_delete_value_max[br])
			# print("to_delete_value_max[{}] = ".format(br - 1), to_delete_value_max[br - 1])
			if to_delete_value_min[br] == to_delete_value_min[br - 1]:
				del to_delete_value_min[br]
		# del max_values[slice][dot]
		# del min_times[slice][dot]
		# del min_values[slice][dot]
		# print("to_delete_value_min = ", to_delete_value_min)

		for s in range(len(to_delete_value_min)):
			for d in to_delete_value_min[s]:
				del min_values[s][d]

	# print("---")
	# print("max_times = ", len(max_times[0]), max_times)
	# print("max_values = ", len(max_values[0]), max_values)
	# print("min_times = ", len(min_times[0]), min_times)
	# print("min_values = ", len(min_values[0]), min_values)
	# print("---")

	amplitudes = []
	for sl in range(len(max_values)):
		amplitudes_slice = 0
		for i in range(len(min_values[sl]) - 1, -1, -1):
			amplitudes_slice += max_values[sl][i] - min_values[sl][i]
		amplitudes.append(abs(amplitudes_slice))

	# for slice_index in range(slice_numbers):
	# 	mins_v = datas[k_min_val][slice_index]
	# 	mins_t = datas[k_min_time][slice_index]

		# if mins_v:
		# 	max_amp_in_mins = max([abs(m) for index, m in enumerate(mins_v) if mins_t[index] >= latencies[slice_index]])
		# 	amplitudes.append(max_amp_in_mins)
		# else:
			# FixMe
			# ToDo write a test for checking linear dot concentration impact
			# (what is better -- remove dot or set data of previous neighbor)
			# amplitudes.append(-999)

	# if len(amplitudes) != slice_numbers:
	# 	raise Exception("Length of amplitudes must be equal to slice numbers!")

	# for l in range(len(latencies)):
	# 	latencies[l] *= step

	for l in range(len(latencies)):
		latencies[l] *= step

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
		plt.bar([ees_index + ees_indexes[0] for ees_index in ees_indexes], amplitudes, width=5, color=color_lat,
		        alpha=0.7, zorder=2)
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


def __process(voltages, stim_indexes, step, debugging, reversed_data=False, inhibition_zero=True, reverse_ees=False,
              after_latencies=False):
	"""
	Unified functionality for finding latencies and amplitudes
	Args:
		voltages (list):
			voltage data
		stim_indexes (list):
			EES stimulations indexes
		step (float):
			step size of data
		debugging (bool):
			If True -- plot all found variables for debugging them
	Returns:
		list: latencies -- latency per slice
		list: amplitudes -- amplitude per slice
	"""
	mins_maxes = calc_max_min(stim_indexes, voltages, find_EES=True)   # check
	# print("len(mins_maxes[0][0]) = ", len(mins_maxes[0][0]))
	# print("len(mins_maxes[1][0]) = ", len(mins_maxes[1][0]))
	# print("len(mins_maxes[2][0]) = ", len(mins_maxes[2][0]))
	# print("len(mins_maxes[3][0]) = ", len(mins_maxes[3][0]))
	ees_indexes = find_ees_indexes(stim_indexes, mins_maxes, reverse_ees=reverse_ees)
	# norm_voltages = normalization(voltages, zero_relative=True)
	mins_maxes = calc_max_min(ees_indexes, voltages, stim_corr=stim_indexes)
	latencies = find_latencies(mins_maxes, step, norm_to_ms=True, reversed_data=reversed_data,
	                           inhibition_zero=inhibition_zero) # , thresholds
	amplitudes = calc_amplitudes(mins_maxes, latencies, step, after_latencies)

	if debugging:
		debug(voltages, mins_maxes, stim_indexes, ees_indexes, latencies, amplitudes, step)

	return latencies, amplitudes


def bio_process(voltages_and_stim, slice_numbers, debugging=False, reversed_data=False, reverse_ees=False):
	"""
	Find latencies in EES mono-answer borders and amplitudes relative from zero
	Args:
		voltages_and_stim (list):
			 voltages data and EES stim indexes
		slice_numbers (int):
			number of slices which we need to use in comparing with simulation data
		debugging (bool):
			If True -- plot all found variables for debugging them
	Returns:
		tuple: bio_lat, bio_amp -- latencies and amplitudes per slice
	"""
	# form EES stimulations indexes (use only from 0 to slice_numbers + 1)
	stim_indexes = voltages_and_stim[k_bio_stim][:slice_numbers + 1]    # +1
	# remove unescesary voltage data by the last EES stimulation index
	voltages = voltages_and_stim[k_bio_volt][:stim_indexes[-1]]
	volts_by_stims = []
	thresholds = []
	offset = 0
	for i in range(int(len(voltages) / 100)):
		volts_by_stims_tmp = []
		for j in range(offset, offset + 100):
			volts_by_stims_tmp.append(voltages[j])
		volts_by_stims.append(volts_by_stims_tmp)
		offset += 100
	for v in volts_by_stims:
		thresholds.append(0.137 * max(v))
	stim_indexes = stim_indexes[:-1]
	# calculate the latencies and amplitudes
	bio_lat, bio_amp = __process(voltages, stim_indexes, bio_step, debugging, reversed_data=reversed_data,
	                             reverse_ees=reverse_ees)

	return bio_lat, bio_amp


def sim_process(voltages, step, debugging=False, inhibition_zero=False, after_latencies=False):
	"""
	Find latencies in EES mono-answer borders and amplitudes relative from zero
	Args:
		voltages (list):
			 voltages data
		debugging (bool):
			If True -- plot all found variables for debugging them
	Returns:
		tuple: sim_lat, sim_amp -- latencies and amplitudes per slice
	"""
	# form EES stimulations indexes (in simulators begin from 0)
	stim_indexes = list(range(0, len(voltages), int(25 / step)))
	# calculate the latencies and amplitudes
	sim_lat, sim_amp = __process(voltages, stim_indexes, step, debugging, inhibition_zero=inhibition_zero,
	                             after_latencies=after_latencies)
	# change the step

	return sim_lat, sim_amp


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


def read_bio_hdf5(path):
	voltages = []
	stimulations = []

	with hdf5.File(path) as file:
		for title, values in file.items():
			if 'Stim' == title:
				stimulations = list(values[:])
			elif 'RMG' == title:
				voltages = list(values[:])
			else:
				raise Exception("Out of the itles border")

	return voltages, stimulations


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


def convert_bio_to_hdf5(voltages, stimulations, filename, path=None):
	with hdf5.File('{}{}.hdf5'.format(path + "/" if path else "", filename), 'w') as file:
		file.create_dataset('Stim', data=stimulations, compression="gzip")
		file.create_dataset('RMG', data=voltages, compression="gzip")


def find_fliers(amplitudes_all_runs, latencies_all_runs):
	"""
	Function for finding the fliers of data
	Args:
		amplitudes_all_runs: list of lists
			amplitudes in all runs
		latencies_all_runs: list of lists
			latencies in all runs
	Returns:
		 latencies_all_runs: list of lists
		    latencies of all runs without fliers
		 amplitudes_all_runs: list of lists
		    amplitudes of all runs without fliers
		 fliers: list of lists
		    indexes of fliers that were deleted from the upper lists
		 fliers_latencies_values: list of lists
		    values of fliers in list of the latencies
		 fliers_amplitudes_values: list of lists
		    values of fliers in list of the amplitudes
	"""
	# calculating the expected value and std in the list
	expected_value_amp = []
	std_amp = []

	for dot in amplitudes_all_runs:
		expected_value_tmp = statistics.mean(dot)
		std_tmp = statistics.stdev(dot)
		expected_value_amp.append(expected_value_tmp)
		std_amp.append(std_tmp)

	expected_value_lat = []
	std_lat = []

	for dot in latencies_all_runs:
		expected_value_tmp = statistics.mean(dot)
		std_tmp = statistics.stdev(dot)
		expected_value_lat.append(expected_value_tmp)
		std_lat.append(std_tmp)

	# checking if the value is in the 3-sigma interval
	amplitudes_all_runs_3sigma = []
	latencies_all_runs_3sigma = []

	# finding the fliers (values which are outside the 3-sigma interval)
	fliers_amplitudes = []
	fliers_amplitudes_values = []
	fliers_latencies = []
	fliers_latencies_values = []
	for dot in range(len(amplitudes_all_runs)):
		amplitudes_all_runs_dot_3sigma_amp = []
		fliers_amplitudes_tmp = []
		for i in range(len(amplitudes_all_runs[dot])):
			if (expected_value_amp[dot] - 3 * std_amp[dot]) < amplitudes_all_runs[dot][i] < \
					(expected_value_amp[dot] + 3 * std_amp[dot]):
				amplitudes_all_runs_dot_3sigma_amp.append(amplitudes_all_runs[dot][i])
			else:
				fliers_amplitudes_tmp.append(i)
		fliers_amplitudes.append(fliers_amplitudes_tmp)
		amplitudes_all_runs_3sigma.append(amplitudes_all_runs_dot_3sigma_amp)

	for dot in range(len(latencies_all_runs)):
		latencies_all_runs_dot_3sigma = []
		fliers_latencies_tmp = []
		for i in range(len(latencies_all_runs[dot])):
			if (expected_value_lat[dot] - 3 * std_lat[dot]) < latencies_all_runs[dot][i] < \
					(expected_value_lat[dot] + 3 * std_lat[dot]):
				latencies_all_runs_dot_3sigma.append(latencies_all_runs[dot][i])
			else:
				fliers_latencies_tmp.append(i)
		fliers_latencies.append(fliers_latencies_tmp)
		latencies_all_runs_3sigma.append(latencies_all_runs_dot_3sigma)

	# gathering the indexes of amplitudes' and latencies' fliers into one list
	fliers = fliers_amplitudes
	for sl in range(len(fliers)):
		for i in fliers_latencies[sl]:
			if i:
				if i not in fliers[sl]:
					fliers[sl].append(i)
		# sorting the lists in the ascending order
		fliers[sl] = sorted(fliers[sl])

	# saving the old lists of latencies and amplitudes
	old_latencies_all_runs = copy.deepcopy(latencies_all_runs)
	old_amplitudes_all_runs = copy.deepcopy(amplitudes_all_runs)

	# finding the values of the fliers
	for dot in range(len(fliers)):
		fliers_latencies_values_tmp = []
		for i in fliers[dot]:
			fliers_latencies_values_tmp.append(old_latencies_all_runs[dot][i])
		fliers_latencies_values.append(fliers_latencies_values_tmp)
	for dot in range(len(fliers)):
		fliers_amplitudes_values_tmp = []
		for i in fliers[dot]:
			fliers_amplitudes_values_tmp.append(old_amplitudes_all_runs[dot][i])
		fliers_amplitudes_values.append(fliers_amplitudes_values_tmp)

	# deleting the fliers in the latencies and amplitudes lists by the found indexes
	for sl in range(len(fliers)):
		for fl in reversed(fliers[sl]):
			if fl:
				del latencies_all_runs[sl][fl]
				del amplitudes_all_runs[sl][fl]

	return latencies_all_runs, amplitudes_all_runs, fliers, fliers_latencies_values, fliers_amplitudes_values


def bio_slices():
	bio_data = bio_data_runs()
	bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_data)))
	bio_mean_data = normalization(bio_mean_data, zero_relative=True)
	offset = 0
	bio_slices = []
	for i in range(int(len(bio_mean_data) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio_mean_data[j])
		offset += 100
		bio_slices.append(bio_slices_tmp)
	return bio_slices


def rotate(A, B, C):
	"""
	Function that determines what side of the vector AB is point C
	(positive returning value corresponds to the left side, negative -- to the right)
	Args:
		A: A coordinate of point
		B: B coordinate of point
		C: C coordinate of point

	Returns:

	"""
	return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])


def grahamscan(A):
	"""

	Args:
		A: list
			coordinates of dots in cloud

	Returns:
		list
			coordinates of dots of convex clouds

	"""
	n = len(A)
	P = []
	for i in range(n):
		P.append(i)
	for i in range(1, n):
		if A[P[i]][0] < A[P[0]][0]:
			P[i], P[0] = P[0], P[i]
	for i in range(2, n):
		j = i
		while j > 1 and (rotate(A[P[0]], A[P[j - 1]], A[P[j]]) < 0):
			P[j], P[j - 1] = P[j - 1], P[j]
			j -= 1
	S = [P[0], P[1]]
	for i in range(2, n):
		while rotate(A[S[-2]], A[S[-1]], A[P[i]]) < 0:
			del S[-1]
		S.append(P[i])
	return S


def find_min_diff(all_maxes, all_mins, step):
	ees_end = 6
	min_difference = []
	max_difference = []
	min_difference_indexes = []
	max_difference_indexes = []

	diffs = []
	for slice in range(len(all_mins)):
		diffs_tmp = []
		for dot in range(len(all_mins[slice])):
			diffs_tmp.append(all_maxes[slice][dot] - all_mins[slice][dot])
		diffs.append(diffs_tmp)

	for slice in range(len(all_mins)):
		# for dot in range(int(6 / step), len(all_mins[slice])):
		# 	print("dot = ", dot)
		max_dif = max(diffs[slice][int(ees_end / step):])
		max_dif_index = diffs[slice].index(max_dif)
		min_dif = min(diffs[slice][int(ees_end / step):max_dif_index])
		min_dif_index = diffs[slice].index(min_dif)

		min_difference.append(min_dif)
		min_difference_indexes.append(min_dif_index)

		max_difference.append(max_dif)
		max_difference_indexes.append(max_dif_index)

		print("min_difference_indexes = ", min_difference_indexes)
		print("min_difference = ", min_difference)

	thresholds = []
	for i in max_difference:
		thresholds.append(i * 0.1)

	vars = []
# for slice in range(len(all_means)):
# 	for dot in range(min_difference_indexes[slice] + 1, len(all_mins[slice])):
# 		if all_maxes[slice][dot] - all_mins[slice][dot] - min_difference[slice] < var:
# 			var = all_maxes[slice][dot] - all_mins[slice][dot] - min_difference[slice]
# 	vars.append(var)
# print("vars = ", vars)
	necessary_values = []
	for i in range(len(thresholds)):
		necessary_values.append(min_difference[i] + thresholds[i])
	print("necessary_values = ", necessary_values)
	necessary_indexes = []
	for slice in range(len(all_mins)):
		for dot in range(min_difference_indexes[slice], len(all_mins[slice])):
			if diffs[slice][dot] > necessary_values[slice]:
				vars.append(diffs[slice][dot])
				necessary_indexes.append(dot)
				break
	print("necessary_indexes = ", necessary_indexes)
	return min_difference_indexes, max_difference_indexes, necessary_indexes

