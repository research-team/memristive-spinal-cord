import csv
import numpy as np
import h5py as hdf5
import pylab as plt
from analysis.namespaces import *
from sklearn.linear_model import LinearRegression
import statistics
import copy
from analysis.patterns_in_bio_data import bio_data_runs


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
	# print("slices_start_time = ", slices_start_time)
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


def find_latencies(mins_maxes, step, norm_to_ms=False, reversed_data=False, inhibition_zero=False, first_kink=False):
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
		slice_values = mins_maxes[1][slice_index]   # was 3
		for value in mins_maxes[3][slice_index]:
			slice_values.append(value)
		slice_times, slice_values = (list(x) for x in zip(*sorted(zip(slice_times, slice_values))))
		# while minimal value isn't found -- find with extended borders [left, right]

		while True:
			if inhibition_zero:
				left = 10 - additional_border   # 11
				right = 25 + additional_border
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
			# for f in found_points:
				# if slice_values[slice_values.index(f) > thresholds[count]]:
					# latencies.append(slice_times[slice_values.index(f)])

			# save index of the minimal element in founded points
			if len(found_points):
				# for f in range(len(found_points)):
			# 		if slice_values[slice_values.index(found_points[f])] > thresholds[count]:
						# latencies.append(slice_times[slice_values.index(found_points[f])])
						# count += 1
						# break
				# else:
				# 		f += 1
				# for i in range(len(found_points)):
					# if found_points[i] <= found_points[i + 1]:
						# minimal_val = found_points[i]
						# print("minimal_val = ", minimal_val)
						# break
				minimal_val = found_points[0] if first_kink else found_points[0] # found_points[0]
				index_of_minimal = slice_values.index(minimal_val)
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
	amplitudes = []
	slice_numbers = len(datas[0])
	dots_per_slice = 0
	if step == 0.25:
		dots_per_slice = 100
	if step == 0.025:
		dots_per_slice = 1000
	# for l in range(len(latencies)):
	# 	latencies[l] *= step
	# 	latencies[l] = int(latencies[l])
	# print("latencies = ", latencies)
	max_times = datas[0]
	max_values = datas[1]
	min_times = datas[2]
	min_values = datas[3]

	# print("amp max_times = ", max_times)
	# print("amp max_values = ", max_values)
	# print("amp min_times = ", min_times)
	# print("amp min_values = ", min_values)
	# print("max_values = (func)", max_values)
	# print("min_times = ", min_times)
	max_times_amp = []
	min_times_amp = []
	max_values_amp = []
	min_values_amp = []

	for i in range(len(latencies)):
		max_times_amp_tmp = []
		for j in range(len(max_times[i])):
			if max_times[i][j] > latencies[i]:
				max_times_amp_tmp.append(max_times[i][j])
		max_times_amp.append(max_times_amp_tmp)
		min_times_amp_tmp = []
		for j in range(len(min_times[i])):
			if min_times[i][j] > latencies[i]:
				min_times_amp_tmp.append(min_times[i][j])
		min_times_amp.append(min_times_amp_tmp)

		max_values_amp.append(max_values[i][len(max_times[i]) - len(max_times_amp[i]):])
		min_values_amp.append(min_values[i][len(min_times[i]) - len(min_times_amp[i]):])

	corrected_max_times_amp = []
	corrected_max_values_amp = []

	wrong_sl = []
	wrong_dot = []

	for index_sl, sl in enumerate(max_times_amp):
		corrected_max_times_amp_tmp = []
		for index_dot, dot in enumerate(sl):
			if dot < dots_per_slice:
				corrected_max_times_amp_tmp.append(dot)
			else:
				wrong_sl.append(index_sl)
				wrong_dot.append(index_dot)
		corrected_max_times_amp.append(corrected_max_times_amp_tmp)

	corrected_max_values_amp = max_values_amp
	for i in range(len(wrong_sl) - 1, -1, -1):
		del corrected_max_values_amp[wrong_sl[i]][wrong_dot[i]]

	corrected_min_times_amp = []
	corrected_min_values_amp = []
	wrong_sl = []
	wrong_dot = []

	for index_sl, sl in enumerate(min_times_amp):
		corrected_min_times_amp_tmp = []
		for index_dot, dot in enumerate(sl):
			if dot < dots_per_slice:
				corrected_min_times_amp_tmp.append(dot)
			else:
				wrong_sl.append(index_sl)
				wrong_dot.append(index_dot)
		corrected_min_times_amp.append(corrected_min_times_amp_tmp)

	corrected_min_values_amp = min_values_amp
	for i in range(len(wrong_sl) - 1, -1, -1):
		del corrected_min_values_amp[wrong_sl[i]][wrong_dot[i]]

	for sl in range(len(corrected_min_times_amp)):
		for dot in range(1, len(corrected_min_times_amp[sl])):
			if corrected_min_times_amp[sl][dot - 1] > corrected_min_times_amp[sl][dot]:
				corrected_min_times_amp[sl] = corrected_min_times_amp[sl][:dot]
				corrected_min_values_amp[sl] = corrected_min_values_amp[sl][:dot]

	for sl in range(len(corrected_max_times_amp)):
		for dot in range(1, len(corrected_max_times_amp[sl])):
			if corrected_max_times_amp[sl][dot - 1] > corrected_max_times_amp[sl][dot]:
				corrected_max_times_amp[sl] = corrected_max_times_amp[sl][:dot]
				corrected_max_values_amp[sl] = corrected_max_values_amp[sl][:dot]
				break

	peaks_number = []
	for sl in range(len(corrected_min_values_amp)):
		peaks_number.append(len(corrected_min_values_amp[sl]) + len(corrected_max_values_amp[sl]))
	amplitudes = []
	print("corrected_min_values_amp = ", corrected_min_values_amp)
	print("corrected_max_values_amp = ", corrected_max_values_amp)

	for sl in range(len(corrected_max_values_amp)):
		print("sl = ", sl)
		amplitudes_sl = []
		try:
			for i in range(len(corrected_max_values_amp[sl]) - 1):
				print("i = ", i)
				amplitudes_sl.append(corrected_max_values_amp[sl][i] - corrected_min_values_amp[sl][i])
				amplitudes_sl.append(corrected_max_values_amp[sl][i + 1] - corrected_min_values_amp[sl][i])
		except IndexError:
			continue

		amplitudes.append(amplitudes_sl)

	# for l in range(len(latencies)):
	# 	latencies[l] /= step
	return amplitudes, peaks_number, corrected_max_times_amp, corrected_max_values_amp, corrected_min_times_amp, \
	       corrected_min_values_amp


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


def __process(latencies, voltages, stim_indexes, step, debugging, inhibition_zero=True, reverse_ees=False,
              after_latencies=False, first_kink=False):
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
	ees_indexes = find_ees_indexes(stim_indexes, mins_maxes, reverse_ees=reverse_ees)
	norm_voltages = normalization(voltages, zero_relative=True)
	mins_maxes = calc_max_min(ees_indexes, voltages, stim_corr=stim_indexes)
	# latencies = find_latencies(mins_maxes, step, norm_to_ms=True, reversed_data=reversed_data,
	#                            inhibition_zero=inhibition_zero, first_kink=first_kink) # , thresholds
	amplitudes, peaks_number, max_times, min_times, max_values, min_values = \
		calc_amplitudes(mins_maxes, latencies, step, after_latencies)

	# if debugging:
	# 	debug(voltages, mins_maxes, stim_indexes, ees_indexes, latencies, amplitudes, step)
	print("amplitudes = ", amplitudes)
	print("peaks_number = ", peaks_number)
	print("max_times = ", max_times)
	print("min_times = ", min_times)
	print("max_values = ", max_values)
	print("min_values = ", min_values)
	return amplitudes, peaks_number, max_times, min_times, max_values, min_values


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


def sim_process(latencies, voltages, step, debugging=False, inhibition_zero=False, after_latencies=False,
                first_kink=False):
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
	amplitudes, peaks_number, max_times, min_times, max_values, min_values = \
		__process(latencies, voltages, stim_indexes, step, debugging, inhibition_zero=inhibition_zero,
		          after_latencies=after_latencies, first_kink=first_kink)
	# change the step
	return latencies, amplitudes, peaks_number, max_times, min_times, max_values, min_values


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


def find_min_diff(all_maxes, all_mins, step, first_kink, from_first_kink=False):
	ees_end = 12
	min_difference = []
	max_difference = []
	min_difference_indexes = []
	max_difference_indexes = []

	# find difference between all maxes and mins where we fill between
	diffs = []
	for slice in range(len(all_mins)):
		diffs_tmp = []
		for dot in range(len(all_mins[slice])):
			diffs_tmp.append(all_maxes[slice][dot] - all_mins[slice][dot])
		diffs.append(diffs_tmp)

	# find max and min differences and their indexes in each slice from the ees
	for slice in range(len(all_mins)):
		print("slice = ", slice)
		max_dif = max(diffs[slice][int(ees_end / step):])
		if diffs[slice].index(max_dif) >= int(ees_end / step):
			max_dif_index = diffs[slice].index(max_dif)
		else:
			del diffs[slice][diffs[slice].index(max_dif)]
			max_dif_index = diffs[slice].index(max_dif) + 1

		print("max_dif_index = ", max_dif_index)
		min_dif = min(diffs[slice][int(ees_end / step):max_dif_index])
		min_dif_index = diffs[slice].index(min_dif)

		min_difference.append(min_dif)
		min_difference_indexes.append(min_dif_index)

		max_difference.append(max_dif)
		max_difference_indexes.append(max_dif_index)

	thresholds = []
	for i in max_difference:
		thresholds.append(i * 0.0125)

	vars = []
# for slice in range(len(all_means)):
# 	for dot in range(min_difference_indexes[slice] + 1, len(all_mins[slice])):
# 		if all_maxes[slice][dot] - all_mins[slice][dot] - min_difference[slice] < var:
# 			var = all_maxes[slice][dot] - all_mins[slice][dot] - min_difference[slice]
# 	vars.append(var)
# print("vars = ", vars)
	necessary_values = []

	if from_first_kink:
		first_kink_indexes = [int(f * 4) for f in first_kink]
		for s in diffs:
			for i in range(len(thresholds)):
				necessary_values.append(s[first_kink_indexes[i]] + thresholds[i])
				break
		necessary_indexes = []
		print("len(all_mins) = ", len(all_mins))
		print("diffs[4] = ", len(diffs[4]), diffs[4][52:])
		print("necessary_values[4] = ", necessary_values[4])
		print("first_kink_indexes[4] = ", first_kink_indexes[4])
		for slice in range(len(all_mins)):
			for dot in range(first_kink_indexes[slice], len(all_mins[slice])):
				# print("diffs[{}][{}] = ".format(slice, dot), diffs[slice][dot])
				# print("necessary_values[{}] = ".format(slice), necessary_values[slice])
				if diffs[slice][dot] > necessary_values[slice]:
					vars.append(diffs[slice][dot])
					necessary_indexes.append(dot)
					# print("diffs[{}][{}] = ".format(slice, dot), diffs[slice][dot])
					# print("necessary_values[{}] = ".format(slice), necessary_values[slice])
					break
	else:
		for i in range(len(thresholds)):
			necessary_values.append(min_difference[i] + thresholds[i])
		necessary_indexes = []
		for slice in range(len(all_mins)):
			for dot in range(min_difference_indexes[slice], len(all_mins[slice])):
				if diffs[slice][dot] > necessary_values[slice]:
					vars.append(diffs[slice][dot])
					necessary_indexes.append(dot)
					break
	return min_difference_indexes, max_difference_indexes, necessary_indexes


def absolute_sum(data_list, step):
	all_bio_slices = []
	dots_per_slice = 0
	if step == 0.25:
		dots_per_slice = 100
	if step == 0.025:
		dots_per_slice = 1000
	# forming list for the plot
	for k in range(len(data_list)):
		bio_slices = []
		offset = 0
		for i in range(int(len(data_list[k]) / dots_per_slice)):
			bio_slices_tmp = []
			for j in range(offset, offset + dots_per_slice):
				bio_slices_tmp.append(data_list[k][j])
			bio_slices.append(normalization(bio_slices_tmp, -1, 1))
			offset += dots_per_slice
		all_bio_slices.append(bio_slices)  # list [4][16][100]
	all_bio_slices = list(zip(*all_bio_slices))  # list [16][4][100]

	instant_mean = []
	for slice in range(len(all_bio_slices)):
		instant_mean_sum = []
		for dot in range(len(all_bio_slices[slice][0])):
			instant_mean_tmp = []
			for run in range(len(all_bio_slices[slice])):
				instant_mean_tmp.append(abs(all_bio_slices[slice][run][dot]))
			instant_mean_sum.append(sum(instant_mean_tmp))
		instant_mean.append(instant_mean_sum)

	volts = []
	for i in instant_mean:
		for j in i:
			volts.append(j)

	return volts