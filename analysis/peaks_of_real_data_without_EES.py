from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.real_data_slices import *


def init():
	raw_real_data = read_data('../bio-data//SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')

	myogram_data = slice_myogram(raw_real_data)
	slices_begin_time = myogram_data[1]
	slices_begin_time = [int(t / real_data_step) for t in slices_begin_time]
	volt_data = myogram_data[0]

	data = calc_max_min(slices_begin_time, volt_data, data_step=0.25)
	data_with_deleted_ees = remove_ees_from_min_max(data[0], data[1], data[2], data[3])

	durations = calc_durations(data_with_deleted_ees[0], data_with_deleted_ees[2])

	dels = delays(data_with_deleted_ees[0], data_with_deleted_ees[2])


def remove_ees_from_min_max(slices_max_time, slices_max_value, slices_min_time, slices_min_value):
	"""

	Parameters
	----------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice
	slices_max_value: dict
		key is index of slice, value is the list of max values in slice
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice
	slices_min_value: dict
		key is index of slice, value is the list of min values in slice

	Returns
	-------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice without the time of EES
	slices_max_value: dict
		key is index of slice, value is the list of max values in slice without the value of EES
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice without the time of EES
	slices_min_value: dict
		key is index of slice, value is the list of min values in slice without the value of EES

	"""
	for slice_index in range(1, len(slices_max_time)):
		del slices_max_time[slice_index][0]
		del slices_max_value[slice_index][0]
		del slices_min_time[slice_index][0]
		del slices_min_value[slice_index][0]
	return slices_max_time, slices_max_value, slices_min_time, slices_min_value


def calc_durations(slices_max_time, slices_min_time, slices_min_value, min_border, max_border, simulator):
	"""

	Parameters
	----------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice without the time of EES
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice without the time of EES
	slices_min_valus: dict
	    key is index of slice, value is the list of min values in slice
	min_border: int or float
	    number bigger that which the value should be to be considered as the min peak value
	max_border: int or float
	    number less that which the value should be to be considered as the min peak value
	simulator: string
	    nest or neuron

	Returns
	-------
	duration_maxes: list
		durations (difference between the last and the first time of max peak) in each slice
	duration_mins: list
		durations (difference between the last and the first time of min peak) in each slice
	"""
	duration_maxes = []
	duration_mins = []
	list_of_true_min_peaks = []
	tmp_list_of_true_min_peaks = []
	list_of_true_min_peak_times = []
	list_of_true_min_peak_times_tmp = []
	for index in slices_max_time.values():
		duration_maxes.append(round(index[-1] - index[0], 3))
	for index in slices_min_value.values():
		for i in range(len(index)):
			if min_border < index[i] < max_border:  # -15 < index[i] < -3.6 for 21 cm/s (neuron)
				#  -15 < index[i] < -4.4: for 15 cm/s (neuron)
				#  -1.5 * 10 ** (-9) < index[i] < -8 * 10 ** (-10) for 6 cm/s
				tmp_list_of_true_min_peaks.append(i)
		list_of_true_min_peaks.append(tmp_list_of_true_min_peaks.copy())
		tmp_list_of_true_min_peaks.clear()
	if simulator == 'neuron':
		del list_of_true_min_peaks[26][0]  # needed only for 6 cm/s
		#     del list_of_true_min_peaks[0][0]    # for neuron 21 cm/s
		#     del list_of_true_min_peaks[1][0]    # for neuron 21 cm/s
		#     del list_of_true_min_peaks[4][0]    # for neuron 21 cm/s
		#     del list_of_true_min_peaks[5][0]    # for neuron 21 cm/s
		#     del list_of_true_min_peaks[5][0]    # for neuron 21 cm/s
		# del list_of_true_min_peaks[5][0]    # for neuron 15cm/s
		# del list_of_true_min_peaks[11][0]   # for neuron 15cm/s
	for sl in range(len(list_of_true_min_peaks)):
		for key in slices_min_time:
			if key == sl + 1:
				for i in range(len(list_of_true_min_peaks[sl])):
					list_of_true_min_peak_times_tmp.append(
						round(slices_min_time[key][list_of_true_min_peaks[sl][i]], 3))
				list_of_true_min_peak_times.append(list_of_true_min_peak_times_tmp.copy())
				list_of_true_min_peak_times_tmp.clear()
	print("list_of_true_min_peak_times = ", list_of_true_min_peak_times)
	for sl in list_of_true_min_peak_times:
		duration_mins.append(round(sl[-1] - sl[0], 3))
	return duration_maxes, duration_mins


def delays(slices_max_time, slices_min_time, slices_min_value, min_border, simulator):
	"""
	todo write descrition

	Args:
		slices_max_time (dict):
			key is index of slice, value is the list of max times in slice
		slices_min_time (dict):
			key is index of slice, value is the list of min times in slice
		slices_min_value (dict):
		    key is index of slice, value is the list of min values in slice
		min_border (int or float):
		    number bigger that which the value should be to be considered as the min peak value
		min_border (int or float):
		    number less that which the value should be to be considered as the min peak value
		simulator (str):
		    nest or neuron

	Returns:
		list: delays_maxes
			 times of the first max peaks without EES in each slice
		list: delays_mins
			times of the first min peaks without EES in each slice
	"""
	delays_maxes = []
	delays_mins = []

	if simulator in ['nest', 'neuron']:
		sim_step = 0.025
	else:
		sim_step = 0.25

	offset_step = int(5 / sim_step)
	print("Offset:", offset_step * sim_step, "ms")

	for times_per_slice in slices_max_time.values():
		print("MAX times_per_slice", times_per_slice)
		for time in times_per_slice:
			if time > offset_step:
				delays_maxes.append(round(time, 3))
				break

	for slice_index, values_per_slice in slices_min_value.items():
		print("MIN values_per_slice", values_per_slice)
		for i in range(len(values_per_slice)):
			time = slices_min_time[slice_index][i]
			if values_per_slice[i] < min_border and time > offset_step:
				delays_mins.append(time)
				break

	print("delays_maxes", delays_maxes)
	print("delays_mins", delays_mins)

	return delays_maxes, delays_mins


if __name__ == '__main__':
	init()
