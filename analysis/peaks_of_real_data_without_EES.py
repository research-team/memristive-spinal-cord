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




def calc_durations(slices_max_time, slices_min_time):
	"""

	Parameters
	----------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice without the time of EES
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice without the time of EES

	Returns
	-------
	duration_maxes: list
	    durations (difference between the last and the first time of max peak) in each slice
	duration_mins: list
		durations (difference between the last and the first time of min peak) in each slice
	"""
	duration_maxes = []
	duration_mins = []
	for index in range(1, len(slices_max_time)):
		duration_maxes.append(slices_max_time[index][-1] - slices_max_time[index][0])
		duration_mins.append(slices_min_time[index][-1] - slices_min_time[index][0])
	return duration_maxes, duration_mins





def delays(slices_max_time, slices_min_time):
	"""

	Parameters
	----------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice without the time of EES
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice without the time of EES


	Returns
	-------
	delays_maxes: list
		times of the first max peaks without EES in each slice
	delays_mins: list
		times of the first min peaks without EES in each slice

	"""
	delays_maxes = []
	delays_mins = []
	for index in range(1, len(slices_max_time)):
		delays_maxes.append(slices_max_time[index][0])
		delays_mins.append(slices_min_time[index][0])
	return delays_maxes, delays_mins


if __name__ == '__main__':
	init()
