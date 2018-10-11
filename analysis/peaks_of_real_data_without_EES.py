from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.real_data_slices import *

raw_real_data = read_data('../bio-data//SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat')
myogram_data = slice_myogram(raw_real_data)
slices_begin_time = myogram_data[1]
slices_begin_time = [int(t / real_data_step) for t in slices_begin_time]
volt_data = myogram_data[0]

data = calc_max_min(slices_begin_time, volt_data, data_step = 0.25)

def remove_ees_from_min_max(slices_max_time, slices_max_value, slices_min_time, slices_min_value):
	"""

	Parameters
	----------
	dict slices_max_time: dict where key is index of slice, value is the list of max times in slice
	dict slices_max_value: dict where key is index of slice, value is the list of max values in slice
	dict slices_min_time: dict where key is index of slice, value is the list of min times in slice
	dict slices_min_value: dict where key is index of slice, value is the list of min values in slice

	Returns
	-------
	dict slices_max_time: dict where key is index of slice, value is the list of max times in slice without the time of EES
	dict slices_max_value: dict where key is index of slice, value is the list of max values in slice without the value of EES
	dict slices_min_time: dict where key is index of slice, value is the list of min times in slice without the time of EES
	dict slices_min_value: dict where key is index of slice, value is the list of min values in slice without the value of EES

	"""
	print("slices_max_time", slices_max_time)
	print("slices_max_value", slices_max_value)
	print("slices_min_time", slices_min_time)
	print("slices_min_value", slices_min_value)
	for index in range (1, len(slices_max_time)):
		del slices_max_time[index][0]
		del slices_max_value [index][0]
		del slices_min_time [index][0]
		del slices_min_value [index][0]
	return slices_max_time, slices_max_value, slices_min_time, slices_min_value


data_with_deleted_ees = remove_ees_from_min_max(data[0], data[1], data[2], data[3])
print("slices_max_time", data_with_deleted_ees[0])
print("slices_max_value", data_with_deleted_ees[1])
print("slices_min_time", data_with_deleted_ees[2])
print("slices_min_value", data_with_deleted_ees[3])