import scipy.io as sio
import numpy as np
# from analysis.real_data_slices import read_data, slice_myogram
real_data_step = 0.25
def read_data(file_path):
	global tickrate
	global title
	mat_data = sio.loadmat(file_path)
	tickrate = int(mat_data['tickrate'][0][0])
	title = mat_data['titles'][0]
	return mat_data


def slice_myogram(raw_data, slicing_index ='Stim'):
	"""
	The function to slice the data from the matlab file of myogram.
	:param dict raw_data:  the myogram data loaded from matlab file.
	:param str slicing_index: the index to be used as the base for slicing, default 'Stim'.
	:return: list volt_data: the voltages array
	:return: list slices_begin_time: the staring time of each slice array.
	"""
	# Collect data
	volt_data = []
	stim_data = []
	slices_begin_time = []

	# data processing
	for index, data_title in enumerate(raw_data['titles']):
		data_start = int(raw_data['datastart'][index]) - 1
		data_end = int(raw_data['dataend'][index])
		float_data = [round(float(x), 3) for x in raw_data['data'][0][data_start:data_end]]
		if slicing_index not in data_title:
			volt_data = float_data
		else:
			stim_data = float_data

	# find peaks in stimulations data
	for index in range(1, len(stim_data) - 1):
		if stim_data[index - 1] < stim_data[index] > stim_data[index + 1] and stim_data[index] > 4:
			slices_begin_time.append(index * real_data_step)  # division by 4 gives us the normal 1 ms step size

	# remove unnecessary data, use only from first stim, and last stim
	volt_data = volt_data[int(slices_begin_time[0] / real_data_step):int(slices_begin_time[-1] / real_data_step)]

	# move times to the begin (start from 0 ms)
	slices_begin_time = [t - slices_begin_time[0] for t in slices_begin_time]

	return volt_data, slices_begin_time




# read real data
path = "SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat"
data = read_data('../bio-data/{}'.format(path))
processed_data = slice_myogram(data)
raw_real_data = processed_data[0]
raw_slices_begin_time = processed_data[1]
print("raw_slices_begin_time = ", raw_slices_begin_time)
print("len(raw_slices_begin_time) = ", len(raw_slices_begin_time))
print("len(raw_real_data) = ", len(raw_real_data))  # 2700
real_data = []
real_data_time = []
for data in range(1200):
	real_data.append(raw_real_data[data])
for time in range(13):
	real_data_time.append(raw_slices_begin_time[time])
print("real_data_time = ", real_data_time)