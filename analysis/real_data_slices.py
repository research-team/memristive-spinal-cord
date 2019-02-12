import numpy as np
import pylab as plt
import scipy.io as sio
from analysis.functions import convert_bio_to_hdf5

datas_max = []
datas_min = []
datas_times = []
datas_max_time = []
datas_min_time = []

title = ""
tickrate = 0
real_data_step = 0.25


def read_data(file_path):
	global tickrate
	global title
	mat_data = sio.loadmat(file_path)
	tickrate = int(mat_data['tickrate'][0][0])
	title = mat_data['titles'][0]
	return mat_data


def trim_myogram(raw_data, path, slicing_index='Stim'):
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
	global title

	# data processing
	title_stim = 'Stim'
	title_rmg = 'RMG'
	for index, data_title in enumerate(raw_data['titles']):
		data_start = int(raw_data['datastart'][index]) - 1
		data_end = int(raw_data['dataend'][index])
		float_data = [round(float(x), 3) for x in raw_data['data'][0][data_start:data_end]]
		if title_rmg in data_title:
			volt_data = float_data
		if title_stim in data_title:
			stim_data = float_data

	# convert_bio_to_hdf5(volt_data, stim_data, path)

	import h5py as hdf5
	# with hdf5.File(path + ".hdf5") as file:
		# for k,v in file.items():
			# print(k, v[:])

	# find peaks in stimulations data
	ms_pause = 0
	bio_step = 0.25
	# print("stim_data = ", stim_data)
	for index in range(1, len(stim_data) - 1):
		if stim_data[index - 1] < stim_data[index] > stim_data[index + 1] and ms_pause <= 0 and\
				stim_data[index] > 0.5:
			slices_begin_time.append(index) # * real_data_step  # division by 4 gives us the normal 1 ms step size
			ms_pause = int(3 / bio_step)
		ms_pause -= 1
	# print("slices_begin_time = ", slices_begin_time)
	# remove unnecessary data, use only from first stim, and last stim
	volt_data = volt_data[slices_begin_time[0]:slices_begin_time[-1]]

	# move times to the begin (start from 0 ms)
	slices_begin_time = [t - slices_begin_time[0] for t in slices_begin_time]
	# print("len(volt_data) = ", len(volt_data))
	return volt_data, slices_begin_time


def plot_1d(volt_data, slices_begin_time):
	# Plot data
	x = [i / tickrate * 1000 for i in range(len(volt_data))]
	plt.plot(x, volt_data, label=title)
	plt.plot(slices_begin_time, [0 for _ in slices_begin_time], ".", color='r')
	plt.xlim(0, slices_begin_time[-1])

	for begin_slice_time in slices_begin_time:
		plt.axvline(x=begin_slice_time, linestyle="--", color="gray")

	plt.xticks(np.arange(0, slices_begin_time[-1] + 1, 25), np.arange(0, slices_begin_time[-1] + 1, 25))
	plt.legend()
	plt.show()


def plot_by_slice(volt_data, slices_begin_time):
	plt.plot(volt_data)
	plt.show()
	for index, slice_begin_time in enumerate(slices_begin_time[:-1]):
		start_time = int(slice_begin_time * 4)
		y = volt_data[start_time:start_time + 100]
		plt.plot([x / 4 for x in range(len(y))],
		         [volt + 2.5 * index  for volt in y],
		         color='gray')
	plt.xlim(0, 25)
	plt.xticks(np.arange(0, 26), np.arange(0, 26))
	plt.show()


def main():
	raw_data = read_data('../bio-data//SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
	volt_data, slices_begin_time = trim_myogram(raw_data)
	# plot_1d(volt_data, slices_begin_time)
	plot_by_slice(volt_data, slices_begin_time)


if __name__ == "__main__":
	main()
