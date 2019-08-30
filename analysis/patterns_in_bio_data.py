from analysis.real_data_slices import read_data, trim_myogram
from matplotlib import pylab as plt
from NEST.misc.converter_hdf5 import write_to_hdf5
import h5py as hdf5
import scipy.io as sio
import numpy as np
import itertools


def bio_data_runs():
	path = '/home/anna/Desktop/data/bio/emg/Trial/100Hz/Rat 25 2-14-2018 100 Hz trial 06_flexor.mat'
	# path3 = '/home/anna/Desktop/data/bio/Different Frequencies_20_40_100_250 Hz/20Hz/SCI-RTA-20Hz/2-SCI Rat-1_11-22-2016_RTA_20Hz_one_step.mat'
	# path2 = '/home/anna/Desktop/data/bio/Different Frequencies_20_40_100_250 Hz/20Hz/SCI-RTA-20Hz/3-SCI Rat-1_11-22-2016_RTA_20Hz_one_step.mat'
	# path4 = '/home/anna/Desktop/data/bio/Different Frequencies_20_40_100_250 Hz/20Hz/SCI-RTA-20Hz/4-SCI Rat-1_11-22-2016_RTA_20Hz_one_step.mat'
	# path5 = '/home/anna/Desktop/data/bio/Different Frequencies_20_40_100_250 Hz/20Hz/SCI-RTA-20Hz/SCI Rat-1_11-22-2016_RTA_20Hz_one_step.mat'
	# path5 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/1_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
	# path7 = '../bio-data/10cms/Week6/7_WEEK6_Rat 40_8-8-2018.mat'
# path8 = '../bio-data/10cms/Week6/8_WEEK6_Rat 40_8-8-2018.mat'
# path9 = '../bio-data/10cms/Week6/9_WEEK6_Rat 40_8-8-2018.mat'
# path10 = '../bio-data/10cms/Week6/10_WEEK6_Rat 40_8-8-2018.mat'

	folder = "/".join(path.split("/")[:-1])
	# align_coef = 0.001

	raw_mat_data = sio.loadmat(path)
	print(raw_mat_data.keys())
	mat_data = raw_mat_data['LTA_muscle']
	print("len(mat_data) = ", len(mat_data), len(mat_data[0]))
	mat_data = np.array(mat_data)
	mat_data =mat_data.T

	mat_data = list(itertools.chain(*mat_data))

	print("mat_data = ", len(mat_data))
	print(raw_mat_data['__header__'])
	print(raw_mat_data['__version__'])
	print(raw_mat_data['__globals__'])

	freq = 100
	slice_duration = int(1 / freq * 1000)
	slice_duration_in_dots = slice_duration * 4
	hundred_slices = slice_duration_in_dots * 50

	print("slice_duration = ", slice_duration)
	print("slice_duration_in_dots = ", slice_duration_in_dots)
	print("hundred_slices = ", hundred_slices)
	offset = 0 + hundred_slices + hundred_slices + hundred_slices + hundred_slices + hundred_slices
	mat_slices = []
	for slices in range(int(len(mat_data[offset:offset + hundred_slices]) / slice_duration_in_dots)):
		mat_slices_tmp = []
		for dots in range(offset, offset + slice_duration_in_dots):
			mat_slices_tmp.append(mat_data[dots])
		mat_slices.append(mat_slices_tmp)
		offset += slice_duration_in_dots

	# print(len(mat_slices), len(mat_slices[0]))

	# times = [time * 0.25 for time in range(len(mat_data))]
	# plt.plot(times, [m + offset for m in mat_data])
	# plt.xlim(0, 25)
	# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
	# plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
	# plt.show()
	# raw_mat_data   = read_data(path)
	# raw_mat_data2  = read_data(path2)
	# raw_mat_data3  = read_data(path3)
	# raw_mat_data4  = read_data(path4)
	# raw_mat_data5  = read_data(path5)
	# raw_mat_data6 = read_data(path6)
# raw_mat_data7 = read_data(path7)
# raw_mat_data8 = read_data(path8)
# raw_mat_data9 = read_data(path9)
# raw_mat_data10 = read_data(path10)

# print(f"mat_data = {mat_data[0]},\n {mat_data[1]}")
# mat_data7 = trim_myogram(raw_mat_data7, folder)
# mat_data8 = trim_myogram(raw_mat_data8, folder)
# mat_data9 = trim_myogram(raw_mat_data9, folder)
# mat_data10 = trim_myogram(raw_mat_data10, folder)

	data = []
	# mat_data = trim_myogram(raw_mat_data, folder)

	# print("len(mat_data) = ", len(mat_data), len(mat_data[0]))
	# mat_data2 = trim_myogram(raw_mat_data2, folder)
	# mat_data3 = trim_myogram(raw_mat_data3, folder)
	# mat_data4 = trim_myogram(raw_mat_data4, folder)
	# mat_data5 = trim_myogram(raw_mat_data5, folder)
	# mat_data6 = trim_myogram(raw_mat_data6, folder)

	# print("len(mat_data2) = ", len(mat_data2[0]))
	# print("len(mat_data3) = ", len(mat_data3[0]))
	# print("len(mat_data4)= ", len(mat_data4[0]))
	# print("len(mat_data5)= ", len(mat_data5[0]))

	# plt.plot(mat_data[0])
	# plt.show()
	offset = 0  # hundred_slices + hundred_slices # + hundred_slices + hundred_slices + hundred_slices
	slices = []

	# for k in range(int(len(mat_data[0][offset:offset + hundred_slices]) / slice_duration_in_dots)):
	# 	slices_tmp = []
	# 	for i in range(offset, offset + slice_duration_in_dots):
	# 		slices_tmp.append(mat_data[0][i])
	# 	offset += slice_duration_in_dots
	# 	slices.append(slices_tmp)
	# slices.append(mat_data[0][80:])
	for s in slices:
		print(len(s))
	print("---")
	print(len(mat_data))
	print("---")
	yticks = []
	for index, sl in enumerate(mat_slices):
		offset = index
		times = [time * 0.25 for time in range(len(sl))]
		plt.plot(times, [s + offset for s in sl])
		yticks.append(sl[0] + offset)
	plt.xticks(range(slice_duration + 1), [i if i % 1 == 0 else "" for i in range(slice_duration + 1)], fontsize=14)
	plt.yticks(yticks, range(1, int(len(mat_data)) + 1), fontsize=14)  # range(1, int(len(mat_data[0]) / 100) + 1)
	plt.xlim(0, slice_duration)
	plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
	plt.show()

	data.append([d for d in mat_data[0][2100:2800]])    # bipedal control rats 21cm/s ex [0:600]
	# no quipazine bipedal [0:1200] quadrupedal [0:1200]
	data.append([d for d in mat_data2[0][1800:2500]])  # no quipazine bipedal [0:1200]
	data.append([d for d in mat_data3[0][1700:2400]])   # no quipazine bipedal [1000:2200]
	data.append([d for d in mat_data4[0][1500:2200]])    # no quipazine bipedal [900:2100]
	data.append([d for d in mat_data5[0][1900:2600]])    # no quipazine bipedal [600:1800] quadrupedal [1400:2600]
	# data.append([-d for d in mat_data6[0][0:1200]])
	# data.append([d for d in mat_data7[0]])
	# data.append([d for d in mat_data8[0]])
	# data.append([d for d in mat_data9[0]])
	# data.append([d for d in mat_data10[0]])
	# for run in data:
	# 	for dot in range(len(run)):
	# 		run[dot] -= align_coef * dot
	raise Exception

	with hdf5.File('/home/anna/Desktop/data/bio/4pedal/bio_F_13.5cms_40Hz_i100_4pedal_no5ht_T_0.25step.hdf5','w') as file:
		for i, d in enumerate(data):
			print(d)
			file.create_dataset(name=str(i), data=d)
	raise Exception
	# write_to_hdf5(data, 'bio_F_21_40Hz_i100_4pedal_no5ht_T_0.25step')
	return data


bio_data_runs()


def bio_slices(data):
	all_bio_slices = []
	# forming list for the plot
	for k in range(len(data)):
		bio_slices = []
		offset = 0
		for i in range(int(len(data[k]) / 100)):
			bio_slices_tmp = []
			for j in range(offset, offset + 100):
				bio_slices_tmp.append(data[k][j])
			bio_slices.append(bio_slices_tmp)
			offset += 100
		all_bio_slices.append(bio_slices)   # list [4][16][100]
	all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]
	return all_bio_slices