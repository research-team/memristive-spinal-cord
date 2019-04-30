from analysis.real_data_slices import read_data, trim_myogram
from matplotlib import pylab as plt
import numpy as np

path = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/10_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
path2 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/5_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
path3 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/4_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
path4 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/3_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
path5 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/1_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
# path6 = '../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz/7Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.mat'
# path7 = '../bio-data/10cms/Week6/7_WEEK6_Rat 40_8-8-2018.mat'
# path8 = '../bio-data/10cms/Week6/8_WEEK6_Rat 40_8-8-2018.mat'
# path9 = '../bio-data/10cms/Week6/9_WEEK6_Rat 40_8-8-2018.mat'
# path10 = '../bio-data/10cms/Week6/10_WEEK6_Rat 40_8-8-2018.mat'

folder = "/".join(path.split("/")[:-1])

raw_mat_data  = read_data(path)
raw_mat_data2  = read_data(path2)
raw_mat_data3  = read_data(path3)
raw_mat_data4  = read_data(path4)
raw_mat_data5  = read_data(path5)
# raw_mat_data6 = read_data(path6)
# raw_mat_data7 = read_data(path7)
# raw_mat_data8 = read_data(path8)
# raw_mat_data9 = read_data(path9)
# raw_mat_data10 = read_data(path10)

mat_data = trim_myogram(raw_mat_data, folder)
mat_data2 = trim_myogram(raw_mat_data2, folder)
mat_data3 = trim_myogram(raw_mat_data3, folder)
mat_data4 = trim_myogram(raw_mat_data4, folder)
mat_data5 = trim_myogram(raw_mat_data5, folder)
# mat_data6 = trim_myogram(raw_mat_data6, folder)
# mat_data7 = trim_myogram(raw_mat_data7, folder)
# mat_data8 = trim_myogram(raw_mat_data8, folder)
# mat_data9 = trim_myogram(raw_mat_data9, folder)
# mat_data10 = trim_myogram(raw_mat_data10, folder)

data = []


def bio_data_runs():
	# data.append([d for d in mat_data[0][0:1800]])
	data.append([d for d in mat_data[0][0:1200]])
	data.append([d for d in mat_data2[0][:1200]])
	data.append([d for d in mat_data3[0][1000:2200]])
	data.append([d for d in mat_data4[0][900:2100]])
	data.append([d for d in mat_data5[0][600:1800]])
	# data.append([d for d in mat_data6[0][:1200]])
	# data.append([d for d in mat_data7[0]])
	# data.append([d for d in mat_data8[0]])
	# data.append([d for d in mat_data9[0]])
	# data.append([d for d in mat_data10[0]])
	# for index, d in enumerate(data):
		# print("data = ", d)
		# plt.plot(d, label=index)
	# plt.legend()
	# plt.show()
	# print("len(data) = ", len(data))
	return data


# data = bio_data_runs()
# plt.plot(data)
# plt.show()


def bio_slices(data=data):
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
	# print("all_bio_slices = ", all_bio_slices)
	all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]
	return all_bio_slices

# bio_data_runs()

# data.append(mat_data6[0][:1200])
"""print(len(mat_data[0]))

indexes = mat_data[1]
indexes2 = mat_data2[1]
indexes3 = mat_data3[1]
indexes4 = mat_data4[1]
# indexes5 = mat_data5[1]
# indexes6 = mat_data6[1]
# indexes5.append(100)
# indexes4.append(100)
# indexes3.append(200)
# indexes4.append(300)
# indexes5.append(300)
# indexes3.append(400)
# indexes5.append(400)
# indexes.append(500)
# indexes3.append(500)
# indexes4.append(500)
# indexes5.append(500)
# indexes.append(600)
# indexes3.append(600)
# indexes4.append(600)
# indexes5.append(600)
# indexes4.append(700)
# indexes5.append(700)
# indexes3.append(900)
# indexes4.append(900)
# indexes5.append(900)
# indexes.append(1000)
# indexes2.append(1000)
# indexes2.append(1100)
# indexes4.append(1100)
# indexes4.append(1200)
# indexes4.append(1300)
# indexes5.append(1100)
# indexes4.append(1300)
# indexes5.append(1300)
# indexes3.append(1400)
# indexes4.append(1400)
# indexes5.append(1400)
# indexes4.append(1500)
# indexes5.append(1500)
# indexes5.append(1600)
# indexes5.append(1500)
# indexes.append(1600)
# indexes4.append(1600)
# indexes4.append(1700)
# indexes4.append(1800)
# indexes3.append(1800)
# indexes4.append(1800)
# indexes4.append(1900)
# indexes4.append(2000)
# indexes.append(2100)
# indexes4.append(2100)
# indexes5.append(2100)
# indexes.append(2200)
# indexes3.append(2200)
# indexes5.append(2200)
# indexes.append(2300)
# indexes2.append(2300)
# indexes3.append(2300)
# indexes.append(2400)
# indexes5.append(2400)
# indexes.append(2500)
# indexes5.append(2500)
# indexes.append(2600)
# indexes5.append(2600)
# indexes5.append(2700)
# indexes5.append(2900)
# indexes5.append(3100)
# indexes5.append(3300)
# indexes6.append(500)
indexes = sorted(indexes)
indexes2 = sorted(indexes2)
indexes3 = sorted(indexes3)
indexes4 = sorted(indexes4)
# indexes5 = sorted(indexes5)
# indexes6 = sorted(indexes6)
print(indexes)
print(indexes2)
print(indexes3)
print(indexes4)
# print(indexes5)
# print(indexes6)
mean_data = list(map(lambda elements: np.mean(elements), zip(*data)))  #
print("mean_data = ", len(mean_data), mean_data)
# plt.plot(mean_data)
# plt.show()
slices = []
slices_mean = []
offset = 0
slice_number = len(indexes4) - 2

for j in range(slice_number):
	# print("j = ", j)
	slices_tmp = []
	for i in range(offset, offset + 100):
		# print("i = ", i)
		slices_tmp.append(mat_data4[0][i])
	offset += 100
	slices.append(slices_tmp)
print("slices = ", slices)

slice_number = len(indexes3) - 1
offset = 0

for j in range(6):
	slices_tmp = []
	for i in range(offset, offset + 100):
		slices_tmp.append(mean_data[i])
	offset += 100
	slices_mean.append(slices_tmp)
print("slices_mean = ", slices_mean)

# plt.plot(mat_data2[0])
# for i in indexes:
# 	plt.axvline(x=i, linestyle='--', color='gray')
# plt.show()


def plot(slices_data, interval, sep_pattern=False):
	x = [t * 0.25 for t in range(100)]
	y = []

	for index, slice_data in enumerate(slices_data):
		offset = index * 2
		color = 'r' if index in interval else 'grey'

		if sep_pattern:
			if index in interval:
				y.append(offset + slice_data[0])
				plt.plot(x, [s + offset for s in slice_data], color=color)
		else:
			y.append(offset + slice_data[0])
			plt.plot(x, [s + offset for s in slice_data], color=color)

	plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)])
	plt.yticks(y, range(slice_number))
	plt.xlim(0, 25)
	plt.show()


plot(slices_mean, interval=range(0, 6), sep_pattern=False)

plot(slices, interval=range(0, 6), sep_pattern=True)"""
