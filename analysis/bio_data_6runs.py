from analysis.functions import read_bio_data
from matplotlib import pylab as plt
import numpy as np
raw_data_run1 = read_bio_data('../bio-data/2Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.txt', 4.962)
# changed the function!
raw_data_run2 = read_bio_data('../bio-data/3Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.txt', 4.96)
raw_data_run3 = read_bio_data('../bio-data/4Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.txt', 4.96)
raw_data_run4 = read_bio_data('../bio-data/5Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.txt', 4.96)
raw_data_run5 = read_bio_data('../bio-data/6Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.txt', 4.96)
raw_data_run6 = read_bio_data('../bio-data/7Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.txt', 4.96)
data_run1 = raw_data_run1[0]
indexes_run1 = raw_data_run1[1]
# print(data_run1)
# print(indexes_run1)
data_run2 = raw_data_run2[0]
indexes_run2 = raw_data_run2[1]
# print(data_run2)
# print(indexes_run2)
data_run3 = raw_data_run3[0]
indexes_run3 = raw_data_run3[1]
# print(data_run3)
# print(indexes_run3)
data_run4 = raw_data_run4[0]
indexes_run4 = raw_data_run4[1]
# print(data_run4)
# print(indexes_run4)
data_run5 = raw_data_run5[0]
indexes_run5 = raw_data_run5[1]
# print(data_run5)
# print(indexes_run5)
data_run6 = raw_data_run6[0]
indexes_run6 = raw_data_run6[1]
# print(data_run6)
# print(indexes_run6)

# plt.plot(data_run6)
# for sl in indexes_run6:
# 	plt.axvline(x=sl, linestyle='--', color='gray')
# plt.xlim(0, 2700)
# plt.show()
del indexes_run4[-3]
del indexes_run5[3]
del indexes_run6[7]
# print(indexes_run4)
indexes_run1 = indexes_run1[:-4]
indexes_run3 = indexes_run3[:-1]
indexes_run5 = indexes_run5[:-8]
indexes_run6 = indexes_run6[:-3]

data_run1 = data_run1[indexes_run1[0]:indexes_run1[-1]]
data_run3 = data_run3[indexes_run1[0]:indexes_run1[-1]]
data_run4 = data_run4[indexes_run1[0]:indexes_run1[-1]]
data_run5 = data_run5[indexes_run1[0]:indexes_run1[-1]]
data_run6 = data_run6[indexes_run1[0]:indexes_run1[-1]]
sliced_data_run1 = []
sliced_data_run2 = []
sliced_data_run3 = []
sliced_data_run4 = []
sliced_data_run5 = []
sliced_data_run6 = []
for i in range(1, len(indexes_run1)):
	sliced_data_run1.append(data_run1[indexes_run1[i - 1]:indexes_run1[i]])
for i in range(1, len(indexes_run2)):
	sliced_data_run2.append(data_run2[indexes_run2[i - 1]:indexes_run2[i]])
for i in range(1, len(indexes_run3)):
	sliced_data_run3.append(data_run3[indexes_run3[i - 1]:indexes_run3[i]])
for i in range(1, len(indexes_run4)):
	sliced_data_run4.append(data_run4[indexes_run4[i - 1]:indexes_run4[i]])
for i in range(1, len(indexes_run5)):
	sliced_data_run5.append(data_run5[indexes_run5[i - 1]:indexes_run5[i]])
for i in range(1, len(indexes_run6)):
	sliced_data_run6.append(data_run6[indexes_run6[i - 1]:indexes_run6[i]])

data = []
data.append(sliced_data_run1)
data.append(sliced_data_run2)
data.append(sliced_data_run3)
data.append(sliced_data_run4)
data.append(sliced_data_run5)
data.append(sliced_data_run6)
data_for_shadows = []
for sl in range(len(data[0])):
	data_for_shadows_tmp = []
	for run in range(len(data)):
		data_for_shadows_tmp.append(data[run][sl])
	data_for_shadows.append(data_for_shadows_tmp)
yticks = []
for index, sl in enumerate(range(len(data_for_shadows))):
	offset = index * 6
	mean_data = list(map(lambda elements: np.mean(elements), zip(*data_for_shadows[sl])))
	times = [time * 0.25 for time in range(len(mean_data))]
	means = [voltage + offset for voltage in mean_data]
	yticks.append(means[0])
	minimal_per_step = [min(a) for a in zip(*data_for_shadows[sl])]
	maximal_per_step = [max(a) for a in zip(*data_for_shadows[sl])]
	print("len(times) = ", len(times))
	print("len(means) = ", len(means))
	print("len(minimal_per_step) = ", len(minimal_per_step))
	print("len(maximal_per_step) = ", len(maximal_per_step))
	plt.plot(times, means, color='k')
	print("f plot")
	plt.fill_between(times, [mini + offset for mini in minimal_per_step], [maxi + offset for maxi in maximal_per_step],
	                 alpha=0.35)
	print("s plot")
	plt.yticks(yticks, range(1, len(data_for_shadows) + 1))
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.xlim(0, 25)
plt.show()