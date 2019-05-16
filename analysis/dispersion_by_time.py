import scipy.io as sio
from analysis.real_data_slices import read_data, trim_myogram
import h5py as hdf5
import numpy as np
from matplotlib import pylab as plt
import matplotlib.pyplot as pyp
recording_step = 0.025
# read real data
path = "SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat"
data = read_data('../bio-data/{}'.format(path))
processed_data = trim_myogram(data)
real_data = processed_data[0]
slices_begin_time = processed_data[1]
slices_begin_time = [int(slice_index) * 4 for slice_index in slices_begin_time]
# print("slices_begin_time = ", slices_begin_time)
# print("len(real_data) = ", len(real_data))
mat_data = sio.loadmat('../bio-data/{}'.format(path))
tickrate = int(mat_data['tickrate'][0][0])
# find peaks of real data
min_elems = []
max_elems = []
temp_min_indexes = []
temp_max_indexes = []
min_indexes = []
max_indexes = []
offset = 0
for index in slices_begin_time[1:]:
    # print("index = ", index)
    for i in range(1 + offset, index - 1):
        if real_data[i- 1] > real_data[i] <= real_data[i + 1]:
            min_elems.append(i * recording_step)
            temp_min_indexes.append(i)
        if real_data[i- 1] < real_data[i] >= real_data[i + 1]:
            max_elems.append(i * recording_step)
            temp_max_indexes.append(i)
    min_indexes.append(temp_min_indexes.copy())
    temp_min_indexes.clear()
    max_indexes.append(temp_max_indexes.copy())
    temp_max_indexes.clear()
    offset = index
# print("max_indexes = ", max_indexes)
# print("min_indexes = ", min_indexes)
max_times = []
min_times = []
max_times_temp = []
min_times_temp = []

for ind in max_indexes:
    max_times.append([i * 0.25 for i in ind])

# print("max_times = ", max_times)


for ind in min_indexes:
    min_times.append([i * 0.25 for i in ind])
# print("min_times = ", min_times)
bio_cutted_max_times = []
bio_cutted_min_times = []
for times in max_times:
    bio_cutted_max_times.append(times[:4])
# print("bio_cutted_max_times = ", bio_cutted_max_times)
for times in min_times:
    bio_cutted_min_times.append(times[:4])
# print("bio_cutted_min_times = ", bio_cutted_min_times)

# plot real data
# plt.plot(real_data)
# for slice_begin in slices_begin_time:
    # plt.axvline(x=slice_begin, linestyle="--", color="gray")
# plt.show()
# read neuron data
neuron_means_s125_dict = {}
path = '/home/anna/snap/telegram-desktop/common/Downloads/Telegram Desktop/res3110.hdf5'
with hdf5.File(path, 'r') as f:
    for test_name, test_values in f.items():
        neuron_means_s125_dict[test_name] = [volt * 10**10 for volt in test_values[:27000]]
# neuron data to list
neuron_means_s125 = []
for key, value in neuron_means_s125_dict.items():
    temp = [value]
    temp = sum(temp, [])
    neuron_means_s125.append(temp)
# mean for 25 runs of neuron data
neuron_means_s125 = list(map(lambda x: np.mean(x), zip(*neuron_means_s125)))
# print("len(neuron_means_s125) = ", len(neuron_means_s125))

def find_mins(array):
    """
    function that finds values and indexes of min peaks
    :param array: array where it is necessary to find min peaks
    :return:
    min_elems: list
        min peaks' values
    indexes: list
        min peaks' indexes
    """
    min_elems = []
    indexes = []
    for index_elem in range(1, len(array) - 1):
        if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]) and array[index_elem] < -15:
            min_elems.append(index_elem * recording_step)
            indexes.append(index_elem)
    return min_elems, indexes

# find indexes of spikes in neuron data
indexes = find_mins(neuron_means_s125)[1]
indexes.append(26999)
# find peaks of neuron data
min_elems = []
max_elems = []
temp_min_indexes = []
temp_max_indexes = []
min_indexes = []
max_indexes = []
offset = 0
for index in indexes[1:]:
    # print("index = ", index)
    # print("offset = ", offset)
    for i in range(1 + offset, index - 1):
        # print("i = ", i)
        if neuron_means_s125[i- 1] > neuron_means_s125[i] <= neuron_means_s125[i + 1]:
            temp_min_indexes.append(i)
        if neuron_means_s125[i - 1] < neuron_means_s125[i] >= neuron_means_s125[i + 1]:
            temp_max_indexes.append(i)
    min_indexes.append(temp_min_indexes.copy())
    temp_min_indexes.clear()
    max_indexes.append(temp_max_indexes.copy())
    temp_max_indexes.clear()
    offset = index
max_times = []
min_times = []
max_times_temp = []
min_times_temp = []

for ind in max_indexes:
    max_times.append([round(i * 0.025, 3) for i in ind])


for ind in min_indexes:
    min_times.append([round(i * 0.025, 3) for i in ind])
neuron_cutted_max_times = []
neuron_cutted_min_times = []
for times in max_times:
    neuron_cutted_max_times.append(times[:4])
for times in min_times:
    neuron_cutted_min_times.append(times[:4])
# plot neuron data
# plt.plot(neuron_means_s125)
# for slice_begin in indexes:
#     plt.axvline(x=slice_begin, linestyle="--", color="gray")

# plt.show()
dif_max = []
dif_min = []
dif_slices_max = []
dif_slices_min = []
for i in range(len(bio_cutted_max_times)):
    for j in range(len(bio_cutted_max_times[i])):
        dif_max.append(round(bio_cutted_max_times[i][j] - neuron_cutted_max_times[i][j], 3))
    dif_slices_max.append(dif_max.copy())
    dif_max.clear()
for i in range(len(bio_cutted_min_times)):
    for j in range(len(bio_cutted_min_times[i])):
        dif_min.append(round(bio_cutted_min_times[i][j] - neuron_cutted_min_times[i][j], 3))
    dif_slices_min.append(dif_min.copy())
    dif_min.clear()
print("dif_slices = ", dif_slices_max)
print("dif_slices = ", dif_slices_min)
print("len(dif_slices_max) = ", len(dif_slices_max))
print(len(dif_slices_min))
print(len(dif_slices_max))
for i in range(len(dif_slices_max)):
    while len(dif_slices_max[i]) < 4:
        dif_slices_max[i].append(0)
for i in range(len(dif_slices_min)):
    while len(dif_slices_min[i]) < 4:
        dif_slices_min[i].append(0)
ax = pyp.axes()
ax.yaxis.grid(True, zorder = 1)
xs = range(len(dif_slices_min))
one = []
two = []
three = []
four = []
for i in range(len(dif_slices_max)):
    one.append(dif_slices_max[i][0])
    two.append(dif_slices_max[i][1])
    three.append(dif_slices_max[i][2])
    four.append(dif_slices_max[i][3])
pyp.bar([x for x in xs], one, width=0.2, color='red', alpha=0.7,
        label='difference between times of the first max peak', zorder=2)
pyp.bar([x + 0.2 for x in xs], two, width=0.2, color='blue', alpha=0.7,
        label='difference between times of the second max peak', zorder=2)
pyp.bar([x + 0.4 for x in xs], three, width=0.2, color='yellow', alpha=0.7,
        label='difference between times of the third max peak', zorder=2)
pyp.bar([x + 0.6 for x in xs], four, width=0.2, color='green', alpha=0.7,
        label='difference between times of the fourth max peak', zorder=2)
pyp.xticks(xs, range(27))
for i in range(27):
    pyp.axvline(x=i, linestyle='--', color='gray')
pyp.legend(loc='upper right')
pyp.show()