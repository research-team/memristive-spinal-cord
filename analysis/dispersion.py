import scipy.io as sio
import numpy as np
from matplotlib import pylab as plt
from analysis.real_data_slices import read_data, slice_myogram
from sklearn import preprocessing
import h5py as hdf5
import pandas as pd
import matplotlib.patches as mpatches
real_data_step = 0.25
# read real data
path = "SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat"
data = read_data('../bio-data/{}'.format(path))
processed_data = slice_myogram(data)
real_data = processed_data[0]
slices_begin_time = processed_data[1]
mat_data = sio.loadmat('../bio-data/{}'.format(path))
tickrate = int(mat_data['tickrate'][0][0])


def normalization(list_of_data_to_normalize, max_value, min_value):
    """

    :param list_of_data_to_normalize: list
        data that is needd to be normalized
    :param max_value: max value of normalized data
    :param min_value: min value of normalized data
    :return: list
        normalized data
    """
    fact_max = max(list_of_data_to_normalize)
    fact_min = min(list_of_data_to_normalize)
    x_max = fact_max / max_value
    x_min = fact_min / min_value
    scale = (x_max + x_min) / 2
    normal_data = []
    for i in range(len(list_of_data_to_normalize)):
        normal_data.append(list_of_data_to_normalize[i] / scale)
    return normal_data


# normalization of real data
normal_real = normalization(real_data, 0.6, -1)
# read neuron data

neuron_means_s125_dict = {}
path = '/home/anna/snap/telegram-desktop/common/Downloads/Telegram Desktop/res3110.hdf5'
with hdf5.File(path, 'r') as f:
    for test_name, test_values in f.items():
        neuron_means_s125_dict[test_name] = [volt * 10**10 for volt in test_values[:27000]]
neuron_means_s125 = []
for key, value in neuron_means_s125_dict.items():
    temp = [value]
    temp = sum(temp, [])
    neuron_means_s125.append(temp)
neuron_means_s125_one_list = sum(neuron_means_s125, [])
# normalization of neuron data
normal_neuron_means_s125_one_list = normalization(neuron_means_s125_one_list, 0.0577, -1)
# converting list of neuron data into the list of lists
temo = []
normal_neuron = []
for i in range(25):
    for j in range(27000):
        temo.append(normal_neuron_means_s125_one_list[j])
    normal_neuron.append(temo.copy())
    temo.clear()
neuron_dots_arrays = []
tmp = []
offset = 0
for iter_begin in range(len(normal_real)):
    for test_number in range(len(neuron_means_s125)):
        for chunk_len in range(offset, offset + 10):
            tmp.append(normal_neuron[test_number][chunk_len])
    neuron_dots_arrays.append(tmp.copy())
    tmp.clear()
    offset += 10
maxes = []
mins = []
dif = []
diffs = []
i = 0
for inner_list in neuron_dots_arrays:
    real_dot = normal_real[i]
    for neuron_dot in inner_list:
        dif.append(abs(real_dot - neuron_dot))
    diffs.append(dif.copy())
    dif.clear()
    i += 1
for d in diffs:
    maxes.append(max(d))
    mins.append(min(d))
times = []
for i in range(90):
    times.append(i)
maxes_by_slice = []
maxes_by_slice_temp = []
mins_by_slice = []
mins_by_slice_temp = []
offset = 0
for i in range(30):
    for j in range(offset, offset + 90):
        maxes_by_slice_temp.append(maxes[j])
        mins_by_slice_temp.append(mins[j])
    maxes_by_slice.append(maxes_by_slice_temp.copy())
    mins_by_slice.append(mins_by_slice_temp.copy())
    maxes_by_slice_temp.clear()
    mins_by_slice_temp.clear()
    offset += 90
plt.figure(figsize=(9, 3))

yticks = []
for index, sl in enumerate(range(len(maxes_by_slice))):
    offset = index * 1
    yticks.append(maxes_by_slice[sl][0]+ offset)
    plt.plot([data + offset for data in maxes_by_slice[sl]], color='r', linewidth=0.8)
    plt.plot([data + offset for data in mins_by_slice[sl]], color='b', linewidth=0.8)
    # plt.fill_between(times, [data + offset for data in mins_by_slice[sl]], [data + offset for data in mins_by_slice[sl]])
    plt.axhline(y=offset, color='gray', linestyle='--')
plt.axhline(y=offset + 1, color='gray', linestyle='--')
red_patch = mpatches.Patch(color='red', label='maxes')
blue_patch = mpatches.Patch(color='blue', label='mins')
plt.legend(handles=[red_patch, blue_patch], loc='upper right')
plt.yticks(yticks, range(len(maxes_by_slice)))
plt.xticks(range(0, 91, 18), range(0, 26, 5))
# plt.fill_between(times, mins, maxes, alpha=0.35, color='red')
plt.xlim(0, 90)
plt.subplots_adjust(left=0.03, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
plt.show()
