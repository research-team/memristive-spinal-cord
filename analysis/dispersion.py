import scipy.io as sio
import numpy as np
from matplotlib import pylab as plt
from analysis.real_data_slices import read_data, slice_myogram
from sklearn import preprocessing
import h5py as hdf5
import pandas as pd
real_data_step = 0.25
# read real data
path = "SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat"
data = read_data('../bio-data/{}'.format(path))
processed_data = slice_myogram(data)
real_data = processed_data[0]
slices_begin_time = processed_data[1]
mat_data = sio.loadmat('../bio-data/{}'.format(path))
tickrate = int(mat_data['tickrate'][0][0])
# normalization of real data
oldmin = min(real_data)
oldmax = max(real_data)
oldrange = oldmax - oldmin
newmin = 0.
newmax = 1.
newrange = newmax - newmin
if oldrange == 0:
    if oldmin < newmin:
        newval = newmin
    elif oldmin > newmax:
        newval = newmax
    else:
        newval = oldmin
    normal = [newval for v in real_data]
else:
    scale = newrange / oldrange
    normal_real = [(v - oldmin) * scale + newmin for v in real_data]
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
normal_neuron_means_s125_one_list = []
min_value = min(neuron_means_s125_one_list)
max_value = max(neuron_means_s125_one_list)
for i in range(len(neuron_means_s125_one_list)):
    normal_neuron_means_s125_one_list.append((neuron_means_s125_one_list[i] - min_value) / (max_value - min_value))
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
print("normal_real[0] = ", normal_real[0])
print("diffs[0] = ", diffs[0])
print("maxes = ", maxes)
print("mins = ", mins)
times = []
for i in range(len(normal_real)):
    times.append(i)

plt.figure(figsize=(9, 3))
plt.plot(maxes, label="maxes")
plt.plot(mins, label="mins")
# plt.fill_between(times, mins, maxes, alpha=0.35, color='red')
ticks = []
labels = []
for t in range(0, 726, 25):
    labels.append(t)
for i in range(0, 2700, 90):
    ticks.append(i)

plt.xticks(ticks, labels)
for tick in ticks:
    plt.axvline(x=tick, linestyle="--", color="gray")

plt.legend()
plt.show()
