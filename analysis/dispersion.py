import scipy.io as sio
import numpy as np
from matplotlib import pylab as plt
from analysis.real_data_slices import read_data, slice_myogram
from sklearn import preprocessing
import h5py as hdf5
real_data_step = 0.25
# read real data
path = "SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat"
data = read_data('../bio-data/{}'.format(path))
processed_data = slice_myogram(data)
real_data = processed_data[0]
slices_begin_time = processed_data[1]
mat_data = sio.loadmat('../bio-data/{}'.format(path))
tickrate = int(mat_data['tickrate'][0][0])
# real_data = []
# for i in range(slices_begin_time[0]:slices_begin_time[-1]):
# for d in range(1200):
# 	real_data.append(raw_real_data[d])
print("len(real_data) = ", len(real_data))  # 2700
# normalization
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
print("normal real = ", normal_real)
print("len(normal real) = ", len(normal_real), normal_real[:5])
# print("real_data = ", real_data)
# plot real data
# plt.plot(real_data)
# x = [i / tickrate * 1000 for i in range(len(real_data))]
# plt.plot(x, real_data, label=title)
# plt.plot(slices_begin_time, [0 for _ in slices_begin_time], ".", color='r')

# for kk in slices_begin_time:
# 	plt.axvline(x=kk, linestyle="--", color="gray")
# plt.xticks(np.arange(0, slices_begin_time[-1] + 1, 25), np.arange(0, slices_begin_time[-1] + 1, 25))

# plt.show()
# read neuron data

neuron_means_s125_dict = {}
path = '/home/anna/snap/telegram-desktop/common/Downloads/Telegram Desktop/res3110.hdf5'
with hdf5.File(path, 'r') as f:
    for test_name, test_values in f.items():
        neuron_means_s125_dict[test_name] = [volt * 10**10 for volt in test_values[:27000]]
#         print(len(neuron_means_s125[test_name]), "neuron_means_s125 = ", neuron_means_s125[test_name])
# raise Exception
# threads = 8
# test_numbers = 25
# speeds = [25, 50, 125]
# neuron_number = 21
# tmp = []
# V_to_uV = 1000000  # convert Volts to micro-Volts (V -> uV)
# neuron_tests_s125 = []
# for thread in range(threads):
#     for test_number in range(test_numbers):
#         for speed in speeds:
#             for neuron_id in range(neuron_number):
#                 if speed == 125:
#                     with open('/home/anna/PycharmProjects/LAB/neuron-data/res3110/vMN{}r{}s{}v{}'.format(neuron_id, thread, speed, test_number), 'r') as file:
#                         print("opened", 'res3110/volMN{}r{}s{}v{}.txt'.format(neuron_id, thread, speed, test_number))
#                         tmp.append([float(i) * V_to_uV for i in file.read().split("\n")[1:-1]])
#                     neuron_tests_s125.append([elem * 10 ** 4 for elem in list(map(lambda x: np.mean(x), zip(*tmp)))])
#                     del tmp[:]
# print("len(neuron_tests_s125) = ", len(neuron_tests_s125))
# print("len(neuron_tests_s125[0]) = ", len(neuron_tests_s125[0]))

# list(map(lambda x: np.mean(x), zip(*neuron_tests_s125)))
# print("len(raw_neuron_means_s125) = ", len(raw_neuron_means_s125))
# print("neuron_means_s125 = ", neuron_means_s125)
# cutted_neuron_tests_s125 = []
# for j in range(len(neuron_tests_s125)):

# neuron_means_s125 append(raw_neuron_means_s125[i])
neuron_means_s125 = []
for key, value in neuron_means_s125_dict.items():
    temp = [value]
    temp = sum(temp, [])
    neuron_means_s125.append(temp)
# for i in range(25):
#     for j in range(27000):
#         if neuron_means_s125[i][j] > 1:
#             print('yes ', 'i = ', i, 'j = ', j)
#         if neuron_means_s125[i][j] < 0:
#             print('no ', 'i = ', i, 'j = ', j)
#             break
print("neuron_means_s125[0][:10] = ", neuron_means_s125[0][:10])
# normalization
normal_neuron = preprocessing.normalize(neuron_means_s125)
print("len(normal_neuron) = ", normal_neuron[0][:10])
# normal_neuron = [volt * 10**4 for volt in normal_neuron]
# print("normal_neuron = ", normal_neuron[0][:10])
# raise Exception
# print("neuron_means_s125 = ", neuron_means_s125)
# raise Exception
"""oldmax_dict = {}
oldmin_dict = {}
oldrange = []
oldmax_temp = []
oldmin_temp = []
oldmax = []
oldmin = []
scale = []
test_numbers = len(neuron_means_s125)

for test_number in neuron_means_s125:
    oldmax.append(max(test_number))
    oldmin.append(min(test_number))
for i in range(len(oldmax)):
    oldrange.append(oldmax[i] - oldmin[i])
# print("oldmin = ", oldmin)
# print("oldmax = ", oldmax)
# # for test_name in neuron_means_s125:
# #     print(neuron_means_s125[test_name][:5])
# # print("oldmax = ", oldmax)
# # print("oldmin = ", oldmin)
# # print(len(oldrange), "oldrange = ", oldrange)
# raise Exception
newmin = 0.
newmax = 1.
newrange = newmax - newmin
normal_neuron = []
normal_neuron_temp= []
newval = []
normal = []
# # print("newmin = ", newmin)
# # print("newmax = ", newmax)
# # raise Exception
# for test_name, test_values in neuron_means_s125.items():
#     print(len(test_values), "v = ", [v for v in test_values])
for i in range(2):   # len(oldrange)
    # newrange.append(newmax[i] - newmin[i])
    if oldrange[i] == 0:
        if oldmin[i] < newmin[i]:
            newval.append(newmin[i])
        elif oldmin[i] > newmax[i]:
            newval.append(newmax[i])
        else:
            newval.append(oldmin[i])
        normal_neuron_temp.append([newval[i] for v in neuron_means_s125[i]])
    else:
        scale.append(newrange / oldrange[i])
        for j in range(27000):   # len(neuron_means_s125[i])
            normal_neuron_temp.append([(v - oldmin[i]) * scale[i] + newmin for v in neuron_means_s125[i]])
            print("j = ", j)
        # print("v = ", [v for v in neuron_means_s125[i]], '\n')
        print("len(normal_neuron_temp) = ", len(normal_neuron_temp))
        normal_neuron.append(normal_neuron_temp.copy())
        normal_neuron_temp.clear()"""
# print("len(normal_neuron) = ", len(normal_neuron))
# print("len(normal_neuron[0]) = ", len(normal_neuron[0]))
# raise Exception
# plt.plot(range(len(neuron_means_s125)), normal_neuron)
# print("normal = ", normal)
neuron_dots_arrays = []
tmp = []
offset = 0
for iter_begin in range(len(normal_real)):
    for test_number in range(len(neuron_means_s125)):
        for chunk_len in range(offset, offset + 10):
            tmp.append(normal_neuron[test_number][chunk_len])
        # print("iter_begin1 = ", iter_begin)
        # print("len(tmp) in cycle1 = ", len(tmp))
    neuron_dots_arrays.append(tmp.copy())
    tmp.clear()
    offset += 10
# print("len(neuron_dots_arrays) in cycle = ", len(neuron_dots_arrays))
# offset = 0
# for iter_begin in range(2400):
# 	for chunk_len in range(offset, offset + 11):
# 		tmp.append(normal_neuron[chunk_len])
# 	# print("iter_begin2 = ", iter_begin)
# 	# print("len(tmp) in cycle2 = ", len(tmp))
# 	copy = list(tmp)
# 	neuron_dots_arrays.append(copy)
# 	del tmp[:]
# 	offset += 11
print("len(neuron_dots_arrays) = ", len(neuron_dots_arrays))
# print("neuron_dots_arrays2 = '\n'", neuron_dots_arrays)
print("len(neuron_dots_arrays[0]) = ", len(neuron_dots_arrays[0]))
print("len(neuron_dots_arrays[-1]) = ", len(neuron_dots_arrays[-1]))
maxes = []
mins = []
dif = []
diffs = []
i = 0
for inner_list in neuron_dots_arrays:
    # print(len(inner_list), "inner_list = ", inner_list)
    real_dot = normal_real[i]
    for neuron_dot in inner_list:
        # print("real_dot = ", real_dot)
        # print("neuron_dot = ", neuron_dot)
        dif.append(abs(real_dot - neuron_dot))
    # print("dif = ", dif)
    #     break
    diffs.append(dif.copy())
    dif.clear()
    i += 1
print("len(diffs) = ", len(diffs))
print("len(diffs[0]) = ", len(diffs[0]))
for d in diffs:
    maxes.append(max(d))
    mins.append(min(d))
# real_plus_maxes = []
# real_minus_mins = []
# for dat in range(len(real_data)):
#     real_plus_maxes.append(normal_real[dat] + maxes[dat])
#     real_minus_mins.append(normal_real[dat] - mins[dat])
print("len(maxes) = ", len(maxes), maxes[:5])
print("len(mins) = ", len(mins), mins[:5])
times = []
for i in range(len(normal_real)):
    times.append(i)
print("times = ", times)
# plt.plot(normal_real, label='normal')
# plt.plot(maxes, label='maxes')
# plt.plot(mins, label='mins')

plt.figure(figsize=(9, 3))
# ppp.plot(times, normal_real, linewidth=1, color='gray')
plt.plot(maxes, label="maxes")
plt.plot(mins, label="mins")
# plt.fill_between(times, mins, maxes, alpha=0.35, color='red')
# ppp.plot(maxes, label='maxes')
# ppp.plot(mins, label='mins')
plt.legend()
plt.show()
