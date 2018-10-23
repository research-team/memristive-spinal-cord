import csv
import pylab as plt
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.peaks_of_real_data_without_EES import remove_ees_from_min_max, delays, calc_durations
from analysis.FFT import fast_fourier_transform
import numpy
import math
from mpl_toolkits.mplot3d import Axes3D

def find_mins(array):
    min_elems = []
    indexes = []
    for index_elem in range(1, len(array) - 1):
        if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]) and array[index_elem] < -0.5:
            min_elems.append(index_elem * recording_step)
            indexes.append(index_elem)
    return min_elems, indexes


recording_step = 0.25

with open('../bio-data//1_Rat-16_5-09-2017_RMG_9m-min_one_step.txt') as file:
    for i in range(6):
        file.readline()
    reader = csv.reader(file, delimiter='\t')
    grouped_elements_by_column = list(zip(*reader))

    raw_data_RMG = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[2]]
    data_stim = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[7]]

stimulations = find_mins(data_stim)[:-1][0]
# print("stimulations", stimulations)
indexes = find_mins(data_stim)[1]
# print("indexes", indexes)
data_RMG = []
for i in range(indexes[0], indexes[-1]):
    data_RMG.append(raw_data_RMG[i])
# print(len(raw_data_RMG), "raw_data_RMG", raw_data_RMG)
# print(len(data_RMG), "data_RMG", data_RMG)
freq = 4000
# length = len(data_RMG)

# print(len(data_RMG), "volt_data", data_RMG)
frequency = fast_fourier_transform(data_RMG)

# print(len(stimulations), "slices_begin_time (stimulations)", stimulations)
# print(len(data_RMG), "data_RMG", data_RMG)
for i in range(len(stimulations)):
    stimulations[i] = stimulations[i] * 4
# print(stimulations)
data = calc_max_min(stimulations, data_RMG, data_step=0.25)
# print(len(data[0]), "slices max time", data[0])
# print(len(data[1]), "slices max value", data[1])
# print(len(data[2]), "slices min time", data[2])
# print(len(data[3]), "slices min value", data[3])

sliced_RMG = []
times = [t * recording_step for t in range(len(data_RMG))]
# for stim_index in range(1, len(stimulations)):
    # print(stimulations[stim_index-1])
    # print(stimulations[stim_index])
    # print(times.index(stimulations[stim_index - 1]))
    # print(times.index(stimulations[stim_index]))
    # sliced_RMG.append(data_RMG[times.index(stimulations[stim_index-1]):times.index(stimulations[stim_index])])
# for slice_index in range(len(data[0])):
#     slice_index += 1
    # plt.plot(data[0][slice_index], data[1][slice_index], ".", color='r', markersize='5')
    # plt.plot(data[2][slice_index], data[3][slice_index], ".", color='b', markersize='5')
    # plt.plot([t * recording_step for t in range(len(sliced_RMG[slice_index]))], sliced_RMG[slice_index])
    # plt.show()

data_with_deleted_ees = remove_ees_from_min_max(data[0], data[1], data[2], data[3])
max_min_delays = delays(data_with_deleted_ees[0], data_with_deleted_ees[2])
max_min_durations = calc_durations(data_with_deleted_ees[0], data_with_deleted_ees[2])
ds = max_min_delays[0]
ls = max_min_durations[0]
# print(len(ds), "ds", ds)
# print(len(ls), "ls", ls)


freq = 4000
# length = len(data_RMG)

# print(len(data_RMG), "volt_data", data_RMG)
frequency = []
offset = int(stimulations[1] - stimulations[0])
start = 0
for j in range(len(stimulations) - 2):
    sliced_data = data_RMG[start:start + offset]
    length = len(sliced_data)
    frequency.append(fast_fourier_transform(sliced_data))
    start += offset
# print(len(frequency), "frequency", frequency)
# plt.xlim(0, len(data_RMG) * recording_step)
# plt.plot([t * recording_step for t in range(len(data_RMG))], data_RMG)
# print("stimulations", stimulations)
# for i in stimulations:
#     plt.axvline(x=i, linestyle="--", color="gray")
# plt.xticks(stimulations)
# plt.xlabel("Time (ms)")
# plt.show()

# plt.plot(frequency)
# plt.show()
a = numpy.fft.rfftfreq(length, 1. / freq)
peaks = []
testing_points = []
# print("numpy.abs(frequency)", len(numpy.abs(frequency)))
# for i in range(len(numpy.abs(frequency))):
#     print(numpy.abs(frequency)[i])
# for i in range(len(a)):
# 	logging.debug(a[i])
# logging.debug(len(numpy.fft.rfftfreq((length ) - 1, 1. / FD)))
# logging.debug(len(numpy.abs(frequency)))
# logging.debug(len(frequency))
# plt.xlim(0, 500)
# plt.title("1_Rat-16_5-09-2017_RMG_9m-min_one_step")
# plt.plot(numpy.fft.fftfreq(length, 1. / freq), numpy.abs(frequency))
# print(len(a))
# for i in range(len(a)):
#     print("a", a[i])
# print(len(stimulations), "stimulations", stimulations)
offset = int(stimulations[1] - stimulations[0])
start = 0
for i in range(len(frequency)):
    peaks.append(max(numpy.abs(frequency[i])))
# print("peaks", len(peaks), peaks)
modified_peaks = []
log_peaks = []
for i in range(len(peaks)):
    modified_peaks.append(peaks[i] * 0.1)
    log_peaks.append(math.log10(modified_peaks[i]))
# print(len(modified_peaks), "modified_peaks", modified_peaks)
# print(len(log_peaks), "log_peaks", log_peaks)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(ds, ls, log_peaks, lw=0.5)
ax.plot(ds, ls, log_peaks, '.', lw=0.5, color='r', markersize=5)
ax.set_xlabel("Delay ms")
ax.set_ylabel("Duration ms")
ax.set_zlabel("Voltage mV")
ax.set_title("Delay - Duration - Voltage")
plt.show()