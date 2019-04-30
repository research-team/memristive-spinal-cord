from analysis.patterns_in_bio_data import bio_data_runs
from matplotlib import pylab as plt, pyplot
from math import sqrt
from cycler import cycler
from analysis.functions import bio_process, normalization
from analysis.neuron_data import neuron_data, neuron_20_runs
from scipy.interpolate import BarycentricInterpolator
from sympy import diff
from scipy.misc import derivative
import autograd.numpy as np
from autograd import grad

# importing the list of all runs of the bio  data from the function 'bio_data_runs'

# neuron = neuron_data()
neuron_20_runs = neuron_20_runs()
print("neuron_20_runs = ", len(neuron_20_runs), len(neuron_20_runs[0]))
# slices = neuron[0]
# a = neuron[1]
# b = neuron[2]

bio_runs = bio_data_runs()
# for i in range(len(bio_runs)):
# 	bio_runs[i] = normalization(bio_runs[i], a, b)
offset = 0
all_bio_slices = []
step = 0.25

# forming list for the plot
for k in range(len(bio_runs)):
	bio_slices = []
	offset = 0
	for i in range(int(len(bio_runs[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio_runs[k][j])
		bio_slices.append(bio_slices_tmp)
		offset += 100
	all_bio_slices.append(bio_slices)   # list [4][16][100]
print("all_bio_slices = ", all_bio_slices)
all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]

# error was found by Lesha
all_neuron_slices = []
for k in range(len(neuron_20_runs)):
	if neuron_20_runs[k]:
		neuron_slices = []
		offset = 0
		for i in range(int(len(neuron_20_runs[k]) / 1000)):
			neuron_slices_tmp = []
			for j in range(offset, offset + 1000):
				neuron_slices_tmp.append(neuron_20_runs[k][j])
			neuron_slices.append(neuron_slices_tmp)
			offset += 1000
		all_neuron_slices.append(neuron_slices)
# for run in neuron_20_runs:
# 	if run:
# 		slices_data = [run[slice_data_index:slice_data_index+1000] for slice_data_index in range(len(run))[::1000]]
# 		all_neuron_slices.append(slices_data)
all_neuron_slices = list(zip(*all_neuron_slices))
# like a zip
# new_one = [[] for _ in range(17)]
# for run in neuron_20_runs:
# 	for index_start in range(len(run))[::1000]:
# 		new_one[index_start // 1000].append(run[index_start:index_start+1000])

# print(list(zip(*all_neuron_slices)))
print("all_neuron_slices =", len(all_neuron_slices))
print("all_neuron_slices =", len(all_neuron_slices[0]))
print("all_neuron_slices =", len(all_neuron_slices[0][0]))

# calculating the instant dispersion
colors = ['black', 'rosybrown', 'firebrick', 'sandybrown', 'gold', 'olivedrab', # 6
          'darksalmon', 'green', 'sienna', 'darkblue', 'coral', 'orange',   # 12
          'darkkhaki', 'red', 'tan', 'steelblue', 'darkgreen', 'darkblue', 'palegreen', 'k', 'forestgreen',   # 21
          'slategray', 'limegreen', 'dimgrey', 'darkorange', 'darkgreen', 'cornflowerblue', 'dimgray', 'burlywood', # 29
          'royalblue', 'grey', 'g', 'gray', 'lime', 'midnightblue', 'seagreen', 'navy']   # 37
plt.rc('axes', prop_cycle=cycler(color=colors))

instant_mean = []
for slice in range(len(all_bio_slices)):
	instant_mean_sum = []
	for dot in range(len(all_bio_slices[slice][0])):
		instant_mean_tmp = []
		for run in range(len(all_bio_slices[slice])):
			instant_mean_tmp.append(abs(all_bio_slices[slice][run][dot]))
		instant_mean_sum.append(sum(instant_mean_tmp))
	instant_mean.append(instant_mean_sum)

# print("len(instant_mean) = ", len(instant_mean))
# print("instant_mean = ", instant_mean)

maxes = []
for sli in instant_mean:
	maxes.append(max(sli))
# for m in maxes:
	# print("maxes = ", m)

shifts = []
for m in maxes:
	shifts.append(0.105 * m)

# for s in shifts:
	# print("shifts = ", s)
# creating the list of dots of stimulations

stimulations = []
for stim in range(0, 1201, 100):
	stimulations.append(stim)

# creating the lists of voltages
volts = []
# interpolated_functions = []
for i in instant_mean:
	for j in i:
		volts.append(j)

# list for latencies' finding
volts_and_stims = [volts, stimulations]

# latencies finding
# latencies = bio_process(volts_and_stims, 12, reverse_ees=True)[0]
# print("latencies = ", latencies)
yticks = []
color_number = 0
for index, sl in enumerate(all_bio_slices):
	offset = index * 5
	# print("sl[{}][0]".format(run), sl[run][0])
	times = [time * step for time in range(len(all_bio_slices[0][0]))]
	for run in range(len(sl)):
		plt.plot(times, [s + offset for s in sl[run]], color=colors[color_number], linewidth=1)
	color_number += 1
	yticks.append(sl[run][0] + offset)

color_number = 12
# yticks = []
for index, sl in enumerate(all_neuron_slices):
	offset = index * 3
	# yticks.append(sl[run][0] + offset)
	times = [time * 0.025 for time in range(len(all_neuron_slices[0][0]))]
	for run in range(len(sl)):
		plt.plot(times, [s + offset for s in sl[run]], linewidth=1, color=colors[color_number])
	color_number -= 1
# yticks = []
times = []
step = 0.025
# for index, sl in enumerate(slices):
# 	offset = index * 16 + 8
	# yticks.append(sl[0] + offset)
	# times = [time * step for time in range(len(sl))]
	# plt.plot(times, [s + offset for s in sl])
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)])
# plt.yticks(yticks, range(1, len(slices) + 1))
# plt.xlim(0, 25)
# plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
# plt.show()

# plotting the dispersion
# yticks = []

# creating the list of x & y coordinates of the lines
x_coor = []
y_coor = []
# plotting of the dots that show the latencies
step = 0.25

latencies_values = []
# for index, sl in enumerate(instant_mean):
	# latencies_values.append(sl[int(latencies[index] / step)])
print("latencies_values = ", latencies_values)

necessary_points = []
for dot in range(len(latencies_values)):
	necessary_points.append(latencies_values[dot]- shifts[dot]) #
print("necessary_points = ", necessary_points)

necessary_latencies = []
count = 0
latency_x = []
for slice in range(len(instant_mean)):
	# for dot in range(int(latencies[slice] * 4), 36, -1):
	# 	print("dot = ", dot / 4)
		# print("instant_mean[{}][{}] = ".format(slice, dot), instant_mean[slice][dot])
		# if instant_mean[slice][dot] < necessary_points[count]:
		# 	necessary_latencies.append(instant_mean[slice][dot])
		# 	latency_x.append(dot / 4)
		# 	print("latency_x = ", latency_x)
		# 	break
	count += 1
	# if len(latency_x) != count:
	# 	latency_x.append(24.75)
print("necessary_latencies = ", necessary_latencies)
print("latency_x = ", latency_x)
for index, sl in enumerate(instant_mean):
	offset = index * 12
	# yticks.append(sl[0] + offset)
	times = [time * step for time in range(len(sl))]
	# plt.plot(times, [s + offset for s in sl], linewidth=2)
	# print("latencies[index] = ", latencies[index])
	# print("sl[int(latencies[index] / step)] = ", sl[int(latencies[index] / step)] )
	# plt.plot(latency_x[index], sl[int(latency_x[index] / step)] + offset, marker='.', markersize=12)
	# plt.text(latencies[index], sl[int(latencies[index] / step)] + offset, '{} %'.format(round(percentage[index]), 3),
	#          fontsize=16)
	# pltting of the lines
	# x_coor.append(latency_x[index])
	# y_coor.append(sl[int(latency_x[index] / step)] + offset)
	x_2_coors = []
	y_2_coors = []
	if len(x_coor) > 1:
		x_2_coors.append(x_coor[-2])
		x_2_coors.append(x_coor[-1])
		y_2_coors.append(y_coor[-2])
		y_2_coors.append(y_coor[-1])
		plt.plot(x_2_coors, y_2_coors, linestyle='--', color='black')
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)])
plt.yticks(yticks, range(1, len(instant_mean) + 1))
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()