from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from matplotlib import pylab as plt
from cycler import cycler
import matplotlib.patches as mpatches
from analysis.shadows import debugging
from analysis.functions import find_min_diff, sim_process
from analysis.cut_several_steps_files import select_slices

# importing bio runs from the function 'bio_data_runs'
bio_runs = bio_data_runs()
bio_mean_data = list(map(lambda voltages: np.mean(voltages), zip(*bio_runs)))

# forming list for shadows plotting
all_bio_slices = []
step = 0.25
for k in range(len(bio_runs)):
	bio_slices= []
	offset= 0
	for i in range(int(len(bio_runs[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio_runs[k][j])
		bio_slices.append(bio_slices_tmp)
		offset += 100
	all_bio_slices.append(bio_slices)   # list [4][16][100]
all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]

neuron = select_slices('../../neuron-data/15FL_serotonin.hdf5', 0, 17000)
print("len(neuron) = ", len(neuron[0]))

all_neuron_slices = []
sim_step = 0.025
for k in range(len(neuron)):
	neuron_slices= []
	offset= 0
	for i in range(int(len(neuron[k]) / 1000)):
		neuron_slices_tmp = []
		for j in range(offset, offset + 1000):
			neuron_slices_tmp.append(neuron[k][j])
		neuron_slices.append(neuron_slices_tmp)
		offset += 1000
	all_neuron_slices.append(neuron_slices)   # list [4][16][100]
all_neuron_slices = list(zip(*all_neuron_slices)) # list [16][4][100]

print("len(all_neuron_slices) = ", len(all_neuron_slices))
print("len(all_neuron_slices[0]) = ", len(all_neuron_slices[0]))
print("len(all_neuron_slices[0][0]) = ", len(all_neuron_slices[0][0]))

yticks = []
times = [time * step for time in range(len(all_bio_slices[0][0]))]
colors = ['black', 'saddlebrown', 'firebrick', 'sandybrown', 'olivedrab']
texts = ['1', '2', '3', '4', '5']
patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]

plt.rc('axes', prop_cycle=cycler(color=colors))
for index, sl in enumerate(all_bio_slices):
	offset = index * 6
	# for run in range(len(sl)):
		# plt.plot(times, [s + offset for s in sl[run]])
# black_patches = mpatches.Patch(color='black', label='1')
# silver_patches = mpatches.Patch(color='silver', label='2')
# firebrick_patches = mpatches.Patch(color='firebrick', label='3')
# sandybrown_patches = mpatches.Patch(color='sandybrown', label='4')
# gold_patches = mpatches.Patch(color='gold', label='5')
# plt.legend(handles=patches, loc='upper right')
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
# plt.yticks(yticks, range(1, len(all_bio_slices) + 1), fontsize=14)
# plt.xlim(0, 25)
# plt.show()
# plot shadows
all_means = []
all_mins = []
all_maxes = []

colors = ['#ed553b', '#079294']

for index, run in enumerate(all_bio_slices):
	# print("index = ", index)
	# print(len(run), run)
	offset = index * 12
	# plt.plot([r + offset for r in run ])
	mean_data = list(map(lambda elements: np.mean(elements), zip(*run)))
	times = [time * step for time in range(len(mean_data))]
	means = [voltage + offset for voltage in mean_data]
	all_means.append(means)
	yticks.append(means[0])
	minimal_per_step = [min(a) for a in zip(*run)]
	all_mins.append([m + 10 for m in minimal_per_step])
	maximal_per_step = [max(a) for a in zip(*run)]
	all_maxes.append([m + 10 for m in maximal_per_step])
	plt.plot(times, means, linewidth=0.5, color='k')
	plt.fill_between(times, [mini + offset for mini in minimal_per_step],
	                 [maxi + offset for maxi in maximal_per_step], alpha=0.35, color=colors[0])

for index, run in enumerate(all_neuron_slices):
	offset = index * 12
	mean_data = list(map(lambda elements: np.mean(elements), zip(*run)))
	times = [time * sim_step for time in range(len(mean_data))]
	means = [voltage + offset for voltage in mean_data]
	# yticks.append(means[0])
	minimal_per_step = [min(a) for a in zip(*run)]
	maximal_per_step = [max(a) for a in zip(*run)]
	# plt.plot(times, means, linewidth=0.5, color='k')
	# plt.fill_between(times, [mini + offset for mini in minimal_per_step],
	#                  [maxi + offset for maxi in maximal_per_step], alpha=0.35, color=colors[1])



# print("necessary_values = ", necessary_values)
# print("necessary_indexes = ", necessary_indexes)
# print("min_indexes_in_min = ", min_indexes_in_min)
# print("min_indexes_in_max = ", min_indexes_in_max)
min_difference_indexes, max_difference_indexes, necessary_indexes = find_min_diff(all_maxes, all_mins, step)

lat = sim_process(bio_mean_data, step=0.25, inhibition_zero=True, first_kink=True)[0]
min_lat = sim_process(bio_mean_data, step=0.25, inhibition_zero=True)[0]
x_coor = []
y_coor = []

for index, run in enumerate(all_bio_slices):
	coord = int(lat[index] * 4)
	min_lat_coord = int(min_lat[index] * 4)
	offset = index * 12
	mean_data = list(map(lambda elements: np.mean(elements), zip(*run)))
	times = [time * step for time in range(len(mean_data))]
	# print("times = ", len(times), times)
	# print("necessary_indexes[{}] = ".format(index), necessary_indexes[index])
	plt.plot(times[min_difference_indexes[index]], mean_data[min_difference_indexes[index]] + offset, marker='.',
	         markersize=12, color='red')
	plt.plot(times[max_difference_indexes[index]], mean_data[max_difference_indexes[index]] + offset, marker='.',
	         markersize=12,	color='blue')
	plt.plot(times[necessary_indexes[index]], mean_data[necessary_indexes[index]] + offset, marker='.',
	         markersize=12,	color='black')
	plt.plot(times[coord], mean_data[coord] + offset, marker='s', markersize=12, color='k')
	plt.plot(times[min_lat_coord], mean_data[min_lat_coord] + offset, marker='d', markersize=12, color='k')
	x_coor.append(times[necessary_indexes[index]])
	y_coor.append(mean_data[necessary_indexes[index]] + offset)
	x_2_coors = []
	y_2_coors = []
	if len(x_coor) > 1:
		x_2_coors.append(x_coor[-2])
		x_2_coors.append(x_coor[-1])
		y_2_coors.append(y_coor[-2])
		y_2_coors.append(y_coor[-1])
		plt.plot(x_2_coors, y_2_coors, linestyle='--', color='black')
		# plt.plot(6, times[necessary_indexes[index]], color='green', marker='.', markersize=24)
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
plt.yticks(yticks, range(1, len(all_bio_slices) + 1), fontsize=14)
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
# all_maxes_sim, all_mins_sim = debugging()
plt.show()

# print("all_means = ", all_means)
# print("all_maxes = ", all_maxes)
# print("all_mins = ", all_mins)
