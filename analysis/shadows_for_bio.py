from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from matplotlib import pylab as plt
from cycler import cycler
import matplotlib.patches as mpatches
from analysis.shadows import debugging
from analysis.functions import find_min_diff, sim_process
from analysis.cut_several_steps_files import select_slices
from GRAS.shadows_boxplot import calc_boxplots

split_coef = 8
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

neuron = select_slices('../../neuron-data/mn_E25tests (8).hdf5', 0, 6000)
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

high_box_Q3 = []
low_box_Q1 = []

colors = ['#ed553b', '#079294']
bio_data_np = np.array(bio_runs)
for dot in bio_data_np.T:
	boxplot_data = calc_boxplots(dot)
	high_box_Q3.append(boxplot_data[1])
	low_box_Q1.append(boxplot_data[2])

print("len(bio_runs) = ", len(all_bio_slices))
high_box_Q3_slices = []
offset = 0
for sl in range(len(all_bio_slices)):
	high_box_Q3_slices_tmp = []
	for dot in range(offset, offset + 100):
		high_box_Q3_slices_tmp.append(high_box_Q3[dot])
	high_box_Q3_slices.append(high_box_Q3_slices_tmp)
	offset += 100

low_box_Q1_slices = []
offset = 0
for sl in range(len(all_bio_slices)):
	low_box_Q1_slices_tmp = []
	for dot in range(offset, offset + 100):
		low_box_Q1_slices_tmp.append(low_box_Q1[dot])
	low_box_Q1_slices.append(low_box_Q1_slices_tmp)
	offset += 100
high_box_Q3_neuron = []
low_box_Q1_neuron = []
neuron_data_np = np.array(neuron)
for dot in neuron_data_np.T:
	boxplot_data = calc_boxplots(dot)
	high_box_Q3_neuron.append(boxplot_data[1])
	low_box_Q1_neuron.append(boxplot_data[2])

high_box_Q3_neuron_slices = []
offset = 0
for sl in range(len(all_neuron_slices)):
	high_box_Q3_slices_tmp = []
	for dot in range(offset, offset + 1000):
		high_box_Q3_slices_tmp.append(high_box_Q3_neuron[dot])
	high_box_Q3_neuron_slices.append(high_box_Q3_slices_tmp)
	offset += 1000

low_box_Q1_neuron_slices = []
offset = 0
for sl in range(len(all_neuron_slices)):
	low_box_Q1_slices_tmp = []
	for dot in range(offset, offset + 1000):
		low_box_Q1_slices_tmp.append(low_box_Q1_neuron[dot])
	low_box_Q1_neuron_slices.append(low_box_Q1_slices_tmp)
	offset += 1000

for index, run in enumerate(all_bio_slices):
	offset = index * split_coef
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
	plt.fill_between(times, [mini + offset for mini in low_box_Q1_slices[index]],
	                 [maxi + offset for maxi in high_box_Q3_slices[index]], alpha=0.35, color=colors[0])

# yticks = []
for index, run in enumerate(all_neuron_slices):
	offset = index * 6
	mean_data = list(map(lambda elements: np.mean(elements), zip(*run)))
	times = [time * sim_step for time in range(len(mean_data))]
	means = [voltage + offset for voltage in mean_data]
	# yticks.append(means[0])
	minimal_per_step = [min(a) for a in zip(*run)]
	maximal_per_step = [max(a) for a in zip(*run)]
	# plt.plot(times, means, linewidth=0.5, color='k')
	# plt.fill_between(times, [mini + offset for mini in minimal_per_step],
	#                  [maxi + offset for maxi in maximal_per_step], alpha=0.35, color=colors[1])


lat = sim_process(bio_mean_data, step=0.25, inhibition_zero=True, first_kink=True)[0]
min_lat = sim_process(bio_mean_data, step=0.25, inhibition_zero=True)[0]
print("len(all_maxes) = ", len(all_maxes[0]))
# min_difference_indexes, max_difference_indexes, necessary_indexes = \
# 	find_min_diff(all_maxes, all_mins, step, lat, from_first_kink=True)

min_difference_indexes, max_difference_indexes, necessary_indexes = \
	find_min_diff(high_box_Q3_slices, low_box_Q1_slices, step, lat, from_first_kink=False)

neuron_min_difference_indexes, neuron_max_difference_indexes, neuron_necessary_indexes = \
	find_min_diff(high_box_Q3_neuron_slices, low_box_Q1_neuron_slices, sim_step, lat, from_first_kink=False)

x_coor = []
y_coor = []

for index, run in enumerate(all_bio_slices):
	coord = int(lat[index] * 4)
	min_lat_coord = int(min_lat[index] * 4)
	offset = index * split_coef
	mean_data = list(map(lambda elements: np.mean(elements), zip(*run)))
	times = [time * step for time in range(len(mean_data))]
	plt.plot(times[min_difference_indexes[index]], mean_data[min_difference_indexes[index]] + offset, marker='.',
	         markersize=12, color='red')
	plt.plot(times[max_difference_indexes[index]], mean_data[max_difference_indexes[index]] + offset, marker='.',
	         markersize=12,	color='blue')
	plt.plot(times[necessary_indexes[index]], mean_data[necessary_indexes[index]] + offset, marker='.',
	         markersize=12,	color='black')
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
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
plt.yticks(yticks, range(1, len(all_bio_slices) + 1), fontsize=14)
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()

for index, run in enumerate(all_neuron_slices):
	coord = int(lat[index] * 4)
	min_lat_coord = int(min_lat[index] * 4)
	offset = index * 6
	mean_data = list(map(lambda elements: np.mean(elements), zip(*run)))
	times = [time * sim_step for time in range(len(mean_data))]
	# plt.plot(times[neuron_min_difference_indexes[index]], mean_data[neuron_min_difference_indexes[index]] + offset,
	#          marker='.', markersize=12, color='red')
	# plt.plot(times[neuron_max_difference_indexes[index]], mean_data[neuron_max_difference_indexes[index]] + offset,
	#          marker='.', markersize=12,	color='blue')
	# plt.plot(times[neuron_necessary_indexes[index]], mean_data[neuron_necessary_indexes[index]] + offset, marker='.',
	#          markersize=12,	color='black')
	x_coor.append(times[neuron_necessary_indexes[index]])
	y_coor.append(mean_data[neuron_necessary_indexes[index]] + offset)
	x_2_coors = []
	y_2_coors = []
	if len(x_coor) > 1:
		x_2_coors.append(x_coor[-2])
		x_2_coors.append(x_coor[-1])
		y_2_coors.append(y_coor[-2])
		y_2_coors.append(y_coor[-1])
		# plt.plot(x_2_coors, y_2_coors, linestyle='--', color='black')
		# plt.plot(6, times[necessary_indexes[index]], color='green', marker='.', markersize=24)
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
# plt.yticks(yticks, range(1, len(all_bio_slices) + 1), fontsize=14)
# plt.xlim(0, 25)
# plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
# all_maxes_sim, all_mins_sim = debugging()
# plt.show()

# print("all_means = ", all_means)
# print("all_maxes = ", all_maxes)
# print("all_mins = ", all_mins)
