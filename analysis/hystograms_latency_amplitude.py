from analysis.functions import read_NEURON_data, list_to_dict, find_mins, read_NEST_data, read_bio_data
import numpy as np
from analysis.peaks_of_real_data_without_EES import delays
from analysis.max_min_values_neuron_nest import calc_max_min
from matplotlib import pylab as plt
import matplotlib.pyplot as pyp
import matplotlib.patches as mpatches
# read data
neuron_dict = read_NEURON_data('../../neuron-data/15cm.hdf5')
nest_dict = read_NEST_data('../../nest-data/sim_extensor_eesF40_i100_s21cms_T.hdf5')
bio = read_bio_data('../bio-data/3_0.91 volts-Rat-16_5-09-2017_RMG_13m-min_one_step.txt')
bio_data = bio[0]
bio_indexes = bio[1]
bio_step = bio_indexes[1] - bio_indexes[0]
bio_indexes_from_zero = []
bio_indexes_from_zero.append(0)
for i in range(len(bio_indexes) - 1):
    bio_indexes_from_zero.append(bio_indexes_from_zero[i] + bio_step)
bio_indexes[-1] = len(bio_data)
bio_data_by_slice = []
for j in range(len(bio_indexes_from_zero) - 2):
    bio_data_by_slice_tmp = []
    for i in range(bio_indexes_from_zero[j], bio_indexes_from_zero[j + 1]):
        bio_data_by_slice_tmp.append(bio_data[i])
    bio_data_by_slice.append(bio_data_by_slice_tmp)
yticks = []
for index, sl in enumerate(range(len(bio_data_by_slice))):
    offset = index * 2
    yticks.append(bio_data_by_slice[sl][0] + offset)
    # plt.plot([data + offset for data in bio_data_by_slice[sl]])
ticks = []
labels = []
for i in range(0, len(bio_data_by_slice[0]) + 1, 4):
    # plt.axvline(x = i, linestyle ='--', color='gray')
    ticks.append(i)
    labels.append(i * 0.25)
# plt.xticks(ticks, labels)
# plt.show()
neuron_list = list_to_dict(neuron_dict)
nest_list = list_to_dict(nest_dict)
neuron_means = list(map(lambda x: np.mean(x), zip(*neuron_list)))
nest_means = list(map(lambda x: np.mean(x), zip(*nest_list)))
slices_start_time_from_EES_neuron = find_mins(neuron_means, -13)[1]
slices_start_time_from_EES_nest = find_mins(nest_means, 30)[1]
step_neuron = slices_start_time_from_EES_neuron[1] - slices_start_time_from_EES_neuron[0]
step_nest = slices_start_time_from_EES_nest[1] - slices_start_time_from_EES_nest[0]
slices_start_time_from_EES_nest.append(slices_start_time_from_EES_nest[-1] + step_nest)
slices_start_time_neuron = []
slices_start_time_neuron.append(0)
for i in range(len(slices_start_time_from_EES_neuron)):
    slices_start_time_neuron.append(slices_start_time_neuron[i] + step_neuron)
plt.plot(neuron_means)
for i in slices_start_time_neuron:
    plt.axvline(x=i, linestyle='--', color='gray')
    plt.axhline(y=-8, linestyle='--', color='gray')
    plt.axhline(y=-3.8, linestyle='--', color='gray')
plt.show()
slices_start_time_nest = []
slices_start_time_nest.append(0)
for i in range(len(slices_start_time_from_EES_nest)):
    slices_start_time_nest.append(slices_start_time_nest[i] + step_nest)
data_step = 0.025
bio_data_step = 0.25
data_neuron = calc_max_min(slices_start_time_neuron, neuron_means, data_step)
data_nest = calc_max_min(slices_start_time_from_EES_nest, nest_means, data_step)
data_bio = calc_max_min(bio_indexes, bio_data, bio_data_step)
latency_neuron = delays(data_neuron[0], data_neuron[2], data_neuron[3], -8, -3.8, 'neuron')[1]
latency_nest = delays(data_nest[0], data_nest[2], data_nest[3], 30, 65, 'nest')[1]
latency_bio = delays(data_bio[0], data_bio[2], data_bio[3], -3, 1, 'bio')[1]
delta_latencies = []
for i in range(len(latency_nest)):
    delta_latencies.append(abs(latency_nest[i] - latency_bio[i]))
amplitudes_monosynaptic = []
amplitudes_neuron = []
amplitudes_nest = []
amplitudes_bio = []
all_delays_neuron = delays(data_neuron[0], data_neuron[2], data_neuron[3], -8, -3.8, 'neuron')[3]
all_delays_nest = delays(data_nest[0], data_nest[2], data_nest[3], 30, 65, 'nest')[3]
all_delays_bio = delays(data_bio[0], data_bio[2], data_bio[3], -3, 1, 'bio')[3]
for i in range(len(all_delays_neuron)):
    amplitudes_neuron.append(abs(np.mean(all_delays_neuron[i])))
max_amplitude_neuron = max(amplitudes_neuron)
for i in range(len(all_delays_nest)):
    amplitudes_nest.append(abs(np.mean(all_delays_nest[i])))
for i in range(len(all_delays_bio)):
    amplitudes_bio.append(abs(np.mean(all_delays_bio[i])))
delta_amplitudes = []
for i in range(len(amplitudes_nest)):
    delta_amplitudes.append(abs(amplitudes_nest[i] - amplitudes_bio[i]))
max_amplitude_nest= max(amplitudes_nest)
max_amplitude_bio = max(amplitudes_bio)
max_latency_neuron = max(latency_neuron)
max_latency_nest = max(latency_nest)
max_latency_bio = max(latency_bio)
scale_neuron = max_latency_neuron / max_amplitude_neuron
scale_nest = max_latency_nest / max_amplitude_nest
scale_bio = max_latency_bio / max_amplitude_bio
normal_amplitudes_neuron = []
normal_amplitudes_bio = []
for i in range(len(amplitudes_neuron)):
    normal_amplitudes_neuron.append(amplitudes_neuron[i] * scale_neuron)
normal_amplitudes_nest = []
for i in range(len(amplitudes_nest)):
    normal_amplitudes_nest.append(amplitudes_nest[i] * scale_nest)
for i in range(len(amplitudes_bio)):
    normal_amplitudes_bio.append(amplitudes_bio[i] * scale_bio)
scale_delta = max(delta_latencies) / max(delta_amplitudes)
normal_delta_amplitudes = []
for i in range(len(delta_amplitudes)):
    normal_delta_amplitudes.append(delta_amplitudes[i] * scale_delta)
ax = pyp.axes()
ax.yaxis.grid(True, zorder = 1)
xs = range(len(normal_delta_amplitudes))
pyp.bar([x for x in xs], delta_latencies, width=0.2, color='#F2AA2E', alpha=0.7, zorder=2)
pyp.bar([x + 0.2 for x in xs], normal_delta_amplitudes, width=0.2, color='#472650', alpha=0.7, zorder=2)
pyp.xticks(xs, [i + 1 for i in range(len(normal_delta_amplitudes))])
ax_01 = plt.axes()
ax_01.set_xlabel(u'Slice')
ax_01.set_ylabel(u'ΔLatency, ms')
ax_02 = ax_01.twinx()
ax_02.axis([-0.5, 6, min(delta_amplitudes), max(delta_amplitudes)])
ax_02.set_ylabel(u'ΔAmplitude, mV')
yellow_patch = mpatches.Patch(color='#F2AA2E', label='ΔLatency')
purple_patch = mpatches.Patch(color='#472650', label='ΔAmplitude')
pyp.legend(handles=[yellow_patch, purple_patch], loc='best')
pyp.show()
ax = pyp.axes()
ax.yaxis.grid(True, zorder = 1)
xs = range(len(amplitudes_neuron))