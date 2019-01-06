import numpy as np
import pylab as plt
import matplotlib.patches as mpatches
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.functions import read_neuron_data, list_to_dict, find_mins, \
	read_nest_data, read_bio_data, normalization_between, alex_latency, find_latencies

bar_width = 0.2
bio_step = 0.25
sim_step = 0.025
color_lat = '#F2AA2E'
color_amp = '#472650'

k_max_time = 0
k_max_val = 1
k_min_time = 2
k_min_val = 3

k_bio_volt = 0
k_bio_stim = 1

def debug(voltages, datas, stim_indexes, ees_indexes, latencies):
	amplitudes_y = []
	amplitudes_x = []

	ax = plt.subplot(2, 1, 1)
	plt.plot(voltages, color='grey', linewidth=1)

	for i in stim_indexes:
		plt.axvline(x=i, linestyle='--', color='k', alpha=0.2, linewidth=0.5)

	for i in ees_indexes:
		plt.axvline(x=i, color='r', alpha=0.5, linewidth=5)

	for index, lat in enumerate(latencies):
		plt.axvline(x=lat + ees_indexes[index], color='g', alpha=0.7, linewidth=2)

	for slice_index in range(len(ees_indexes)):
		print(slice_index)

		# plot mins
		min_times = datas[k_min_time][slice_index]
		min_values = datas[k_min_val][slice_index]
		max_times = datas[k_max_time][slice_index]
		max_values = datas[k_max_val][slice_index]
		print("MIN slice {}, len {}, {}".format(slice_index, len(min_values), min_values))
		print("MAX slice {}, len {}, {}".format(slice_index, len(max_values), max_values))
		amplitudes = [round(abs(minimal - maximal), 3) for minimal, maximal in zip(min_values, max_values)]
		mean_amp = round(np.mean(amplitudes), 3)
		print("AMP slice {}, len {}, mean {}, {}".format(slice_index, len(amplitudes), mean_amp, amplitudes))
		print("LAT slice {}, len {} {}".format(slice_index, len(latencies), latencies))
		amplitudes_y.append(mean_amp)
		if slice_index == len(ees_indexes)-1:
			amplitudes_x.append(range(ees_indexes[slice_index], ees_indexes[slice_index] + int(25 / sim_step)))
		else:
			amplitudes_x.append(range(ees_indexes[slice_index], ees_indexes[slice_index+1]))
		plt.plot([kek + ees_indexes[slice_index] for kek in min_times], min_values, '.', color='b', markersize=5)
		plt.plot([kek + ees_indexes[slice_index] for kek in max_times], max_values, '.', color='r', markersize=5)
#		plt.ylim(-1, 1)
	plt.legend()
	plt.subplot(2, 1, 2, sharex=ax)
	for i in ees_indexes:
		plt.axvline(x=i, color='r')
	print("- " * 10)
	for x, y in zip(amplitudes_x, amplitudes_y):
		print(x, y)
		plt.plot(x, [y]*len(x), color='g')
		plt.annotate("{:.2f}".format(y), xy=(np.mean(x)-25, y + 0.05), textcoords='data')
	plt.xlim(0, len(voltages))
	plt.ylim(0, 1)
	plt.show()
	plt.close()


def find_ees_indexes(stim_indexes, datas):
	ees_indexes = []
	for slice_index in range(len(stim_indexes)):
		# EES peak is the most minimal, and can be the secondary min peak in the slice (!)
		ees_value_index = datas[k_min_val][slice_index].index(min(datas[k_min_val][slice_index]))
		# add index of EES peak. Index is found by minimal EES peak value
		ees_indexes.append(stim_indexes[slice_index] + datas[k_min_time][slice_index][ees_value_index])
	return ees_indexes


def processing_data(neuron_tests, nest_tests, bio):
	slice_numbers = int(len(neuron_tests[0]) * sim_step // 25)
	bio_voltages = bio[k_bio_volt]
	bio_stim_indexes = bio[k_bio_stim][:-1]
	# calculate mean of voltages
	neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_tests)))
	nest_means = list(map(lambda voltages: np.mean(voltages), zip(*nest_tests)))
	# calculate EES stim indexes
	sim_stim_indexes = range(0, len(nest_means), int(25 / sim_step))

	bio_datas = calc_max_min(bio_stim_indexes, bio_voltages)
	bio_ees_indexes = find_ees_indexes(bio_stim_indexes, bio_datas)

	# remove unnesesary bio data
	index_last_useful_slice = bio_ees_indexes[slice_numbers]
	bio_voltages = bio_voltages[:index_last_useful_slice]
	bio_ees_indexes = bio_ees_indexes[:slice_numbers+1] # +1 because we need to get over 25ms points

	bio_voltages = normalization_between(bio_voltages, -1, 1)
	bio_datas = calc_max_min(bio_ees_indexes, bio_voltages, remove_micro=True)
	# fixme empty last unnecesary slice
	bio_lat = find_latencies(bio_datas)

	debug(bio_voltages, bio_datas, bio_stim_indexes, bio_ees_indexes, bio_lat)

	nest_datas = calc_max_min(sim_stim_indexes, nest_means)
	nest_ees_indexes = find_ees_indexes(sim_stim_indexes, nest_datas)
	nest_means = normalization_between(nest_means, -1, 1)
	nest_datas = calc_max_min(nest_ees_indexes, nest_means, remove_micro=True)
	nest_lat = find_latencies(nest_datas)

	debug(nest_means, nest_datas, sim_stim_indexes, nest_ees_indexes, nest_lat)

	neuron_datas = calc_max_min(sim_stim_indexes, neuron_means)
	neuron_ees_indexes = find_ees_indexes(sim_stim_indexes, neuron_datas)
	neuron_means = normalization_between(neuron_means, -1, 1)
	neuron_datas = calc_max_min(neuron_ees_indexes, neuron_means, remove_micro=True)
	neuron_lat = find_latencies(neuron_datas, with_afferent=True)

	debug(neuron_means, neuron_datas, sim_stim_indexes, neuron_ees_indexes, neuron_lat)

	# plot latency
	plt.bar(range(len(bio_lat)), [d * bio_step for d in bio_lat],
	        width=bar_width, color='b', alpha=0.7, zorder=2, label="biological")
	plt.bar([d + bar_width for d in range(len(nest_lat))], [d * sim_step for d in nest_lat],
	        width=bar_width, color='r', alpha=0.7, zorder=2, label="NEST")
	plt.bar([d + 2 * bar_width for d in range(len(neuron_lat))], [d * sim_step for d in neuron_lat],
	        width=bar_width, color='g', alpha=0.7, zorder=2, label="Neuron")
	plt.legend()
	plt.show()

	raise Exception

	latencies_delta = [abs(nest - bio) for nest, bio in zip(latencies_nest, latencies_bio)]



	amplitudes_neuron = [abs(np.mean(element)) for element in all_delays_neuron]
	amplitudes_nest = [abs(np.mean(element)) for element in all_delays_nest]
	amplitudes_bio = [abs(np.mean(element)) for element in all_delays_bio]

	amplitudes_delta = [abs(nest - bio) for nest, bio in zip(amplitudes_nest, amplitudes_bio)]

	scale_neuron = max(latencies_neuron) / max(amplitudes_neuron)
	scale_nest = max(latencies_nest) / max(amplitudes_nest)
	scale_bio = max(latencies_bio) / max(amplitudes_bio)
	scale_delta = max(latencies_delta) / max(amplitudes_delta)

	normal_amplitudes_neuron = [amplitude * scale_neuron for amplitude in amplitudes_neuron]
	normal_amplitudes_nest = [amplitude * scale_nest for amplitude in amplitudes_nest]
	normal_amplitudes_bio = [amplitude * scale_bio for amplitude in amplitudes_bio]
	normal_delta_amplitudes = [amplitude * scale_delta for amplitude in amplitudes_delta]

	return normal_delta_amplitudes, amplitudes_delta, latencies_delta


def draw(normal_delta_amplitudes, amplitudes_delta, latencies_delta):
	"""
	# todo add description
	Args:
		normal_delta_amplitudes (list):
			fff
		amplitudes_delta (list):
			fff
		latencies_delta (list):
			fff
	"""
	# create axes
	ax = plt.axes()
	ax.yaxis.grid()
	xticks = range(len(normal_delta_amplitudes))
	# plot latency
	plt.bar(xticks, latencies_delta, width=bar_width, color=color_lat, alpha=0.7, zorder=2)
	# plot amplitudes
	plt.bar([x + bar_width for x in xticks], normal_delta_amplitudes, width=bar_width, color=color_amp, alpha=0.7, zorder=2)
	# set xticks by slice numbers
	plt.xticks(xticks, range(1, len(normal_delta_amplitudes) + 1))
	# create second axes
	axis_latency = plt.axes()
	axis_latency.set_xlabel("Slice")
	axis_latency.set_ylabel("Δ Latency, ms")
	# share x-axis
	axis_amplitude = axis_latency.twinx()
	axis_amplitude.set_ylabel("Δ Amplitude, mV")
	# axis_amplitude.axis([-0.5, 6, min(amplitudes_delta), max(amplitudes_delta)])
	# add legends
	lat_legend = mpatches.Patch(color=color_lat, label="Δ Latency")
	amp_legend = mpatches.Patch(color=color_amp, label="Δ Amplitude")
	plt.legend(handles=[lat_legend, amp_legend], loc='best')
	plt.show()


def run():
	neuron_tests = read_neuron_data('/home/alex/neuron_21cms.hdf5')
	nest_tests = read_nest_data('/home/alex/nest_21cms.hdf5')
	bio = read_bio_data('/home/alex/bio_21cms.txt')

	data = processing_data(neuron_tests, nest_tests, bio)

	draw(*data)


if __name__ == "__main__":
	run()
