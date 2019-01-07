import numpy as np
import pylab as plt
import matplotlib.patches as mpatches
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.functions import read_neuron_data, list_to_dict, find_mins, read_nest_data, read_bio_data, normalization, find_latencies

# some constants and keys
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


def debug(voltages, datas, orig_stim_indexes, ees_indexes, orig_latencies, step):
	"""
	Temporal function for visualization of preprocessed data
	Args:
		voltages (list):
			voltage data
		datas (list of lists):
			includes min/max time min/max value for each slice
		orig_stim_indexes (list):
			indexes of EES stimlations
		ees_indexes (list):
			indexes of EES answers (mono-answer)
		orig_latencies (list):
			latencies of the first poly-answers for each slice
		step (float):
			 step size of the data
	"""
	amplitudes_y = []
	amplitudes_x = []

	# the 1st subplot demonstrates a voltage data, ees answers, ees stimulations and found latencies
	ax = plt.subplot(2, 1, 1)
	# plot the voltage data
	plt.plot([t * step for t in range(len(voltages))], voltages, color='grey', linewidth=1)
	# standartization to the step size
	for slice_index in range(len(datas[0])):
		datas[0][slice_index] = [d * step for d in datas[0][slice_index]]
	for slice_index in range(len(datas[2])):
		datas[2][slice_index] = [d * step for d in datas[2][slice_index]]
	stim_indexes = list(orig_stim_indexes)
	for i in range(len(stim_indexes)):
		stim_indexes[i] *= step
	latencies = list(orig_latencies)
	for i in range(len(ees_indexes)):
		ees_indexes[i] *= step
		latencies[i] *= step
	# plot the EES stimulation
	for i in stim_indexes:
		plt.axvline(x=i, linestyle='--', color='k', alpha=0.35, linewidth=1)
		plt.annotate("Stim\n{:.2f}".format(i), xy=(i-1.3, 0), textcoords='data', color='k')
	# plot the EES answers
	for index, v in enumerate(ees_indexes):
		plt.axvline(x=v, color='r', alpha=0.5, linewidth=5)
		plt.annotate("EES\n{:.2f}".format(v - stim_indexes[index]), xy=(v + 1, -0.2), textcoords='data', color='r')
	# plot the latencies
	for index, lat in enumerate(latencies):
		lat += ees_indexes[index]
		plt.axvline(x=lat, color='g', alpha=0.7, linewidth=2)
		plt.annotate("from ees {:.2f}".format(lat - ees_indexes[index]), xy=(lat + 0.2, -0.4), textcoords='data', color='g')
		plt.annotate("from stim: {:.2f}".format(lat - stim_indexes[index]), xy=(lat + 0.2, -0.5), textcoords='data', color='g')
	# plot min/max points for each slice and calculate their amplitudes
	for slice_index in range(len(ees_indexes)):
		min_times = datas[k_min_time][slice_index]
		min_values = datas[k_min_val][slice_index]
		max_times = datas[k_max_time][slice_index]
		max_values = datas[k_max_val][slice_index]
		# calc an amplitude
		amplitudes = [round(abs(minimal - maximal), 3) for minimal, maximal in zip(min_values, max_values)]
		mean_amp = round(np.mean(amplitudes), 3)
		print("AMP slice {}, len {}, mean {}, {}".format(slice_index, len(amplitudes), mean_amp, amplitudes))
		if slice_index == len(ees_indexes)-1:
			amplitudes_x.append(np.arange(ees_indexes[slice_index], ees_indexes[slice_index] + 25))
		else:
			amplitudes_x.append(np.arange(ees_indexes[slice_index], ees_indexes[slice_index+1]))
		amplitudes_y.append(mean_amp)
		# plot them
		plt.plot([kek + ees_indexes[slice_index] for kek in min_times], min_values, '.', color='b', markersize=5)
		plt.plot([kek + ees_indexes[slice_index] for kek in max_times], max_values, '.', color='r', markersize=5)
	plt.legend()

	# plot the amplitudes with shared x-axis
	plt.subplot(2, 1, 2, sharex=ax)
	# plot the EES answers
	for i in ees_indexes:
		plt.axvline(x=i, color='r')
	# plot amplitudes by the horizontal line
	for x, y in zip(amplitudes_x, amplitudes_y):
		plt.plot(x, [y]*len(x), color='g')
		plt.annotate("{:.2f}".format(y), xy=(np.mean(x), y + 0.05), textcoords='data')
	plt.xlim(0, 150)
	plt.ylim(0, 1)
	plt.show()
	plt.close()


def find_ees_indexes(stim_indexes, datas):
	"""
	Function for finding the indexes of the EES mono-answer in borders formed by stimulations time
	Args:
		stim_indexes (list):
			indexes of the EES stimulations
		datas (list of list):
			includes min/max times and min/max values
	Returns:
		list: global indexes of the EES mono-answers
	"""
	ees_indexes = []
	for slice_index in range(len(stim_indexes)):
		min_values = datas[k_min_val][slice_index]
		min_times = datas[k_min_time][slice_index]
		# EES peak is the minimal one
		ees_value_index = min_values.index(min(min_values))
		# calculate the EES answer as the sum of the local time of the found EES peak (of the layer)
		# and global time of stimulation for this layer
		ees_indexes.append(stim_indexes[slice_index] + min_times[ees_value_index])
	return ees_indexes


def processing_data(neuron_tests, nest_tests, bio):
	"""
	Function for demonstrating latencies/amplutudes for each simulator and the bio data
	Args:
		neuron_tests (list of list):
			voltages per test
		nest_tests (list of list):
			voltages per test
		bio (list):
			voltages data
	Returns:
		list: amplitudes
		list: latencies
	"""
	# ToDo collapse the block of codes which answering for finding amp/lat

	# get the slice number in the data
	slice_numbers = int(len(neuron_tests[0]) * sim_step // 25)
	# get bio voltages and EES stimulations from the argument
	bio_voltages = bio[k_bio_volt]
	bio_stim_indexes = bio[k_bio_stim][:-1]
	# calculate mean of voltages for simulators
	neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_tests)))
	nest_means = list(map(lambda voltages: np.mean(voltages), zip(*nest_tests)))
	# calculate EES stimulation indexes for the simulators
	sim_stim_indexes = list(range(0, len(nest_means), int(25 / sim_step)))

	'''block of code for finding latencies and amplitudes in BIO data'''
	# get the min/max extrema based on stimulation indexes
	bio_datas = calc_max_min(bio_stim_indexes, bio_voltages, bio_step)
	# find EES answers basing on min/max extrema
	bio_ees_indexes = find_ees_indexes(bio_stim_indexes, bio_datas)
	# remove unnesesary bio data (after the last EES answer)
	bio_voltages = bio_voltages[:bio_ees_indexes[slice_numbers]]
	bio_ees_indexes = bio_ees_indexes[:slice_numbers]
	# normalize data
	bio_voltages = normalization(bio_voltages, -1, 1)
	# get the min/max extrema based on EES answers indexes (because we need the data after 25ms of the slice)
	bio_datas = calc_max_min(bio_ees_indexes, bio_voltages, bio_step, remove_micropeaks=True)
	# get the latencies based on min/max extrema
	bio_lat = find_latencies(bio_datas, bio_step)
	# demonstrate the reuslts
	debug(bio_voltages, bio_datas, bio_stim_indexes, bio_ees_indexes, bio_lat, bio_step)

	'''block of code for finding latencies and amplitudes in NEST data'''
	# the steps are the same as above
	nest_datas = calc_max_min(sim_stim_indexes, nest_means, sim_step)
	nest_ees_indexes = find_ees_indexes(sim_stim_indexes, nest_datas)
	nest_means = normalization(nest_means, -1, 1)
	nest_datas = calc_max_min(nest_ees_indexes, nest_means, sim_step, remove_micropeaks=True)
	nest_lat = find_latencies(nest_datas, sim_step)

	debug(nest_means, nest_datas, sim_stim_indexes, nest_ees_indexes, nest_lat, sim_step)

	'''block of code for finding latencies and amplitudes in NEURON data'''
	# the steps are the same as above
	neuron_datas = calc_max_min(sim_stim_indexes, neuron_means, sim_step)
	neuron_ees_indexes = find_ees_indexes(sim_stim_indexes, neuron_datas)
	neuron_means = normalization(neuron_means, -1, 1)
	neuron_datas = calc_max_min(neuron_ees_indexes, neuron_means, sim_step, remove_micropeaks=True)
	neuron_lat = find_latencies(neuron_datas, sim_step, with_afferent=True)

	debug(neuron_means, neuron_datas, sim_stim_indexes, neuron_ees_indexes, neuron_lat, sim_step)

	# plot latency
	plt.bar(range(len(bio_lat)), [d * bio_step for d in bio_lat],
	        width=bar_width, color='b', alpha=0.7, label="biological")
	plt.bar([d + bar_width for d in range(len(nest_lat))], [d * sim_step for d in nest_lat],
	        width=bar_width, color='r', alpha=0.7, label="NEST")
	plt.bar([d + 2 * bar_width for d in range(len(neuron_lat))], [d * sim_step for d in neuron_lat],
	        width=bar_width, color='g', alpha=0.7, label="Neuron")
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
	todo add description
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
