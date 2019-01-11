import numpy as np
import pylab as plt
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.functions import read_nest_data, read_neuron_data, read_bio_data, normalization, find_latencies

# some constants and keys
bar_width = 0.2
bio_step = 0.25
sim_step = 0.025
gpu_step = 0.1
color_lat = '#F2AA2E'
color_amp = '#472650'

k_max_time = 0
k_max_val = 1
k_min_time = 2
k_min_val = 3

k_bio_volt = 0
k_bio_stim = 1

debugging_flag = True


def debug(voltages, datas, stim_indexes, ees_indexes, latencies, amplitudes, step):
	"""
	Temporal function for visualization of preprocessed data
	Args:
		voltages (list):
			voltage data
		datas (list of list):
			includes min/max time min/max value for each slice
		stim_indexes (list):
			indexes of EES stimlations
		ees_indexes (list):
			indexes of EES answers (mono-answer)
		latencies (list):
			latencies of the first poly-answers per slice
		amplitudes (list):
			amplitudes per slice
		step (float):
			 step size of the data
	"""
	amplitudes_y = []

	slice_indexes = range(len(ees_indexes))

	show_text = True
	show_amplitudes = True
	show_points = True
	show_axvlines = True

	# the 1st subplot demonstrates a voltage data, ees answers, ees stimulations and found latencies
	ax = plt.subplot(2, 1, 1)
	# plot the voltage data
	norm_voltages = normalization(voltages, zero_relative=True)

	plt.plot([t * step for t in range(len(norm_voltages))], norm_voltages, color='grey', linewidth=1)
	# standartization to the step size
	for slice_index in slice_indexes:
		datas[k_max_time][slice_index] = [d * step for d in datas[0][slice_index]]
	for slice_index in slice_indexes:
		datas[k_min_time][slice_index] = [d * step for d in datas[2][slice_index]]

	stim_indexes = [index * step for index in stim_indexes]
	ees_indexes = [index * step for index in ees_indexes]

	# plot the EES stimulation
	for i in stim_indexes:
		if show_axvlines:
			plt.axvline(x=i, linestyle='--', color='k', alpha=0.35, linewidth=1)
		if show_text:
			plt.annotate("Stim\n{:.2f}".format(i), xy=(i-1.3, 0), textcoords='data', color='k')
	# plot the EES answers
	for index, v in enumerate(ees_indexes):
		if show_axvlines:
			plt.axvline(x=v, color='r', alpha=0.5, linewidth=5)
		if show_text:
			plt.annotate("EES\n{:.2f}".format(v - stim_indexes[index]), xy=(v + 1, -0.2), textcoords='data', color='r')
	# plot the latencies
	for index, lat in enumerate(latencies):
		lat += ees_indexes[index]
		if show_axvlines:
			plt.axvline(x=lat, color='g', alpha=0.7, linewidth=2)
		if show_text:
			plt.annotate("from ees {:.2f}".format(lat - ees_indexes[index]), xy=(lat + 0.2, -0.4), textcoords='data',
			             color='g')
			plt.annotate("from stim: {:.2f}".format(lat - stim_indexes[index]), xy=(lat + 0.2, -0.5), textcoords='data',
			             color='g')

	# plot min/max points for each slice and calculate their amplitudes
	plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
	plt.axhline(y=-1, color='r', linestyle='--', linewidth=1)

	for slice_index in slice_indexes:
		min_times = datas[k_min_time][slice_index]
		min_values = datas[k_min_val][slice_index]
		max_times = datas[k_max_time][slice_index]
		max_values = datas[k_max_val][slice_index]
		amplitudes_y.append(amplitudes[slice_index])
		# plot them
		if show_points:
			plt.plot([kek + ees_indexes[slice_index] for kek in min_times], min_values, '.', color='b', markersize=5)
			plt.plot([kek + ees_indexes[slice_index] for kek in max_times], max_values, '.', color='r', markersize=5)
	plt.legend()

	# plot the amplitudes with shared x-axis
	plt.subplot(2, 1, 2, sharex=ax)
	# plot the EES answers
	if show_amplitudes:
		for i in ees_indexes:
			plt.axvline(x=i, color='r')
		# plot amplitudes by the horizontal line

		plt.bar([ees_index + ees_indexes[0] for ees_index in ees_indexes], amplitudes, width=5, color=color_lat,
		        alpha=0.7, zorder=2)
		for slice_index in slice_indexes:
			x = ees_indexes[slice_index] + ees_indexes[0] - 5 / 2
			y = amplitudes[slice_index]
			plt.annotate("{:.2f}".format(y), xy=(x, y + 0.01), textcoords='data')
		plt.ylim(0, 0.8)
	else:
		plt.plot([t * step for t in range(len(voltages))], voltages, color='grey', linewidth=1)
	plt.xlim(0, 150)
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


def calc_amplitudes(datas, latencies):
	"""
	Function for calculating amplitudes
	Args:
		datas (list of list):
			includes min/max time min/max value for each slice
		latencies (list):
			latencies pr slice for calculating only after the first poly-answer
	Returns:
		list: amplitudes per slice
	"""
	amplitudes = []
	slice_numbers = len(datas[0])

	for slice_index in range(slice_numbers):
		maxes_v = datas[k_max_val][slice_index]
		maxes_t = datas[k_max_time][slice_index]
		mins_v = datas[k_min_val][slice_index]
		mins_t = datas[k_min_time][slice_index]

		max_amp_in_maxes = max([abs(m) for index, m in enumerate(maxes_v) if maxes_t[index] >= latencies[slice_index]])
		max_amp_in_mins = max([abs(m) for index, m in enumerate(mins_v) if mins_t[index] >= latencies[slice_index]])

		amplitudes.append(max([max_amp_in_maxes, max_amp_in_mins]))

	if len(amplitudes) != slice_numbers:
		raise Exception("Length of amplitudes must be equal to slice numbers!")

	return amplitudes


# def processing_data(bio, nest_tests, neuron_tests):
def processing_data(bio, nest_tests):
# def processing_data(nest_tests, neuron_tests):
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
		tuple: bio_pack -- bio latencies and amplitudes
		tuple: nest_pack -- NEST latencies and amplitudes
		tuple: neuron_pack -- Neuron latencies and amplitudes
	"""
	# ToDo collapse the block of codes which answering for finding amp/lat

	# get the slice number in the data
	print("nest_tests = ", type(nest_tests))
	nest_means = list(map(lambda voltages: np.mean(voltages), zip(*nest_tests)))
	plt.plot(nest_means)
	plt.show()
	slice_numbers = int(len(nest_tests[0]) * sim_step // 25)
	# get bio voltages and EES stimulations from the argument
	bio_stim_indexes = bio[k_bio_stim][:slice_numbers + 1]
	# remove unescesary data
	bio_voltages = bio[k_bio_volt][:bio_stim_indexes[-1]]
	# calculate mean of voltages for simulators
	neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_tests)))
	nest_means = list(map(lambda voltages: np.mean(voltages), zip(*nest_tests)))
	# calculate EES stimulation indexes for the simulators
	sim_stim_indexes = list(range(0, len(nest_means), int(25 / sim_step)))

	'''block of code for finding latencies and amplitudes in BIO data'''
	# get the min/max extrema based on stimulation indexes
	bio_datas = calc_max_min(bio_stim_indexes, bio_voltages, bio_step)
	# find EES answers basing on min/max extrema
	bio_ees_indexes = find_ees_indexes(bio_stim_indexes[:-1], bio_datas)
	# normalize data
	norm_bio_voltages = normalization(bio_voltages, zero_relative=True)
	# get the min/max extrema based on EES answers indexes (because we need the data after 25ms of the slice)
	bio_datas = calc_max_min(bio_ees_indexes, norm_bio_voltages, bio_step, remove_micropeaks=True)
	# get the latencies and amplitudes based on min/max extrema
	bio_lat = find_latencies(bio_datas, bio_step, norm_to_ms=True)
	bio_amp = calc_amplitudes(bio_datas, bio_lat)

	# the steps are the same as above
	nest_datas = calc_max_min(sim_stim_indexes, nest_means, sim_step)


	print("sim_stim_indexes = ", sim_stim_indexes)
	print("nest_means = ", nest_means)
	nest_ees_indexes = find_ees_indexes(sim_stim_indexes, nest_datas)
	norm_nest_means = normalization(nest_means, zero_relative=True)
	nest_datas = calc_max_min(nest_ees_indexes, norm_nest_means, sim_step, remove_micropeaks=True)
	nest_lat = find_latencies(nest_datas, sim_step, norm_to_ms=True)
	nest_amp = calc_amplitudes(nest_datas, nest_lat)

	# the steps are the same as above
	neuron_datas = calc_max_min(sim_stim_indexes, neuron_means, sim_step)
	neuron_ees_indexes = find_ees_indexes(sim_stim_indexes, neuron_datas)
	norm_neuron_means = normalization(neuron_means, zero_relative=True)
	neuron_datas = calc_max_min(neuron_ees_indexes, norm_neuron_means, sim_step, remove_micropeaks=True)
	neuron_lat = find_latencies(neuron_datas, sim_step, with_afferent=True, norm_to_ms=True)
	neuron_amp = calc_amplitudes(neuron_datas, neuron_lat)

	if debugging_flag:
		debug(bio_voltages, bio_datas, bio_stim_indexes, bio_ees_indexes, bio_lat, bio_amp, bio_step)
		debug(nest_means, nest_datas, sim_stim_indexes, nest_ees_indexes, nest_lat, nest_amp, sim_step)
		debug(neuron_means, neuron_datas, sim_stim_indexes, neuron_ees_indexes, neuron_lat, neuron_amp, sim_step)

	# forming data 'packs'
	bio_pack = ("Biological", bio_lat, bio_amp)
	nest_pack = ("NEST", nest_lat, nest_amp)
	neuron_pack = ("Neuron", neuron_lat, neuron_amp)

	# return bio_pack, nest_pack, neuron_pack
	return bio_pack, nest_pack
	# return nest_pack, neuron_pack


# def draw_lat_amp(bio_pack, nest_pack, neuron_pack, plot_delta=False):
def draw_lat_amp(bio_pack, nest_pack, plot_delta=False):
# def draw_lat_amp(nest_pack, neuron_pack, plot_delta=False):
	"""
	Function for drawing latencies and amplitudes in one plot
	Args:
		bio_pack (tuple):
			biological data pack of latenccies and amplitudes
		nest_pack (tuple):
			NEST simulator data pack of latenccies and amplitudes
		neuron_pack (tuple):
			Neuron simulator data pack of latenccies and amplitudes
		plot_delta (bool):
			Plot and calculate delta of datas or their separated values
	"""
	if plot_delta:
		# for sim_pack in [nest_pack, neuron_pack]:
		for sim_pack in [bio_pack, nest_pack]:
			bio_title = bio_pack[0]
			bio_latencies = bio_pack[1]
			bio_amplitudes = bio_pack[2]

			sim_title = sim_pack[0]
			sim_latencies = sim_pack[1]
			sim_amplitudes = sim_pack[2]

			# create axes
			fig, lat_axes = plt.subplots(1, 1, figsize=(12, 9))
			plt.title("Delta of {} data and {} simulator".format(bio_title, sim_title))
			xticks = range(len(sim_amplitudes))

			lat_deltas = [abs(sim - bio) for sim, bio in zip(sim_latencies, bio_latencies)]
			lat_plot = lat_axes.bar(xticks, lat_deltas, width=bar_width, color=color_lat, alpha=0.7, zorder=2)
			lat_axes.set_xlabel('Slice')
			lat_axes.set_ylabel("Latency, ms")

			amp_axes = lat_axes.twinx()
			xticks = [x + bar_width for x in xticks]
			amp_deltas = [abs(sim - bio) for sim, bio in zip(sim_amplitudes, bio_amplitudes)]
			amp_plot = amp_axes.bar(xticks, amp_deltas, width=bar_width, color=color_amp, alpha=0.7, zorder=2)
			amp_axes.set_ylabel("Amplitude, mV")

			# plot text annotation for data
			for index in range(len(sim_amplitudes)):
				amp = amp_deltas[index]
				lat = lat_deltas[index]
				lat_axes.text(index - bar_width / 2, lat + max(lat_deltas) / 50, str(round(lat, 2)))
				amp_axes.text(index + bar_width / 2, amp + max(amp_deltas) / 50, str(round(amp, 2)))

			plt.legend((lat_plot, amp_plot), ("Latency", "Amplitude"), loc='best')
			plt.show()
	else:
		# for pack in [bio_pack, nest_pack, neuron_pack]:
		for pack in [bio_pack, nest_pack]:
		# for pack in [nest_pack, neuron_pack]:
			title = pack[0]
			latencies = pack[1]
			amplitudes = pack[2]

			# create axes
			fig, lat_axes = plt.subplots(1, 1, figsize=(15, 12))
			plt.title("{} data".format(title))
			xticks = range(len(amplitudes))

			lat_plot = lat_axes.bar(xticks, latencies, width=bar_width, color=color_lat, alpha=0.7, zorder=2)
			lat_axes.set_xlabel('Slice')
			lat_axes.set_ylabel("Latency, ms")

			amp_axes = lat_axes.twinx()
			xticks = [x + bar_width for x in xticks]
			amp_plot = amp_axes.bar(xticks, amplitudes, width=bar_width, color=color_amp, alpha=0.7, zorder=2)
			amp_axes.set_ylabel("Amplitude, mV")

			# plot text annotation for data
			for index in range(len(amplitudes)):
				amp = round(amplitudes[index], 2)
				lat = round(latencies[index], 2)
				lat_axes.text(index - bar_width / 2, lat + max(latencies) / 50, str(lat))
				amp_axes.text(index + bar_width / 2, amp + max(amplitudes) / 50, str(amp))

			plt.legend((lat_plot, amp_plot), ("Latency", "Amplitude"), loc='best')
			# plt.tick_params(labelsize=42)
			plt.show()


def run():
	bio = read_bio_data('../bio-data/3_1.31 volts-Rat-16_5-09-2017_RMG_9m-min_one_step.txt')
	nest_tests = read_nest_data('../../nest-data/21cms/extensor_21cms_40Hz_100inh.hdf5')
	neuron_tests = read_neuron_data('../../neuron-data/sim_healthy_neuron_extensor_eesF40_i100_s21cms_T_100runs.hdf5')

	# bio_pack, nest_pack, neuron_pack = processing_data(bio, nest_tests, neuron_tests)
	bio_pack, nest_pack = processing_data(bio, nest_tests)
	# nest_pack, neuron_pack = processing_data(nest_tests, neuron_tests)
	# bio_pack[2] = [round(i, 3) for i in bio_pack[2]]
	# print(bio_pack)
	# draw_lat_amp(bio_pack, nest_pack, neuron_pack, plot_delta=False)
	draw_lat_amp(bio_pack, nest_pack, plot_delta=False)
	# draw_lat_amp(nest_pack, neuron_pack, plot_delta=False)


if __name__ == "__main__":
	run()
