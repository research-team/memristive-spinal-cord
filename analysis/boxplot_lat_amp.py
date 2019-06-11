from matplotlib import pylab as plt
import matplotlib.patches as mpatches
from analysis.functions import read_nest_data, read_neuron_data
from analysis.histogram_lat_amp import sim_process
from analysis.namespaces import *
from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from analysis.cut_several_steps_files import select_slices
from analysis.functions import normalization

color_lat = '#4f120d'
fill_color_lat = '#a6261d'
color_amp = '#472650'
fill_color_amp = '#855F8E'

bar_width = 0.5


def recolor(boxplot_elements, color, fill_color):
	"""
	Add colors to bars (setup each element)
	Args:
		boxplot_elements (dict):
			components of the boxplot
		color (str):
			HEX color of outside lines
		fill_color (str):
			HEX color of filling
	"""
	for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
		plt.setp(boxplot_elements[element], color=color)
	plt.setp(boxplot_elements["fliers"], markeredgecolor=color)
	for patch in boxplot_elements['boxes']:
		patch.set(facecolor=fill_color)


def plot_box(latencies_per_test, amplitudes_per_test):
	"""
	Args:
		latencies_per_test (list of list):
			latencies per test data
		amplitudes_per_test (list of list):
			amplitudes per test data
	"""
	box_distance = 1.2
	slice_indexes = range(len(latencies_per_test[0]))
	# reshape to M dots per N slices instead of N dots in slices per M tests
	latencies = list(zip(*latencies_per_test))
	amplitudes = list(zip(*amplitudes_per_test))

	# fixme
	for slice_index in slice_indexes:
		amplitudes[slice_index] = [d for d in amplitudes[slice_index] if d > 0]
		latencies[slice_index] = [d for d in latencies[slice_index] if d > 0]

	# create subplots
	fig, lat_axes = plt.subplots()

	# property of bar fliers
	fliers = dict(markerfacecolor='k', marker='*', markersize=3)

	# plot latencies
	xticks = [x * box_distance for x in slice_indexes]
	plt.xticks(fontsize=56)
	plt.yticks(fontsize=56)
	lat_plot = lat_axes.boxplot(latencies, positions=xticks, widths=bar_width, patch_artist=True, flierprops=fliers)
	recolor(lat_plot, color_lat, fill_color_lat)
	# add the second y-axis
	amp_axes = lat_axes.twinx()

	# plot amplitudes
	xticks = [x * box_distance + bar_width for x in slice_indexes]
	amp_plot = amp_axes.boxplot(amplitudes, positions=xticks, widths=bar_width, patch_artist=True, flierprops=fliers)
	recolor(amp_plot, color_amp, fill_color_amp)

	# set the legend
	lat_patch = mpatches.Patch(color=fill_color_lat, label='Latency')
	amp_patch = mpatches.Patch(color=fill_color_amp, label='Amplitude')
	# plt.legend(handles=[lat_patch, amp_patch], loc='best')
	# plot setting
	lat_axes.set_xlabel('Slices', fontsize=56)
	lat_axes.set_ylabel("Latencies, ms", fontsize=56)
	amp_axes.set_ylabel("Amplitudes, mV (normalized)", fontsize=56)
	plt.xticks([i * box_distance for i in slice_indexes], [i + 1 for i in slice_indexes])
	plt.yticks(fontsize=56)
	plt.xlim(-0.5, len(slice_indexes) * box_distance)
	plt.show()


def run():
	# get data
	# nest_tests = read_nest_data('../../GPU_extensor_eesF40_inh100_s6cms_T.hdf5')
	neuron_tests_ex = select_slices('../../neuron-data/3steps_speed15_EX.hdf5', 0, 11000)
	# 15cm/s [0, 12000] 21cm/s [0, 6000]
	gras_tests_ex = select_slices('../../GRAS/E_15cms_40Hz_100%_2pedal_no5ht.hdf5', 10000, 22000)
	# 15cm/s [10000, 22000] 21cm/s [5000, 11000]

	for test in range(len(neuron_tests_ex)):
		neuron_tests_ex[test] = normalization(neuron_tests_ex[test], -1, 1)

	for test in range(len(gras_tests_ex)):
		gras_tests_ex[test] = normalization(gras_tests_ex[test], -1, 1)

	# the main loop of simulations data
	# bio_data_ex = bio_data_runs('RMG')
	# bio_data_fl = bio_data_runs('RTA')
	# for test in range(len(bio_data_ex)):
	# 	bio_data_ex[test] = normalization(bio_data_ex[test], -1, 1)
	# for test in range(len(bio_data_fl)):
	# 	bio_data_fl[test] = normalization(bio_data_fl[test], -1, 1)
	for sim_datas, step in [(neuron_tests_ex, sim_step),
	                                (gras_tests_ex, sim_step)]:
		latencies = []
		amplitudes = []
		# collect amplitudes and latencies per test data
		for i, test_data in enumerate(sim_datas):
			print("i = ", i)
			sim_lat, sim_amp = sim_process(test_data, step, after_latencies=False)
			latencies.append(sim_lat)
			amplitudes.append(sim_amp)
		print("amplitudes = ", amplitudes)
		# plot simulation data
		plot_box(latencies, amplitudes)


if __name__ == "__main__":
	run()
