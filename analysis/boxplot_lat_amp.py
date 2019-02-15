from matplotlib import pylab as plt
import matplotlib.patches as mpatches
from analysis.functions import read_nest_data, read_neuron_data
from analysis.histogram_lat_amp import sim_process
from analysis.namespaces import *

color_lat = '#BD821B'
fill_color_lat = '#F3CB84'
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
	plt.legend(handles=[lat_patch, amp_patch], loc='best')
	# plot setting
	lat_axes.set_xlabel('Slices')
	lat_axes.set_ylabel("Latencies, ms")
	amp_axes.set_ylabel("Amplitudes, mV (normalized)")
	plt.xticks([i * box_distance for i in slice_indexes], [i + 1 for i in slice_indexes])
	plt.xlim(-0.5, len(slice_indexes) * box_distance)
	plt.show()


def run():
	# get data
	nest_tests = read_nest_data('../../GPU_extensor_eesF40_inh100_s6cms_T.hdf5')
	neuron_tests = read_neuron_data('../../neuron-data/sim_healthy_neuron_extensor_eesF40_i100_s21cms_T_100runs.hdf5')
	# the main loop of simulations data
	for sim_datas, step in [(nest_tests, gpu_step), (neuron_tests, sim_step)]:
		latencies = []
		amplitudes = []
		# collect amplitudes and latencies per test data
		for i, test_data in enumerate(sim_datas):
			sim_lat, sim_amp = sim_process(test_data, step)
			latencies.append(sim_lat)
			amplitudes.append(sim_amp)
		# plot simulation data
		plot_box(latencies, amplitudes)


if __name__ == "__main__":
	run()
