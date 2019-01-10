from matplotlib import pylab as plt
import matplotlib.patches as mpatches
from analysis.functions import read_nest_data, read_neuron_data
from analysis.hystograms_latency_amplitude import process

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
	for patch in boxplot_elements['boxes']:
		patch.set(facecolor=fill_color)


def plot_bars(latencies_per_test, amplitudes_per_test):
	"""
	Args:
		latencies_per_test (list of list):
			latencies per test data
		amplitudes_per_test (list of list):
			amplitudes per test data
	"""
	slice_indexes = range(len(latencies_per_test[0]))
	# reshape to 100 dots per slice instad of 100 tests with 6 slices
	latencies = list(zip(*latencies_per_test))
	amplitudes = list(zip(*amplitudes_per_test))

	# create subplots
	fig, lat_axes = plt.subplots()
	# property of bar fliers
	fliers = dict(markerfacecolor='k', marker='*', markersize=3)

	# plot latencies
	xticks = [x * 1.2 for x in slice_indexes]
	lat_plot = lat_axes.boxplot(latencies, positions=xticks, widths=bar_width, patch_artist=True, flierprops=fliers)
	recolor(lat_plot, color_lat, fill_color_lat)

	# add the second y-axis
	amp_axes = lat_axes.twinx()

	# plot amplitudes
	xticks = [x * 1.2 + bar_width for x in slice_indexes]
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
	plt.xticks([i * 1.2 for i in slice_indexes], [i + 1 for i in slice_indexes])
	plt.xlim(-0.5, len(slice_indexes) + 1)
	plt.show()


def run():
	# get data
	nest_tests = read_nest_data('/home/alex/nest_21cms.hdf5')
	neuron_tests = read_neuron_data('/home/alex/neuron_21cms.hdf5')

	# the main loop of simulations data
	for sim_datas in [nest_tests, neuron_tests]:
		latencies = []
		amplitudes = []
		# collect amplitudes and latencies per test data
		for test_data in sim_datas:
			sim_lat, sim_amp = process(test_data)
			latencies.append(sim_lat)
			amplitudes.append(sim_amp)
		# plot simulation data
		plot_bars(latencies, amplitudes)


if __name__ == "__main__":
	run()
