import numpy as np
import pylab as plt
from analysis.functions import *
from analysis.namespaces import *

delta_y_step = 0.3

k_mean = 0
k_x_data = 1
k_y_data = 2


def plot_linear(bio_pack, sim_pack, slices_number):
	"""
	Args:
		bio_pack (tuple):
			voltages, x-data and y-data
		sim_pack (tuple):
			mean of voltages, x-data and y-data
		slices_number (int):
			number of slices
	"""
	sim_mean_volt = sim_pack[k_mean]
	sim_x = sim_pack[k_x_data]
	sim_y = sim_pack[k_y_data]

	bio_volt = bio_pack[k_mean]
	bio_x = bio_pack[k_x_data]
	bio_y = bio_pack[k_y_data]

	slice_indexes = range(slices_number)

	# plot mean data per slice
	for slice_index in slice_indexes:
		start = int(slice_index * 25 / bio_step)
		end = int((slice_index + 1) * 25 / bio_step)
		sliced_data = bio_volt[start:end]
		offset = slice_index * delta_y_step

		plt.plot([t * bio_step for t in range(len(sliced_data))],
		         [offset + d for d in sliced_data],
		         color='#FF7254')

		start = int(slice_index * 25 / sim_step)
		end = int((slice_index + 1) * 25 / sim_step)
		sliced_data = sim_mean_volt[start:end]

		plt.plot([t * sim_step for t in range(len(sliced_data))],
		         [offset + d for d in sliced_data],
		         color='#54CBFF')

	# BIO data processing
	x = np.array(bio_x)
	y = np.array(bio_y)

	xfit, yfit = calc_linear(x, y)

	plt.scatter(x, y, color='#FF4B25', alpha=0.3)
	plt.plot(xfit, yfit, color='#CA2D0C', linestyle='--', linewidth=3, label='BIO')

	# SIM data processing
	x = np.array(sim_x)
	y = np.array(sim_y)

	xfit, yfit = calc_linear(x, y)

	plt.scatter(x, y, color='#25C7FF', alpha=0.3)
	plt.plot(xfit, yfit, color='#0C88CA', linestyle='--', linewidth=3, label='NEST')

	# plot properties
	plt.xlabel("Time, ms")
	plt.ylabel("Slices")
	plt.yticks([])
	plt.xlim(0, 25)
	plt.ylim(-1, slices_number * delta_y_step)
	plt.legend()
	plt.show()


def form_bio_pack(volt_and_stim, slice_numbers):
	bio_lat, bio_amp = bio_process(volt_and_stim, slice_numbers)

	voltages = volt_and_stim[0]
	bio_voltages = normalization(voltages, zero_relative=True)

	bio_x = bio_lat
	bio_y = [index * delta_y_step + bio_voltages[int((bio_x[index] + 25 * index) / bio_step)]
	         for index in range(slice_numbers)]

	# form biological data pack
	bio_pack = (bio_voltages, bio_x, bio_y)

	return bio_pack


def form_sim_pack(tests_data):
	sim_x = []
	sim_y = []

	# collect amplitudes and latencies per test data
	for test_data in tests_data:
		lat, amp = sim_process(test_data)
		sim_x += lat

		test_data = normalization(test_data, zero_relative=True)

		for slice_index in range(len(lat)):
			a = slice_index * delta_y_step + test_data[int(lat[slice_index] / sim_step + slice_index * 25 / sim_step)]
			sim_y.append(a)

	# form simulation data pack
	sim_mean_volt = normalization(list(map(lambda voltages: np.mean(voltages), zip(*tests_data))), zero_relative=True)
	sim_pack = (sim_mean_volt, sim_x, sim_y)

	return sim_pack


def run():
	bio_volt_and_stim = read_bio_data('/home/alex/bio_21cms.txt')
	nest_tests = read_nest_data('/home/alex/nest_21cms.hdf5')
	neuron_tests = read_neuron_data('/home/alex/neuron_21cms.hdf5')

	slice_numbers = int(len(neuron_tests[0]) * sim_step // 25)

	bio_pack = form_bio_pack(bio_volt_and_stim, slice_numbers)

	# loop of NEST and Neuron data
	for sim_tests in [nest_tests, neuron_tests]:
		sim_pack = form_sim_pack(sim_tests)

		plot_linear(bio_pack, sim_pack, slice_numbers)


if __name__ == "__main__":
	run()
