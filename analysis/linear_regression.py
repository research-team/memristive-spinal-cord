import numpy as np
import pylab as plt
from analysis.functions import *
from analysis.namespaces import *
# from analysis.bio_data_6runs import bio_several_runs
import matplotlib.patches as mpatches
from analysis.patterns_in_bio_data import bio_data_runs
from analysis.cut_several_steps_files import select_slices

delta_y_step = 0.25

k_mean = 0
k_x_data = 1
k_y_data = 2
color_bio = '#a6261d'
color_sim = '#472650'


def plot_linear(bio_pack, sim_pack, slices_number, bio_step, sim_step):
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

	print("sim_x = ", len(sim_x))
	bio_volt = bio_pack[k_mean]
	bio_x = bio_pack[k_x_data]  # [1]
	bio_y = bio_pack[k_y_data]  # [2]
	runs_number = bio_pack[3]

	print("bio_x = ", len(bio_x))
	print("bio_y = ", len(bio_y))

	slice_indexes = range(slices_number)

	yticks = []
	# plot mean data per slice
	for slice_index in slice_indexes:
		# bio data
		start = int(slice_index * 25 / bio_step)
		end = int((slice_index + 1) * 25 / bio_step)
		sliced_data = bio_volt[start:end]
		offset = slice_index * delta_y_step

		plt.plot([t * bio_step for t in range(len(sliced_data))],
		         [offset + d for d in sliced_data],
		         color=color_bio, linewidth=2)
		yticks.append(sliced_data[0] + offset)

		# sim data
		start = int(slice_index * 25 / sim_step)
		end = int((slice_index + 1) * 25 / sim_step)
		sliced_data = sim_mean_volt[start:end]
		# print("len(sim sliced_data) = ", len(sliced_data))
		# plt.plot([t * sim_step for t in range(len(sliced_data))],
		#          [offset + d for d in sliced_data],
		#          color=color_sim, linewidth=2)

	# BIO data processing
	x = np.array(bio_x)
	y = np.array(bio_y)

	print("len(x) = ", len(x))
	xfit, yfit = calc_linear(x, y)

	num_dots = int(len(x) / runs_number)
	print("num_dots = ", num_dots)
	x_per_test = []
	y_per_test = []
	start = 0
	for i in range(runs_number):
		x_per_test.append(x[start:start + num_dots])
		y_per_test.append(y[start:start + num_dots])
		start += num_dots

	tmp_x = list(zip(*x_per_test))
	tmp_y = list(zip(*y_per_test))

	print("len(tmp_x) = ", len(tmp_x))
	for i in range(slices_number):
		plt.scatter(tmp_x[i], tmp_y[i], label=i, color=color_bio, alpha=0.6)  #color_bio
	plt.plot(xfit, yfit, color=color_bio, linestyle='--', linewidth=6, label='BIO')
	# plt.legend()
	# SIM data processing
	x = np.array(sim_x)
	y = np.array(sim_y)

	xfit, yfit = calc_linear(x, y)

	# plt.scatter(x, y, color=color_sim, alpha=0.6)
	# plt.plot(xfit, yfit, color=color_sim, linestyle='--', linewidth=6, label='NEST')
	# plot properties
	plt.xlabel("Time, ms", fontsize=28)
	plt.ylabel("Slices", fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks([], fontsize=28)
	plt.xlim(0, 25)
	plt.ylim(-1.5, slices_number * delta_y_step)
	plt.show()


def form_bio_pack(volt_and_stim, slice_numbers, reversed_data=False):
	bio_lat, bio_amp = bio_process(volt_and_stim, slice_numbers, debugging=False, reversed_data=reversed_data)

	voltages = volt_and_stim[0]
	bio_voltages = normalization(voltages, zero_relative=True)

	bio_x = bio_lat
	bio_y = [index * delta_y_step + bio_voltages[int((bio_x[index] + 25 * index) / bio_step)]
	         for index in range(slice_numbers)]

	# form biological data pack
	bio_pack = (bio_voltages, bio_x, bio_y)

	return bio_pack


def form_sim_pack(tests_data, step, reversed_data=False, debugging=False, inhibition_zero=True, bio=False):
	sim_x = []
	sim_y = []

	# collect amplitudes and latencies per test data
	count = 0

	for e, test_data in enumerate(tests_data):
		lat, amp = sim_process(test_data, step, debugging=debugging, inhibition_zero=inhibition_zero)
		# for el in range(len(lat)):
		# 	if bio:
		# 		if lat[el] < 20:
		# 			lat[el] = lat[el - 1]

		# fixme
		# lat = [d if d > 0 else d[-1] for d in lat ]
		sim_x += lat
		test_data = normalization(test_data, zero_relative=True)
		count += 1

		for slice_index in range(len(lat)):
			a = slice_index * delta_y_step + test_data[int(lat[slice_index] / step + slice_index * 25 / step)]
			sim_y.append(a)
	print("sim_x = ", len(sim_x))
	print("sim_y = ", len(sim_y))

	# form simulation data pack
	sim_mean_volt = normalization(list(map(lambda voltages: np.mean(voltages), zip(*tests_data))), zero_relative=True)
	sim_pack = (sim_mean_volt, sim_x, sim_y, len(tests_data))

	return sim_pack


def run():
	nest_tests = select_slices('../../GRAS/E_21cms_40Hz_100%_2pedal_no5ht.hdf5', 5000, 11000)
	# 21flexor [1000, 5000] 15flexor [1000, 10000]
	neuron_tests = select_slices('../../neuron-data/3steps_newmodel_FL.hdf5', 7000, 11000)
	# 21flexor [7000, 11000] 15flexor [13000, 18000]

	# FixMe add new functionality for several bio datas
	bio_volt = bio_data_runs()
	print("---")
	bio_pack = form_sim_pack(bio_volt, step=bio_step, debugging=False, bio=True)
	print("printed bio pack")

	# loop of NEST and Neuron data
	for sim_tests in [nest_tests, neuron_tests]:
		sim_pack = form_sim_pack(sim_tests, sim_step)
		print("printed sim pack")
		plot_linear(bio_pack, sim_pack, 6, bio_step, sim_step) # min(len(result[1]), len(result[3]))

	quipazine_patch = mpatches.Patch(color='#FF7254', label='quipazine')
	no_quipazine_patch = mpatches.Patch(color='#54CBFF', label='no quipazine')
	plt.legend()


if __name__ == "__main__":
	run()
