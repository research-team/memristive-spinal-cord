import numpy as np
import pylab as plt
from analysis.functions import *
from analysis.namespaces import *
from analysis.patterns_in_bio_data import bio_data_runs
from analysis.linear_regression import form_sim_pack, form_bio_pack
from analysis.cut_several_steps_files import select_slices


def calc_delta(bio_pack, sim_pack):
	diff_lat = [abs(bio - sim) for bio, sim in zip(bio_pack[0], sim_pack[0])]
	diff_amp = [abs(bio - sim) for bio, sim in zip(bio_pack[1], sim_pack[1])]
	print("bio_pack = ", bio_pack)
	print("sim_pack = ", sim_pack)
	print("diff_lat = ", diff_lat)
	return diff_lat, diff_amp


def draw_lat_amp(data_pack):
	"""
	Function for drawing latencies and amplitudes in one plot
	Args:
		data_pack (tuple):
			data pack of latenccies and amplitudes
	"""
	bar_width = 0.35
	latencies = data_pack[0]
	amplitudes = data_pack[1]

	# create axes
	fig, lat_axes = plt.subplots(1, 1, figsize=(15, 12))
	xticks = [i + 1 for i in range(len(amplitudes))]

	print("xticks = ", len(xticks), xticks)
	print("latencies = ", len(latencies), latencies)
	lat_plot = lat_axes.bar(xticks, latencies, width=bar_width, color=color_lat, alpha=0.7, zorder=2)
	lat_axes.set_xlabel('Slice', fontsize=56)
	lat_axes.set_ylabel("Latency, ms", fontsize=56)

	plt.xticks(fontsize=56)
	plt.yticks(fontsize=56)

	amp_axes = lat_axes.twinx()
	xticks = [x + bar_width for x in xticks]
	amp_plot = amp_axes.bar(xticks, amplitudes, width=bar_width, color=color_amp, alpha=0.7, zorder=2)
	amp_axes.set_ylabel("Amplitude, mV", fontsize=56)

	# plot text annotation for data
	for index in range(len(amplitudes)):
		amp = round(amplitudes[index], 2)
		lat = round(latencies[index], 2)
		# lat_axes.text(index - bar_width / 2, lat + max(latencies) / 50, str(lat))
		# amp_axes.text(index + bar_width / 2, amp + max(amplitudes) / 50, str(amp))
	plt.xticks(fontsize=56)
	plt.yticks(fontsize=56)
	# plt.legend((lat_plot, amp_plot), ("Latency", "Amplitude"), loc='best')
	plt.show()


def run():
	plot_delta = True

	bio = read_bio_data('../bio-data/3_1.31 volts-Rat-16_5-09-2017_RMG_9m-min_one_step.txt')
	nest_tests = read_nest_data('../../nest-data/sim_extensor_eesF40_i100_s15cms_T.hdf5')
	neuron_tests = select_slices('../../neuron-data/mn_E25tests (8).hdf5', 0, 6000)
	gpu_data = select_slices('../../GRAS/E_15cms_40Hz_100%_2pedal_no5ht.hdf5', 10000, 22000)
	slice_numbers = int(len(neuron_tests[0]) / 25 * sim_step)

	# collect amplitudes and latencies per test data
	if plot_delta:
		bio_volt = bio_data_runs()
		bio_data = list(map(lambda voltages: np.mean(voltages), zip(*bio_volt)))
		bio_volt = normalization(bio_data, -1, 1)
		bio_pack = sim_process(bio_volt, step=bio_step, debugging=False, inhibition_zero=True, after_latencies=False)
		print("bio_pack = ", bio_pack)
		nest_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*nest_tests))), sim_step,
		                        inhibition_zero=True, after_latencies=False)
		neuron_data = list(map(lambda voltages: np.mean(voltages), zip(*neuron_tests)))
		print("neuron_data = ", len(neuron_data))
		neuron_data = normalization(neuron_data, -1, 1)
		print("neuron_data = ", neuron_data)

		neuron_pack = sim_process(neuron_data, sim_step, inhibition_zero=True, after_latencies=False)
		print("neuron_pack = ", neuron_pack)
		gpu_data = list(map(lambda voltages: np.mean(voltages), zip(*gpu_data)))
		gpu_data = normalization(gpu_data, -1, 1)
		gpu_pack = sim_process(gpu_data, sim_step,
		                       inhibition_zero=True, after_latencies=False)

		res_pack = calc_delta(bio_pack, neuron_pack)
		print("bio - neuron")
		draw_lat_amp(res_pack)

		# res_pack = calc_delta(bio_pack, neuron_pack)
		# draw_lat_amp(res_pack)

		res_pack = calc_delta(bio_pack, gpu_pack)
		print("bio - gpu")
		draw_lat_amp(res_pack)
	else:
		# bio_pack = bio_process(bio, slice_numbers, reversed_data=True)
		# bio_pack = bio_process(bio, slice_numbers)
		# nest_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*nest_tests))), sim_step)
		bio_volt = bio_data_runs()
		bio_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*bio_volt))), step=bio_step,
		                       inhibition_zero=True)

		neuron_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*neuron_tests))), sim_step,
		                          inhibition_zero=True)

		gpu_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*gpu_data))), sim_step,
		                       inhibition_zero=True)

		draw_lat_amp(bio_pack)
		# draw_lat_amp(nest_pack)
		draw_lat_amp(neuron_pack)
		draw_lat_amp(gpu_pack)


if __name__ == "__main__":
	run()
