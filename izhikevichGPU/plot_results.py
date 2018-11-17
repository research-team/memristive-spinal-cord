"""
Run the program as
python3 plot_results.py /path/to/sim_results.txt
"""

import os
import sys
import logging
import numpy as np
import pylab as plt
import matplotlib.patches as mpatches
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)

moto_color = '#ff8989'
pool_color = '#9287ff'

neuron_ids = []
spike_data = {}
weights_data = {}
voltage_data = {}
currents_data = {}

k_id = 0
k_obj = 1
k_name = 2
k_iter = 3
k_spikes = 4
k_voltage = 5
k_currents = 6

sim_time = 0
voltage_nodes = defaultdict(list)
spikes_nodes = defaultdict(list)
nodes_curr = defaultdict(list)


def read_data(file_path):
	global sim_time
	global spikes_nodes
	global voltage_nodes
	global nodes_curr
	with open(file_path, "r") as file:
		for data_block in file.read().split("-" * 15 + "\n")[:-1]:
			data_block = data_block.split("\n")[:-1]
			gid = int(data_block[k_id].replace("ID: ", ""))
			group_name = data_block[k_name].replace("Name: ", "")
			sim_time = int(float(data_block[k_iter].replace("Iter: ", "")) / 10)
			has_spikes = len(data_block) > 4 and "Spikes" in data_block[k_spikes]
			has_voltage = len(data_block) > 5 and "Voltage" in data_block[k_voltage]
			has_currents = len(data_block) > 6 and "I_potential" in data_block[k_currents]

			if has_spikes:
				raw_data = data_block[k_spikes].replace("Spikes: [", "").replace("]", "")
				if len(raw_data) > 0:
					data = [float(i) for i in raw_data.split(",")[:-1]]
					spikes_nodes[group_name] += data

			if has_voltage:
				raw_data = data_block[k_voltage].replace("Voltage: [", "").replace("]", "")
				if len(raw_data) > 0:
					data = [float(i) for i in raw_data.split(",")[:-1]]
					voltage_nodes[group_name].append(data)
			if has_currents:
				raw_data = data_block[k_currents].replace("I_potential: [", "").replace("]", "")
				if len(raw_data) > 0:
					data = [float(i) for i in raw_data.split(",")[:-1]]
					nodes_curr[group_name].append(data)
			# currents_data[gid] = [float(i) for i in raw_data.split(",")[:-1]]


def get_slices(mean_voltage):
	slices = []
	yticks = []
	for index_slice, iter_slice in enumerate(range(len(mean_voltage))[::250]):
		yticks.append(-(mean_voltage[iter_slice] + index_slice * 30))
		slices.append(mean_voltage[iter_slice:iter_slice + 250])
	return slices, yticks


def plot():
	global voltage_nodes
	global nodes_curr
	global sim_time

	len_nodes = len(voltage_nodes)

	for key_index, key_name in enumerate(voltage_nodes.keys()):
		logging.info("Plotting {} ({}/{})".format(key_name, key_index + 1, len_nodes))
		mean_voltage = list(map(lambda x: np.mean(x), zip(*voltage_nodes[key_name])))

		if key_name == "MP_E":
			logging.info("Plotting slices {}".format(key_name))
			# create plot
			plt.figure(figsize=(10, 5))
			plt.suptitle("Voltage Slices {}".format(key_name))

			# split data
			moto_slices, yticks = get_slices(mean_voltage)
			# plot data
			for pool_index, slice_data in enumerate(moto_slices):
				times = [time / 10 for time in range(len(slice_data))]
				voltages = [-(y + pool_index * 30) for y in slice_data]
				plt.plot(times, voltages, color='k', linewidth=0.5)

			# plot lines (which of them)
			for slice_index, time_line in enumerate([13, 13, 15, 17, 17, 21]):
				plt.plot([time_line, time_line],
				         [yticks[slice_index] - 15, yticks[slice_index] + 15],
				         color='r', linewidth=1.5)
			plt.axvline(x=5, linewidth=0.8, color='r')
			plt.xticks(range(0, 26), range(0, 26))
			plt.yticks(yticks, range(1, len(moto_slices)+1))
			plt.xlim(0, 25)
			plt.grid()

			plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.88, wspace=0.0, hspace=0.04)
			plt.savefig("{}/results/Slices-{}.png".format(os.getcwd(), key_name), dpi=200)
			plt.close('all')

		# create plot
		plt.figure(figsize=(10, 5))
		plt.suptitle("Mean voltage '{}'".format(key_name))

		plt.subplot(2, 1, 1)
		mean_voltage = list(map(lambda x: np.mean(x), zip(*voltage_nodes[key_name])))
		plt.grid()
		plt.plot([time / 10 for time in range(len(mean_voltage))], mean_voltage)
		plt.plot(spikes_nodes[key_name], [0] * len(spikes_nodes[key_name]), '.', color='r', markersize=2)
		plt.xlim(0, sim_time)
		# plot the slice border
		for i in range(0, int(sim_time), 25):
			plt.axvline(x=i, linewidth=1.5, color='k')
		plt.xticks(range(0, int(sim_time) + 1, 5),
					 [""] * ((int(sim_time) + 1) // 5))
		plt.ylim(-100, 40)
		plt.ylabel("Voltage [mV]")

		plt.subplot(2, 1, 2)
		mean_currents = list(map(lambda x: np.mean(x), zip(*nodes_curr[key_name])))
		plt.plot([time / 10 for time in range(len(mean_currents))], mean_currents, color='green')
		# plot the slice border
		for i in range(0, int(sim_time), 25):
			plt.axvline(x=i, linewidth=1.5, color='k')
		plt.xlim(0, sim_time)
		plt.xticks(range(0, int(sim_time) + 1, 5),
					 ["{}\n{}".format(global_time - 25 * (global_time // 25), global_time)
					  for global_time in range(0, int(sim_time) + 1, 5)],
					 fontsize=8)
		plt.yticks(fontsize=8)
		plt.ylabel("Current [pA]")
		plt.xlabel("Simulation time [ms]")

		#plt.subplots_adjust(left=0.15, bottom=0.10, right=0.97, top=0.88, wspace=0.0, hspace=0.1)
		plt.grid()
		plt.savefig("{}/results/{}.png".format(os.getcwd(), key_name), dpi=200)
		plt.close('all')


if __name__ == "__main__":
	if len(sys.argv) >= 2:
		read_data(str(sys.argv[1]))
	else:
		read_data("/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU/cmake-build-debug/sim_results.txt")
	plot()
