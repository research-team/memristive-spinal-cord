"""
Run the program as
python3 plot_results.py /path/to/sim_results.txt
"""

import os
import sys
import numpy as np
import pylab as plt
from collections import defaultdict

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
nodes = defaultdict(list)


def read_data(file_path):
	global sim_time
	global nodes
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
					spike_data[gid] = [float(i) for i in raw_data.split(",")[:-1]]

			if has_voltage:
				raw_data = data_block[k_voltage].replace("Voltage: [", "").replace("]", "")
				if len(raw_data) > 0:
					data = [float(i) for i in raw_data.split(",")[:-1]]
					nodes[group_name].append(data)
			if has_currents:
				raw_data = data_block[k_currents].replace("I_potential: [", "").replace("]", "")
				if len(raw_data) > 0:
					currents_data[gid] = [float(i) for i in raw_data.split(",")[:-1]]


def plot():
	global nodes
	global sim_time
	for node_name, neuron_voltages in nodes.items():
		plt.figure()
		plt.suptitle("Voltage {}".format(node_name))
		mean_voltage = list(map(lambda x: np.mean(x), zip(*neuron_voltages)))
		plt.plot([time / 10 for time in range(len(mean_voltage))], mean_voltage)
		plt.xlim(0, sim_time)
		plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.88, wspace=0.0, hspace=0.04)
		plt.savefig("{}/results/{}.png".format(os.getcwd(), node_name), dpi=200)
		plt.close('all')


if __name__ == "__main__":
	read_data(str(sys.argv[1]))
	plot()

#plt.subplot(414)
#plt.ylabel("Synapse weights, nA")
#plt.xlabel("Time, ms")
##weights = weights_data[1]
#times = [i / 10 for i in range(len(currents_data[1]))]
##plt.plot(times, weights, label=str(1), color=colors[1])
#plt.xticks(range(0, sim_time + 1, 25))
#plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
"""
# plot voltages
	plt.subplot(311)
	plt.ylabel("Voltages, mV")
	has_voltage = nrn_id in voltage_data.keys()
	has_spikes = nrn_id in spike_data.keys()
	has_currents = nrn_id in currents_data.keys()
	has_weights = nrn_id in weights_data.keys()

	if has_voltage and len(voltage_data[nrn_id]) > 0:
		voltages = voltage_data[nrn_id]
		times = [i / 10 for i in range(len(voltage_data[nrn_id]))]
		plt.xticks(range(0, sim_time + 1, 25), [])
		plt.xlim(0, times[-1])
		plt.plot(times, voltages, label=str(nrn_id), color=colors[nrn_id % len(colors)])
		plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)

	# plot spikes
	plt.subplot(312)
	plt.ylabel("Neuron ID")
	if has_spikes:
		spike_times = spike_data[nrn_id]
		plt.xticks(range(0, sim_time+1, 25), [])
		plt.xlim(0, sim_time)
		plt.plot(spike_times, [nrn_id for _ in spike_times], '.', color='k', markersize=5)
		plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)

	# plot currents
	plt.subplot(313)
	plt.ylabel("Currents, nA")
	if has_currents:
		currents = currents_data[nrn_id]
		times = [i / 10 for i in range(len(currents_data[nrn_id]))]
		plt.plot(times, currents, label=str(nrn_id), color=colors[nrn_id % len(colors)])
		plt.xlim(0, times[-1])
		plt.xticks(range(0, sim_time + 1, 25))
		plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
"""


