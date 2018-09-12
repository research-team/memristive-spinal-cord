"""
Run the program as
python3 plot_results.py /path/to/sim_results.txt 5

recommended show not more than 5-7 neurons
"""

import sys
import pylab as plt

spike_data = {}
voltage_data = {}
currents_data = {}
neuron_ids = []
file_path = str(sys.argv[1])
neurons_to_show = int(sys.argv[2])
sim_time = 0

with open(file_path, "r") as file:
	for data_block in file.read().split("-"*15 + "\n")[:-1]:
		data_block = data_block.split("\n")[:-1]
		# 0 ID
		# 1 Obj
		# 2 Iter
		# 3 Spikes
		# 4 Voltages
		# 5 Currents
		has_spikes = len(data_block) > 3 and "Spikes" in data_block[3]
		has_voltage = len(data_block) > 4 and "Voltage" in data_block[4]
		has_currents = len(data_block) > 5 and "I_potential" in data_block[5]
		print(data_block[:3], has_spikes, has_voltage, has_currents)

		gid = int(data_block[0].replace("ID: ", ""))
		sim_time = float(data_block[2].replace("Iter: ", "")) / 10

		if has_spikes:
			raw_data = data_block[3].replace("Spikes: [", "").replace("]", "")
			if len(raw_data) > 0:
				spike_data[gid] = [float(i) for i in raw_data.split(",")[:-1]]
		if has_voltage:
			raw_data = data_block[4].replace("Voltage: [", "").replace("]", "")
			if len(raw_data) > 0:
				voltage_data[gid] = [float(i) for i in raw_data.split(",")[:-1]]
		if has_currents:
			raw_data = data_block[5].replace("I_potential: [", "").replace("]", "")
			if len(raw_data) > 0:
				currents_data[gid] = [float(i) for i in raw_data.split(",")[:-1]]
		neuron_ids.append(gid)


colors = ["#e6194B", "#3cb44b", "#4363d8", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
          "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"]

if neurons_to_show > 10:
	neurons_to_show = 10
if neurons_to_show > len(neuron_ids):
	neurons_to_show = len(neuron_ids)

plt.figure()
plt.suptitle("Izhikevich GPU model results")
for nrn_id in neuron_ids:
	# plot voltages
	plt.subplot(311)
	plt.ylabel("Voltages, mV")
	has_voltage = nrn_id in voltage_data.keys()
	has_spikes = nrn_id in spike_data.keys()
	has_currents = nrn_id in currents_data.keys()

	if has_voltage:
		voltages = voltage_data[nrn_id]
		times = [i / 10 for i in range(len(voltage_data[nrn_id]))]
		plt.xticks(times[::int(sim_time/10)], [])
		plt.xlim(0, times[-1])
		plt.plot(times, voltages, label=str(nrn_id), color=colors[nrn_id])
		plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)

	# plot spikes
	plt.subplot(312)
	plt.ylabel("Neuron ID")
	if has_spikes:
		spike_times = spike_data[nrn_id]
		plt.xticks(spike_times[::int(sim_time / 10)], [])
		plt.xlim(0, sim_time)
		plt.plot(spike_times, [nrn_id for _ in spike_times], '.', color='k', markersize=5)
		plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)

	# plot currents
	plt.subplot(313)
	plt.ylabel("Currents, nA")
	plt.xlabel("Time, ms")
	if has_currents:
		currents = currents_data[nrn_id]
		times = [i / 10 for i in range(len(currents_data[nrn_id]))]
		plt.plot(times, currents, label=str(nrn_id), color=colors[nrn_id])
		plt.xlim(0, times[-1])
		plt.xticks(times[::int(len(times) / 10)])
		plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)

plt.legend(neuron_ids, loc='upper left')
plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.88, wspace=0.0, hspace=0.04)
plt.show()
