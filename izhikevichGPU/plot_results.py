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
neuron_id = 0
file_path = str(sys.argv[1])
neurons_to_show = int(sys.argv[2])

with open(file_path, "r") as file:
	for line in file:
		if "Voltage" in line:
			voltage_data[neuron_id] = [float(i) for i in line.replace("Voltage: [", "").replace(", ]", "").split(",")]
		if "Spikes" in line:
			spike_data[neuron_id] = [float(i) for i in line.replace("Spikes: [", "").replace(", ]", "").split(",")]
		if "I_potential" in line:
			currents_data[neuron_id] = [float(i) for i in line.replace("I_potential: [", "").replace(", ]", "").split(",")]
			neuron_id += 1


colors = ["#e6194B", "#3cb44b", "#4363d8", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
          "#008080", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"]

if neurons_to_show > 10:
	neurons_to_show = 10
if neurons_to_show > neuron_id:
	neurons_to_show = neuron_id

plt.figure()
plt.suptitle("Izhikevich GPU model results")
for nrn_id in range(neurons_to_show):
	times = [i / 10 for i in range(len(voltage_data[nrn_id]))]
	voltages = voltage_data[nrn_id]
	spike_times = spike_data[nrn_id]
	currents = currents_data[nrn_id]

	# plot voltages
	plt.subplot(211)
	plt.plot(times, voltages, label=str(neuron_id), color=colors[nrn_id])
	plt.xlim(times[0], times[-1])
	plt.xticks(times[::int(len(times)/10)], [])
	plt.ylabel("Voltages, mV")
	# plot spikes
	plt.plot(spike_times, [0 for _ in spike_times], '.', color='k', markersize=5)

	# plot currents
	plt.subplot(212)
	plt.plot(times, currents, label=str(neuron_id), color=colors[nrn_id])
	plt.xlim(times[0], times[-1])
	plt.xticks(times[::int(len(times) / 10)])
	plt.ylabel("Currents, nA")
	plt.xlabel("Time, ms")

plt.legend(range(neuron_id), loc='upper left')
plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.88, wspace=0.0, hspace=0.04)
plt.show()
