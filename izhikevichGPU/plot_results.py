"""
Run the program as
python3 plot_results.py /path/to/sim_results.txt
"""

import sys
import pylab as plt

data = {}
neuron_id = 0
file_path = sys.argv[1]

with open(file_path, "r") as file:
	for line in file:
		if "Voltage" in line:
			data[neuron_id] = [float(i) for i in line.replace("Voltage: [", "").replace(", ]", "").split(",")]
			neuron_id += 1

plt.figure()
for neuron_id, v_m in data.items():
	plt.plot([i + 1 for i in range(len(v_m))], v_m, label=neuron_id)
plt.legend(loc='upper left')
plt.show()
