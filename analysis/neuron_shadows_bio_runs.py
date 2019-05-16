from analysis.neuron_data import neuron_20_runs
from matplotlib import pylab as plt
import numpy as np

neuron_20_runs = neuron_20_runs()
all_neuron_slices = []
for k in range(len(neuron_20_runs)):
	if neuron_20_runs[k]:
		neuron_slices = []
		offset = 0
		for i in range(int(len(neuron_20_runs[k]) / 1000)):
			neuron_slices_tmp = []
			for j in range(offset, offset + 1000):
				neuron_slices_tmp.append(neuron_20_runs[k][j])
			neuron_slices.append(neuron_slices_tmp)
			offset += 1000
		all_neuron_slices.append(neuron_slices)
# for run in neuron_20_runs:
# 	if run:
# 		slices_data = [run[slice_data_index:slice_data_index+1000] for slice_data_index in range(len(run))[::1000]]
# 		all_neuron_slices.append(slices_data)
all_neuron_slices = list(zip(*all_neuron_slices))
neuron_step = 0.025

yticks = []
for index, run in enumerate(all_neuron_slices):
	print(len(run), run)
	offset = index * 5
	plt.plot([r + offset for r in run ])
	mean_data = list(map(lambda elements: np.mean(elements), zip(*run)))
	times = [time * neuron_step for time in range(len(mean_data))]
	means = [voltage + offset for voltage in mean_data]
	yticks.append(means[0])
	minimal_per_step = [min(a) for a in zip(*run)]
	maximal_per_step = [max(a) for a in zip(*run)]
	plt.plot(times, means, linewidth=0.5, color='k')
	plt.fill_between(times, [mini + offset for mini in minimal_per_step],
	                 [maxi + offset for maxi in maximal_per_step], alpha=0.35)
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
plt.yticks(yticks, range(1, len(all_neuron_slices) + 1), fontsize=14)
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()