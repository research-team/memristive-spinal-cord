from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from analysis.cut_several_steps_files import select_slices
from analysis.functions import sim_process
neuron_list = select_slices('../../neuron-data/mn_E15_speed25tests.hdf5', 0, 12000)

neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=True)[0]
neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=True, after_latencies=True)[1]
neuron_peaks_number = []

count = 0
for index, run in enumerate(neuron_list):
	print("index = ", index)

	try:
		neuron_peaks_number_run = sim_process(run, sim_step, inhibition_zero=True, after_latencies=True)[2]
		count += 1
	except IndexError:
		continue

	neuron_peaks_number.append(neuron_peaks_number_run)

fig = pyplot.figure()
ax = Axes3D(fig)