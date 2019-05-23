import numpy as np
from matplotlib.mlab import PCA
from analysis.patterns_in_bio_data import bio_data_runs
from matplotlib import pylab as plt
from analysis.functions import read_neuron_data
from analysis.histogram_lat_amp import sim_process
from analysis.cut_several_steps_files import select_slices

# bio_data = bio_data_runs()
# bio_np_array = np.array([np.array(x) for x in bio_data])

# bio_np_array = bio_np_array.T
sim_step = 0.025
neuron_list = select_slices('../../neuron-data/3steps_speed15_EX.hdf5', 17000, 29000)
gras_list = select_slices('../../GRAS/E_15cms_40Hz_100%_2pedal_no5ht.hdf5', 10000, 22000)

neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
gras_means = list(map(lambda voltages: np.mean(voltages), zip(*gras_list)))

# calculating latencies and amplitudes of mean values
neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=False)[0]
neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=False)[1]

gras_means_lat = sim_process(gras_means, sim_step, inhibition_zero=False)[0]
gras_means_amp = sim_process(gras_means, sim_step, inhibition_zero=False)[1]

neuron_lat_nparray = np.array([np.array(x) for x in neuron_means_lat])
neuron_amp_nparray = np.array([np.array(x) for x in neuron_means_amp])

gras_lat_nparray = np.array([np.array(x) for x in gras_means_lat])
gras_amp_nparray = np.array([np.array(x) for x in gras_means_amp])

neuron_amp_nparray = neuron_amp_nparray.T
neuron_lat_nparray = neuron_lat_nparray.T

gras_amp_nparray = gras_amp_nparray.T
gras_lat_nparray = gras_lat_nparray.T

neuron_amp_nparray = np.reshape(neuron_amp_nparray, (len(neuron_means_lat), 1))
neuron_lat_nparray = np.reshape(neuron_lat_nparray, (len(neuron_means_lat), 1))

gras_amp_nparray = np.reshape(gras_amp_nparray, (len(gras_means_lat), 1))
gras_lat_nparray = np.reshape(gras_lat_nparray, (len(gras_means_lat), 1))

print("len(neuron_amp_nparray) = ", len(neuron_amp_nparray))
print("len(neuron_lat_nparray) = ", len(neuron_lat_nparray))
# neuron_np_array = np.array([np.array(x) for x in cutted_neuron])
# print(len(cutted_neuron), len(cutted_neuron[0]))
# neuron_np_array = neuron_np_array.T
# print("len(neuron_np_array = ", len(neuron_np_array))
# bio_np_array = np.reshape(bio_np_array, (1200, 1))
# neuron_np_array = np.reshape(neuron_np_array, (1200, 1))
data = np.hstack((neuron_amp_nparray, neuron_lat_nparray))
data_gras = np.hstack((gras_amp_nparray, gras_lat_nparray))
print("data = ", data)

mu = data.mean(axis=0)
mu_gras = data_gras.mean(axis=0)

data = data - mu
data_gras = data_gras - mu_gras

eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
eigenvectors_gras, eigenvalues_gras, V_gras = np.linalg.svd(data_gras.T, full_matrices=False)

projected_data = np.dot(data, eigenvectors)
projected_data_gras = np.dot(data_gras, eigenvectors_gras)
sigma = projected_data.std(axis=0).mean()
sigma_gras = projected_data_gras.std(axis=0).mean()
print("eigenvectors = ", eigenvectors)

fig, ax = plt.subplots()
ax.scatter(neuron_amp_nparray, neuron_lat_nparray)   # , c=colors
for axis in eigenvectors:
	start, end = mu, mu + sigma * axis
	print("start = ", len(start), start)
	print("end = ", len(end), end)
	ax.annotate(
		'', xy=end, xycoords='data',
		xytext= start, textcoords='data',
		arrowprops=dict(facecolor='red', width=2.0)
		)
plt.show()
fig, ax = plt.subplots()
ax.scatter(gras_amp_nparray, gras_lat_nparray)   # , c=colors
for axis in eigenvectors_gras:
	start, end = mu_gras, mu_gras + sigma_gras * axis
	ax.annotate(
		'', xy=end, xycoords='data',
		xytext= start, textcoords='data',
		arrowprops=dict(facecolor='red', width=2.0)
		)
plt.show()