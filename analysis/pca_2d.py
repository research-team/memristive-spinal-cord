import numpy as np
from analysis.patterns_in_bio_data import bio_data_runs
from matplotlib import pylab as plt
from analysis.histogram_lat_amp import sim_process
from analysis.cut_several_steps_files import select_slices
from analysis.functions import normalization, grahamscan

sim_step = 0.025
bio_step = 0.25
offset = 0
all_bio_slices = []

bio_data = bio_data_runs()

for k in range(len(bio_data)):
	bio_slices = []
	offset = 0
	for i in range(int(len(bio_data[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio_data[k][j])
		bio_slices.append(bio_slices_tmp)
		offset += 100
	all_bio_slices.append(bio_slices)   # list [4][16][100]
print("all_bio_slices = ", all_bio_slices)
all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]

instant_mean = []
for slice in range(len(all_bio_slices)):
	instant_mean_sum = []
	for dot in range(len(all_bio_slices[slice][0])):
		instant_mean_tmp = []
		for run in range(len(all_bio_slices[slice])):
			instant_mean_tmp.append(abs(all_bio_slices[slice][run][dot]))
		instant_mean_sum.append(sum(instant_mean_tmp))
	instant_mean.append(instant_mean_sum)
for sl in range(len(instant_mean)):
	instant_mean[sl] = normalization(instant_mean[sl], -1, 1)

print("instant_mean = ", instant_mean)
index = 1
yticks = []
yticks_placeholders = []
for slice_mean in instant_mean:
	offset = index
	yticks_placeholders.append(index)
	print("slice_mean = ", slice_mean)
	# plt.plot([sl + offset for sl in slice_mean])
	yticks.append([sl + offset for sl in slice_mean])
	index += 1
# plt.show()

# creating the lists of voltages
volts = []
for i in instant_mean:
	for j in i:
		volts.append(j)

neuron_list = select_slices('../../neuron-data/mn_E25tests (7).hdf5', 0, 12000)
gras_list = select_slices('../../GRAS/E_21cms_40Hz_100%_2pedal_no5ht.hdf5', 5000, 11000)

neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
gras_means = list(map(lambda voltages: np.mean(voltages), zip(*gras_list)))

neuron_means = normalization(neuron_means, -1, 1)
gras_means = normalization(gras_means, -1, 1)

# calculating latencies and amplitudes of mean values
bio_means_lat = sim_process(volts, bio_step, inhibition_zero=True)[0]
bio_means_amp = sim_process(volts, bio_step, inhibition_zero=True, after_latencies=True)[1]

# print("bio_means_lat = ", bio_means_lat)
# print("bio_means_amp = ", bio_means_amp)
neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=True)[0]
neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=True, after_latencies=True)[1]

# print("neuron_means_amp = ", neuron_means_amp)

gras_means_lat = sim_process(gras_means, sim_step, inhibition_zero=True)[0]
gras_means_amp = sim_process(gras_means, sim_step, inhibition_zero=True, after_latencies=True)[1]

# print("gras_means_amp = ", gras_means_amp)

bio_lat_nparray = np.array([np.array(x) for x in bio_means_lat])
bio_amp_nparray = np.array([np.array(x) for x in bio_means_amp])

neuron_lat_nparray = np.array([np.array(x) for x in neuron_means_lat])
neuron_amp_nparray = np.array([np.array(x) for x in neuron_means_amp])

gras_lat_nparray = np.array([np.array(x) for x in gras_means_lat])
gras_amp_nparray = np.array([np.array(x) for x in gras_means_amp])

bio_amp_nparray = bio_amp_nparray.T
bio_lat_nparray = bio_lat_nparray.T

neuron_amp_nparray = neuron_amp_nparray.T
neuron_lat_nparray = neuron_lat_nparray.T

gras_amp_nparray = gras_amp_nparray.T
gras_lat_nparray = gras_lat_nparray.T

bio_amp_nparray = np.reshape(bio_amp_nparray, (len(bio_means_lat), 1))
bio_lat_nparray = np.reshape(bio_lat_nparray, (len(bio_means_lat), 1))
# print("bio_amp_nparray = ", bio_amp_nparray)
# print("bio_lat_nparray = ", bio_lat_nparray)

neuron_amp_nparray = np.reshape(neuron_amp_nparray, (len(neuron_means_lat), 1))
neuron_lat_nparray = np.reshape(neuron_lat_nparray, (len(neuron_means_lat), 1))

print("neuron_amp_nparray = ", neuron_amp_nparray)

gras_amp_nparray = np.reshape(gras_amp_nparray, (len(gras_means_lat), 1))
gras_lat_nparray = np.reshape(gras_lat_nparray, (len(gras_means_lat), 1))

print("gras_amp_nparray = ", gras_amp_nparray)

# print("len(neuron_amp_nparray) = ", len(neuron_amp_nparray))
# print("len(neuron_lat_nparray) = ", len(neuron_lat_nparray))

data_bio = np.hstack((bio_amp_nparray, bio_lat_nparray))
data = np.hstack((neuron_amp_nparray, neuron_lat_nparray))
data_gras = np.hstack((gras_amp_nparray, gras_lat_nparray))
# print("data = ", data)

mu_bio = data_bio.mean(axis=0)
mu = data.mean(axis=0)
mu_gras = data_gras.mean(axis=0)

data_bio = data_bio - mu_bio
data = data - mu
data_gras = data_gras - mu_gras

eigenvectors_bio, eigenvalues_bio, V_bio = np.linalg.svd(data_bio.T, full_matrices=False)
eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
eigenvectors_gras, eigenvalues_gras, V_gras = np.linalg.svd(data_gras.T, full_matrices=False)

projected_data_bio = np.dot(data_bio, eigenvectors_bio)
projected_data = np.dot(data, eigenvectors)
projected_data_gras = np.dot(data_gras, eigenvectors_gras)

sigma_bio = projected_data_bio.std(axis=0).mean()
sigma = projected_data.std(axis=0).mean()
sigma_gras = projected_data_gras.std(axis=0).mean()
# print("eigenvectors = ", eigenvectors)

# print("bio_amp_nparray = ", bio_amp_nparray)
# print("bio_lat_nparray = ", bio_lat_nparray)

bio_coords = []
for sl in range(len(bio_means_amp)):
	one_dot = []
	one_dot.append(bio_means_amp[sl])
	one_dot.append(bio_means_lat[sl])
	bio_coords.append(one_dot)
# print("bio_coords = ", bio_coords)

neuron_coords = []
for sl in range(len(neuron_means_amp)):
	one_dot = []
	one_dot.append(neuron_means_amp[sl])
	one_dot.append(neuron_means_lat[sl])
	neuron_coords.append(one_dot)

gras_coords = []
for sl in range(len(gras_means_amp)):
	one_dot = []
	one_dot.append(gras_means_amp[sl])
	one_dot.append(gras_means_lat[sl])
	gras_coords.append(one_dot)

convex_bio = grahamscan(bio_coords)
convex_neuron = grahamscan(neuron_coords)
convex_gras = grahamscan(gras_coords)
# print("convex_bio = ", convex_bio)

convex_amp_bio = []
convex_lat_bio = []

for sl in convex_bio:
	convex_amp_bio.append(bio_means_amp[sl])
	convex_lat_bio.append(bio_means_lat[sl])
convex_amp_bio.append(convex_amp_bio[0])
convex_lat_bio.append(convex_lat_bio[0])

convex_amp_neuron = []
convex_lat_neuron = []
for sl in convex_neuron:
	convex_amp_neuron.append(neuron_means_amp[sl])
	convex_lat_neuron.append(neuron_means_lat[sl])
convex_amp_neuron.append(convex_amp_neuron[0])
convex_lat_neuron.append(convex_lat_neuron[0])

convex_amp_gras = []
convex_lat_gras = []

for sl in convex_gras:
	convex_amp_gras.append(gras_means_amp[sl])
	convex_lat_gras.append(gras_means_lat[sl])
convex_amp_gras.append(convex_amp_gras[0])
convex_lat_gras.append(convex_lat_gras[0])

fig, ax = plt.subplots()
ax.scatter(bio_amp_nparray, bio_lat_nparray, color='#a6261d', label='bio', s=80)
for axis in eigenvectors_bio:
	start, end = mu_bio, mu_bio + sigma_bio * axis
	ax.annotate(
		'', xy=end, xycoords='data',
		xytext= start, textcoords='data',
		arrowprops=dict(facecolor='#a6261d', width=4.0)
		)
	plt.plot(convex_amp_bio, convex_lat_bio, color='#a6261d')
	plt.fill_between(convex_amp_bio, convex_lat_bio, min(convex_lat_bio), color='#a6261d', alpha=0.3)
	break   # to draw only one vector

plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel('Amplitudes, mV', fontsize=28)
plt.ylabel('Latencies, ms', fontsize=28)

ax.scatter(neuron_amp_nparray, neuron_lat_nparray, color='#f2aa2e', label='neuron', s=80)
for axis in eigenvectors:
	start, end = mu, mu + sigma * axis
	print("start = ", len(start), start)
	print("end = ", len(end), end)
	ax.annotate(
		'', xy=end, xycoords='data',
		xytext= start, textcoords='data',
		arrowprops=dict(facecolor='#f2aa2e', width=4.0)
		)
	plt.plot(convex_amp_neuron, convex_lat_neuron, color='#f2aa2e')
	plt.fill_between(convex_amp_neuron, convex_lat_neuron, min(convex_lat_neuron), color='#f2aa2e', alpha=0.3)
	break

ax.scatter(gras_amp_nparray, gras_lat_nparray, color='#287a72', label='gras', s=80)
for axis in eigenvectors_gras:
	start, end = mu_gras, mu_gras + sigma_gras * axis
	ax.annotate(
		'', xy=end, xycoords='data',
		xytext= start, textcoords='data',
		arrowprops=dict(facecolor='#287a72', width=4.0)
		)
	plt.plot(convex_amp_gras, convex_lat_gras, color='#287a72')
	plt.fill_between(convex_amp_gras, convex_lat_gras, min(convex_lat_gras), color='#287a72', alpha=0.3)
	break

plt.legend()
plt.show()