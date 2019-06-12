import numpy as np
import h5py as hdf5
import scipy.io as sio
from itertools import chain
from matplotlib import pylab as plt
from analysis.histogram_lat_amp import sim_process
from analysis.functions import normalization, grahamscan

sim_step = 0.025
bio_step = 0.25

data_folder = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/bio_data/"


def read_mat_data(file_path):
	return sio.loadmat(file_path)


def trim_myogram(raw_data):
	volt_data = []
	stim_data = []
	slices_begin_time = []
	ms_pause = 0

	# data processing
	title_rmg = 'RMG'
	title_stim = 'Stim'

	for index, data_title in enumerate(raw_data['titles']):
		data_start = int(raw_data['datastart'][index]) - 1
		data_end = int(raw_data['dataend'][index])
		float_data = [round(float(x), 3) for x in raw_data['data'][0][data_start:data_end]]

		if title_rmg in data_title:
			volt_data = float_data
		if title_stim in data_title:
			stim_data = float_data

	for index in range(1, len(stim_data) - 1):
		if stim_data[index - 1] < stim_data[index] > stim_data[index + 1] and ms_pause <= 0 and stim_data[index] > 0.5:
			slices_begin_time.append(index)
			ms_pause = int(3 / bio_step)
		ms_pause -= 1

	offset = slices_begin_time[0]
	volt_data = volt_data[slices_begin_time[0]:slices_begin_time[-1]]
	slices_begin_time = [t - offset for t in slices_begin_time]

	return volt_data, slices_begin_time


def bio_data_runs():
	data = []

	filenames = [f"{index}.mat" for index in range(1, 6)]
	mat_datas = []

	for filename in filenames:
		raw_mat_data = read_mat_data(f"{data_folder}/{filename}")
		mat_datas.append(trim_myogram(raw_mat_data))

	data.append(mat_datas[0][0][0:1200])
	data.append(mat_datas[1][0][0:1200])
	data.append(mat_datas[2][0][1000:2200])
	data.append(mat_datas[3][0][900:2100])
	data.append(mat_datas[4][0][600:1800])

	return data


def read_data(filepath):
	with hdf5.File(filepath) as file:
		data_by_test = [test_values[:] for test_values in file.values()]
	return data_by_test


def select_slices(path, begin, end):
	return [data[begin:end] for data in read_data(path)]


def run():
	points_in_slice = int(25 / bio_step)

	# group by 100 in each data case
	all_bio_slices = [zip(*[iter(bio_data)] * points_in_slice) for bio_data in bio_data_runs()]
	# reverse data: 5 tests with 11 slices with each 100 dots -> 11 slices with 5 tests with each 100 dots
	all_bio_slices = list(zip(*all_bio_slices))

	instant_mean = []
	for slice_data in all_bio_slices:
		instant_mean.append(normalization([sum(map(abs, x)) for x in zip(*slice_data)], -1, 1))

	print("instant_mean = ", instant_mean)

	# creating the lists of voltages
	bio_volts = list(chain.from_iterable(instant_mean))

	neuron_list = select_slices(f'{data_folder}/mn_E25tests (7).hdf5', 0, 12000)
	neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
	neuron_means = normalization(neuron_means, -1, 1)

	gras_list = select_slices(f'{data_folder}/E_21cms_40Hz_100%_2pedal_no5ht.hdf5', 5000, 11000)
	gras_means = list(map(lambda voltages: np.mean(voltages), zip(*gras_list)))
	gras_means = normalization(gras_means, -1, 1)

	# calculating latencies and amplitudes of mean values
	bio_means_lat = sim_process(bio_volts, bio_step, inhibition_zero=True)[0]
	bio_means_amp = sim_process(bio_volts, bio_step, inhibition_zero=True, after_latencies=True)[1]

	neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=True)[0]
	neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=True, after_latencies=True)[1]

	gras_means_lat = sim_process(gras_means, sim_step, inhibition_zero=True)[0]
	gras_means_amp = sim_process(gras_means, sim_step, inhibition_zero=True, after_latencies=True)[1]

	bio_lat_nparray = np.array([np.array(x) for x in bio_means_lat]).T
	bio_amp_nparray = np.array([np.array(x) for x in bio_means_amp]).T

	neuron_lat_nparray = np.array([np.array(x) for x in neuron_means_lat]).T
	neuron_amp_nparray = np.array([np.array(x) for x in neuron_means_amp]).T

	gras_lat_nparray = np.array([np.array(x) for x in gras_means_lat]).T
	gras_amp_nparray = np.array([np.array(x) for x in gras_means_amp]).T

	bio_amp_nparray = np.reshape(bio_amp_nparray, (len(bio_means_lat), 1))
	bio_lat_nparray = np.reshape(bio_lat_nparray, (len(bio_means_lat), 1))

	neuron_amp_nparray = np.reshape(neuron_amp_nparray, (len(neuron_means_lat), 1))
	neuron_lat_nparray = np.reshape(neuron_lat_nparray, (len(neuron_means_lat), 1))

	gras_amp_nparray = np.reshape(gras_amp_nparray, (len(gras_means_lat), 1))
	gras_lat_nparray = np.reshape(gras_lat_nparray, (len(gras_means_lat), 1))

	data_bio = np.hstack((bio_amp_nparray, bio_lat_nparray))
	data_gras = np.hstack((gras_amp_nparray, gras_lat_nparray))
	data_neuron = np.hstack((neuron_amp_nparray, neuron_lat_nparray))

	mu_bio = data_bio.mean(axis=0)
	mu_gras = data_gras.mean(axis=0)
	mu_neuron = data_neuron.mean(axis=0)

	data_bio = data_bio - mu_bio
	data_gras = data_gras - mu_gras
	data_neuron = data_neuron - mu_neuron

	eigenvectors_bio, eigenvalues_bio, V_bio = np.linalg.svd(data_bio.T, full_matrices=False)
	eigenvectors_neuron, eigenvalues, V = np.linalg.svd(data_neuron.T, full_matrices=False)
	eigenvectors_gras, eigenvalues_gras, V_gras = np.linalg.svd(data_gras.T, full_matrices=False)

	projected_data_bio = np.dot(data_bio, eigenvectors_bio)
	projected_data = np.dot(data_neuron, eigenvectors_neuron)
	projected_data_gras = np.dot(data_gras, eigenvectors_gras)

	sigma_bio = projected_data_bio.std(axis=0).mean()
	sigma = projected_data.std(axis=0).mean()
	sigma_gras = projected_data_gras.std(axis=0).mean()

	bio_coords = list(zip(bio_means_amp, bio_means_lat))
	neuron_coords = list(zip(neuron_means_amp, neuron_means_lat))
	gras_coords = list(zip(gras_means_amp, gras_means_lat))

	convex_bio = grahamscan(bio_coords)
	convex_neuron = grahamscan(neuron_coords)
	convex_gras = grahamscan(gras_coords)

	# bio
	convex_amp_bio = [bio_means_amp[sl] for sl in convex_bio]
	convex_amp_bio.append(convex_amp_bio[0])

	convex_lat_bio = [bio_means_lat[sl] for sl in convex_bio]
	convex_lat_bio.append(convex_lat_bio[0])

	# neuron
	convex_amp_neuron = [neuron_means_amp[sl] for sl in convex_neuron]
	convex_amp_neuron.append(convex_amp_neuron[0])

	convex_lat_neuron = [neuron_means_lat[sl] for sl in convex_neuron]
	convex_lat_neuron.append(convex_lat_neuron[0])

	# gras
	convex_amp_gras = [gras_means_amp[sl] for sl in convex_gras]
	convex_amp_gras.append(convex_amp_gras[0])

	convex_lat_gras = [gras_means_lat[sl] for sl in convex_gras]
	convex_lat_gras.append(convex_lat_gras[0])

	# plot
	fig, ax = plt.subplots()

	# BIO
	ax.scatter(bio_amp_nparray, bio_lat_nparray, color='#a6261d', label='bio', s=80)
	start = mu_bio
	end = mu_bio + sigma_bio * eigenvectors_bio[0]
	ax.annotate('', xy=end, xycoords='data',
	            xytext=start, textcoords='data',
	            arrowprops=dict(facecolor='#a6261d', width=4.0))
	plt.plot(convex_amp_bio, convex_lat_bio, color='#a6261d')
	plt.fill_between(convex_amp_bio, convex_lat_bio, min(convex_lat_bio), color='#a6261d', alpha=0.3)

	# NEURON
	ax.scatter(neuron_amp_nparray, neuron_lat_nparray, color='#f2aa2e', label='neuron', s=80)
	start = mu_neuron
	end = mu_neuron + sigma * eigenvectors_neuron[0]
	ax.annotate('', xy=end, xycoords='data',
	            xytext=start, textcoords='data',
	            arrowprops=dict(facecolor='#f2aa2e', width=4.0))
	plt.plot(convex_amp_neuron, convex_lat_neuron, color='#f2aa2e')
	plt.fill_between(convex_amp_neuron, convex_lat_neuron, min(convex_lat_neuron), color='#f2aa2e', alpha=0.3)

	# GRAS
	ax.scatter(gras_amp_nparray, gras_lat_nparray, color='#287a72', label='gras', s=80)
	start = mu_gras
	end = mu_gras + sigma_gras * eigenvectors_gras[0]
	ax.annotate('', xy=end, xycoords='data',
	            xytext=start, textcoords='data',
	            arrowprops=dict(facecolor='#287a72', width=4.0))
	plt.plot(convex_amp_gras, convex_lat_gras, color='#287a72')
	plt.fill_between(convex_amp_gras, convex_lat_gras, min(convex_lat_gras), color='#287a72', alpha=0.3)

	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.xlabel('Amplitudes, mV', fontsize=28)
	plt.ylabel('Latencies, ms', fontsize=28)

	plt.legend()
	plt.show()


if __name__ == "__main__":
	run()
