import math
import numpy as np
import pylab as plt
import h5py as hdf5
import scipy.io as sio
from itertools import chain
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from analysis.functions import normalization
from analysis.histogram_lat_amp import sim_process

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


def hex_to_rgb(hex_color):
	hex_color = hex_color.lstrip('#')
	hlen = len(hex_color)
	x = [int(hex_color[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3)]
	return [i / 256 for i in x]


def dotproduct(v1, v2):
	return sum((a * b) for a, b in zip(v1, v2))


def length(v):
	return math.sqrt(dotproduct(v, v))


def get_angle(v1, v2):
	return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))


def run():

	points_in_slice = int(25 / bio_step)

	# group by 100 in each data case
	all_bio_slices = [zip(*[iter(bio_data)] * points_in_slice) for bio_data in bio_data_runs()]
	# reverse data: 5 tests with 11 slices with each 100 dots -> 11 slices with 5 tests with each 100 dots
	all_bio_slices = list(zip(*all_bio_slices))

	instant_mean = []
	for slice_data in all_bio_slices:
		instant_mean.append(normalization([sum(map(abs, x)) for x in zip(*slice_data)], -1, 1))

	# creating the lists of voltages
	bio_volts = list(chain.from_iterable(instant_mean))

	neuron_list = select_slices(f'{data_folder}/mn_E25tests (7).hdf5', 0, 12000)
	neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
	neuron_means = normalization(neuron_means, -1, 1)

	gras_list = select_slices(f'{data_folder}/E_15cms_40Hz_100%_2pedal_no5ht.hdf5', 10000, 22000)
	gras_means = list(map(lambda voltages: np.mean(voltages), zip(*gras_list)))
	gras_means = normalization(gras_means, -1, 1)

	# calculating latencies and amplitudes of mean values
	bio_means_lat = sim_process(bio_volts, bio_step, inhibition_zero=True)[0]
	bio_means_amp = sim_process(bio_volts, bio_step, inhibition_zero=True, after_latencies=True)[1]

	neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=True)[0]
	neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=True, after_latencies=True)[1]

	gras_means_lat = sim_process(gras_means, sim_step, inhibition_zero=True)[0]
	gras_means_amp = sim_process(gras_means, sim_step, inhibition_zero=True, after_latencies=True)[1]

	bio_pack = [np.array(list(zip(bio_means_amp, bio_means_lat))), '#a6261d', 'bio']
	neuron_pack = [np.array(list(zip(neuron_means_amp, neuron_means_lat))), '#f2aa2e', 'neuron']
	gras_pack = [np.array(list(zip(gras_means_amp, gras_means_lat))), '#287a72', 'gras']

	# start plot
	fig, ax = plt.subplots()

	for coords, color, label in [bio_pack, neuron_pack, gras_pack]:
		# create PCA instance
		pca = PCA(n_components=2)
		# fit the model with coords
		pca.fit(coords)

		# calc vectors
		vectors = []
		for v_length, vector in zip(pca.explained_variance_, pca.components_):
			v = vector * 3 * np.sqrt(v_length)
			vectors.append((pca.mean_, pca.mean_ + v))

		# calc angle
		center = pca.mean_
		p1 = np.array([center[0], center[1] + 10])
		p2 = np.array(vectors[0][1])
		p_center = np.array(center)

		# check on angle sign
		if vectors[0][1][0] > p1[0]:
			sign = -1
		else:
			sign = 1

		ba = p1 - p_center
		bc = p2 - p_center

		cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

		angle = np.degrees(np.arccos(cosine_angle))
		print(f"angle {angle:.2f}")

		first = np.array(vectors[0][1])
		second = np.array(vectors[1][1])
		ba = first - p_center
		bc = second - p_center

		width_radius = (bc[0] ** 2 + bc[1] ** 2) ** 0.5
		height_radius = (ba[0] ** 2 + ba[1] ** 2) ** 0.5

		# plot vectors
		for vector in vectors:
			ax.annotate('', vector[1], vector[0], arrowprops=dict(facecolor=color, linewidth=1.0))

		# plot dots
		ax.scatter(coords[:, 0], coords[:, 1], color=color, label=label, s=80)

		# plot ellipse
		ell = Ellipse(xy=center, width=width_radius * 2, height=height_radius * 2, angle=angle * sign)
		ax.add_artist(ell)
		ell.set_fill(False)
		ell.set_edgecolor(hex_to_rgb(color))

		# fill convex
		hull = ConvexHull(coords)
		plt.fill(coords[hull.vertices, 0], coords[hull.vertices, 1], color=color, alpha=0.3)

	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.xlabel('Amplitudes, mV', fontsize=28)
	plt.ylabel('Latencies, ms', fontsize=28)

	plt.legend()
	plt.show()


if __name__ == "__main__":
	run()
