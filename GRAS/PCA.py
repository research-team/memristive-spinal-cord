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

bio_step = 0.25
sim_step = 0.025

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


def hex2rgb(hex_color):
	hex_color = hex_color.lstrip('#')
	return [int("".join(gr), 16) / 256 for gr in zip(*[iter(hex_color)] * 2)]


def length(v):
	return np.sqrt(v[0] ** 2 + v[1] ** 2)


def unit_vector(vector):
	return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def run(with_mono=False, debugging=False):
	after_latencies = not with_mono

	# keys
	X = 0
	Y = 1
	# calc how much dots in one slice
	dots_in_slice = int(25 / bio_step)

	# group by 100 in each data case
	all_bio_slices = [zip(*[iter(bio_data)] * dots_in_slice) for bio_data in bio_data_runs()]
	# reverse data: 5 tests with 11 slices with each 100 dots -> 11 slices with 5 tests with each 100 dots
	all_bio_slices = list(zip(*all_bio_slices))

	normalized_means_per_slice = []
	for slice_data in all_bio_slices:
		normalized_means_per_slice.append(normalization([sum(map(abs, x)) for x in zip(*slice_data)], -1, 1))

	# create the list of normalized mean voltages
	bio_means = list(chain.from_iterable(normalized_means_per_slice))

	neuron_list = select_slices(f'{data_folder}/mn_E25tests (7).hdf5', 0, 12000)
	neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
	neuron_means = normalization(neuron_means, -1, 1)

	gras_list = select_slices('/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/dat/MN_E.hdf5', 5000, 11000)
	gras_means = list(map(lambda voltages: np.mean(voltages), zip(*gras_list)))
	gras_means = normalization(gras_means, -1, 1)

	# calculating latencies and amplitudes of mean values
	bio_means_lat = sim_process(bio_means, bio_step, inhibition_zero=True)[0]
	bio_means_amp = sim_process(bio_means, bio_step, inhibition_zero=True, after_latencies=after_latencies)[1]

	neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=True)[0]
	neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=True, after_latencies=after_latencies)[1]

	gras_means_lat = sim_process(gras_means, sim_step, inhibition_zero=True)[0]
	gras_means_amp = sim_process(gras_means, sim_step, inhibition_zero=True, after_latencies=after_latencies)[1]

	bio_pack = [np.array(list(zip(bio_means_amp, bio_means_lat))), '#a6261d', 'bio']
	neuron_pack = [np.array(list(zip(neuron_means_amp, neuron_means_lat))), '#f2aa2e', 'neuron']
	gras_pack = [np.array(list(zip(gras_means_amp, gras_means_lat))), '#287a72', 'gras']

	# start plotting
	fig, ax = plt.subplots()

	# plot per data pack
	for coords, color, label in [bio_pack, neuron_pack, gras_pack]:
		pca = PCA(n_components=2)     # create PCA instance
		pca.fit(coords)               # fit the model with coords
		center = np.array(pca.mean_)  # get the center (mean value)

		# calc vectors
		vectors = []
		for v_length, vector in zip(pca.explained_variance_, pca.components_):
			v = vector * 3 * np.sqrt(v_length)
			vectors.append((center, center + v))

		# calc an angle between vector[first vector][top coords of vector] and vertical vector from the center
		first_vector = np.array(vectors[0][1])
		vertical = np.array([center[X], center[Y] + 10])
		angle_degrees = angle_between(vertical - center, first_vector - center)

		# check on angle sign (vector[first vector][top coord of vector][x coord]) > point1[x coord]
		sign = -1 if vectors[0][1][0] > vertical[0] else 1

		# calculate ellipse size
		ellipse_width = length(vectors[1][1] - center) * 2
		ellipse_height = length(vectors[0][1] - center) * 2

		# plot vectors
		for vector in vectors:
			ax.annotate('', vector[1], vector[0], arrowprops=dict(facecolor=color, linewidth=1.0))

		# plot dots
		ax.scatter(coords[:, X], coords[:, Y], color=color, label=label, s=80)
		if debugging:
			for index, x, y in zip(range(len(coords[:, X])), coords[:, X], coords[:, Y]):
				ax.text(x, y, index + 1)

		# plot ellipse
		ellipse = Ellipse(xy=center, width=ellipse_width, height=ellipse_height, angle=angle_degrees * sign)
		ellipse.set_fill(False)
		ellipse.set_edgecolor(hex2rgb(color))
		ax.add_artist(ellipse)

		# fill convex
		hull = ConvexHull(coords)
		ax.fill(coords[hull.vertices, X], coords[hull.vertices, Y], color=color, alpha=0.3)

	# plot atributes
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.xlabel('Amplitudes, mV', fontsize=28)
	plt.ylabel('Latencies, ms', fontsize=28)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	run()
