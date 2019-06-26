import numpy as np
import pylab as plt
import h5py as hdf5
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from analysis.functions import normalization
from analysis.histogram_lat_amp import sim_process
from analysis.patterns_in_bio_data import bio_data_runs

bio_step = 0.25
sim_step = 0.025

data_folder = "/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/bio_data/"


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

	bio_data = bio_data_runs()

	all_bio_slices = []

	for k in range(len(bio_data)):
		bio_slices = []
		offset = 0
		for i in range(int(len(bio_data[k]) / 100)):
			bio_slices_tmp = []
			for j in range(offset, offset + 100):
				bio_slices_tmp.append(bio_data[k][j])
			bio_slices.append(bio_slices_tmp)
			offset += 100
		all_bio_slices.append(bio_slices)  # list [4][16][100]
	print("all_bio_slices = ", all_bio_slices)
	all_bio_slices = list(zip(*all_bio_slices))  # list [16][4][100]

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

	volts = []
	for i in instant_mean:
		for j in i:
			volts.append(j)

	# bio_means = np.sum(np.array([np.absolute(data) for data in read_data(f"{data_folder}/bio_15.hdf5")]), axis=0)
	# bio_means = normalization(bio_means, -1, 1)
	bio_means = volts

	neuron_list = select_slices('../../neuron-data/mn_E25tests.hdf5', 0, 6000)  # (7000, 11000) no_q_bip_21_FL
	# (13000, 17000) no_q_bip_15_FL # (0, 12000) no_q_bip_15_EX
	gras_list = select_slices('../../GRAS/E_21cms_40Hz_100%_2pedal_no5ht.hdf5', 5000, 11000)
	# (0, 5000) no_q_bip_21_FL
	# (6000, 10000) no_q_bip_15_FL
	# (10000, 22000) no_q_bip_15_EX

	neuron_means = np.sum(np.array([np.absolute(data) for data in neuron_list]), axis=0)
	neuron_means = normalization(neuron_means, -1, 1)

	gras_means = np.sum(np.array([np.absolute(data) for data in gras_list]), axis=0)
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
