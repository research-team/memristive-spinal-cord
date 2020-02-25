import ntpath
import numpy as np
from matplotlib import ticker

from analysis.PCA import get_lat_matrix, get_area_extrema_matrix
from analysis.PCA import form_ellipse
from analysis.functions import auto_prepare_data, parse_filename
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import pylab as plt
import h5py

def bootstr_for_test(filepath, convert_dstep_to=None):
	folder = ntpath.dirname(filepath)
	e_filename = ntpath.basename(filepath)
	if convert_dstep_to is None:
		dstep_to = parse_filename(e_filename)[-1]
	else:
		dstep_to = convert_dstep_to

	e_prepared_data = auto_prepare_data(folder, e_filename, dstep_to=dstep_to)
	e_latencies = get_lat_matrix(e_prepared_data, dstep_to)
	e_peaks, e_amplitudes = get_area_extrema_matrix(e_prepared_data, e_latencies, dstep_to)

	a = np.random.choice(e_latencies.ravel() * dstep_to, size=len(e_latencies.ravel() * dstep_to), replace=True, p=None)
	b = np.random.choice(e_amplitudes.ravel(), size=len(e_amplitudes.ravel()), replace=True, p=None)
	c = np.random.choice(e_peaks.ravel(), size=len(e_peaks.ravel()), replace=True, p=None)
	d = np.vstack((a, b, c))

	return d


def get_dice(sample1, sample2):
	'''

	:param sample1:
	:param sample2:
	:return:
	'''
	data_pack_xyz = []
	volume_sum = 0
	for sample in [sample1, sample2]:
		sample = sample.T
		pca = PCA(n_components=3)
		# coords is a matrix of coordinates, stacked as [[x1, y1, z1], ... , [xN, yN, zN]]
		pca.fit(sample)
		# get the center (mean value of points cloud)
		center = pca.mean_
		# get PCA vectors' head points (semi axis)
		vectors_points = [3 * np.sqrt(val) * vec for val, vec in zip(pca.explained_variance_, pca.components_)]
		vectors_points = np.array(vectors_points)
		# form full axis points (original vectors + mirrored vectors)
		axis_points = np.concatenate((vectors_points, -vectors_points), axis=0)
		# centering vectors and axis points
		vectors_points += center
		axis_points += center
		# calculate radii and rotation matrix based on axis points
		radii, rotation, matrixA = form_ellipse(axis_points)
		volume = (4 / 3) * np.pi * radii[0] * radii[1] * radii[2]
		volume_sum += volume

		phi = np.linspace(0, np.pi, 200)
		theta = np.linspace(0, 2 * np.pi, 200)
		# cartesian coordinates that correspond to the spherical angles
		x = radii[0] * np.outer(np.cos(theta), np.sin(phi))
		y = radii[1] * np.outer(np.sin(theta), np.sin(phi))
		z = radii[2] * np.outer(np.ones_like(theta), np.cos(phi))
		# rotate accordingly
		for i in range(len(x)):
			for j in range(len(x)):
				x[i, j], y[i, j], z[i, j] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
		data_pack_xyz.append((matrixA, center, x.flatten(), y.flatten(), z.flatten()))
	points_inside = []
	# get data of two ellipsoids: A matrix, center and points coordinates
	A1, C1, x1, y1, z1 = data_pack_xyz[0]
	A2, C2, x2, y2, z2 = data_pack_xyz[1]

	# based on stackoverflow.com/a/34385879/5891876 solution with own modernization
	# the equation for the surface of an ellipsoid is (x-c)TA(x-c)=1.
	# all we need to check is whether (x-c)TA(x-c) is less than 1 for each of points
	for coord in np.stack((x1, y1, z1), axis=1):
		if np.sum(np.dot(coord - C2, A2 * (coord - C2))) <= 1:
			points_inside.append(coord)
	# do the same for another ellipsoid
	for coord in np.stack((x2, y2, z2), axis=1):
		if np.sum(np.dot(coord - C1, A1 * (coord - C1))) <= 1:
			points_inside.append(coord)
	points_inside = np.array(points_inside)

	hull = ConvexHull(points_inside)
	v_intersection = hull.volume
	# dice
	dice = 2 * v_intersection / volume_sum

	return dice


def run_m_k(file_1, file_2, save_to, n, step):
	s = open(f"{save_to}", "w")
	samples1 = []
	samples2 = []
	dice_stat = []

	for i in range(n):
		print("STEP: ", i)
		pack3d = bootstr_for_test(file_1, convert_dstep_to=step)
		samples1.append(pack3d)
	print("________________________________")
	for i in range(n):
		print("STEP: ", i)
		pack3d = bootstr_for_test(file_2, convert_dstep_to=step)
		samples2.append(pack3d)

	print("START DICE: ")
	for s1, s2 in zip(samples1, samples2):
		dice_val = get_dice(s1, s2)
		dice_stat.append(dice_val)
		s.write(f"{dice_val}\n")
	# with h5py.File(f"{save_to}/'dice_coef.hdf5', 'w') as f:
	# 	dset = f.create_dataset("Dice", data=dice_stat)
	# f.close()
	return dice_stat


file1 = '..\\neuron_e_for_k_s\\neuron_E_AIR_13.5cms_40Hz_2pedal_0.025step.hdf5'
# file2 = '..\\neuron_e_for_k_s\\neuron_E_QPZ_13.5cms_40Hz_2pedal_0.025step.hdf5'
file2 = '..\\hdf5\\bio_E_AIR_13.5cms_40Hz_2pedal_0.1step.hdf5'
save_to = '..\\dice\\dice_b_n_air_13_5_2pedal_1000.txt'


dice = run_m_k(file1, file2, save_to, 1000, 0.1)
x = np.sort(dice)
y = np.arange(len(x)) / len(x)
plt.plot(x, y)
plt.show()

