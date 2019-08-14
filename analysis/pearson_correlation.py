import numpy as np
import pylab as plt
import h5py as hdf5


def corr_coef_2D(A, B):
	# row-wise mean of input arrays & subtract from input arrays themeselves
	A_mA = A - A.mean(1)[:, None]
	B_mB = B - B.mean(1)[:, None]
	# sum of squares across rows
	ssA = (A_mA ** 2).sum(1)[:, None]
	ssB = (B_mB ** 2).sum(1)[:, None]
	# finally get corr coeff
	return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA, ssB.T))


def read_data(path, shrink=False):
	with hdf5.File(path) as file:
		if shrink:
			return np.array([d[:][::10][:1200] for d in file.values()])
		else:
			return np.array([d[:] for d in file.values()])


def split_to(data, mono=False, poly=False):
	mono_end_step = int(10 / 0.25)

	if not mono and not poly:
		raise Exception

	if poly:
		return data[:, mono_end_step:]

	if mono:
		return data[:, :mono_end_step]


def run(bio_path, sim_path):
	# prepare bio data
	bio_data = read_data(bio_path)
	bio_mono = split_to(bio_data, mono=True)
	bio_poly = split_to(bio_data, poly=True)
	# prepare sim data
	sim_data = read_data(sim_path, shrink=True)
	sim_mono = split_to(sim_data, mono=True)
	sim_poly = split_to(sim_data, poly=True)
	# calculate correlations
	mono_corr = np.abs(np.array(corr_coef_2D(sim_mono, bio_mono)).flatten())
	poly_corr = np.abs(np.array(corr_coef_2D(sim_poly, bio_poly)).flatten())
	# plot boxplots
	plt.boxplot([mono_corr, poly_corr], showfliers=False, whis=[5, 95])

	for i, data, label in (1, mono_corr, 'mono'), (2, poly_corr, 'poly'):
		plt.plot([i] * len(data), data, '.', label=label)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	bio_path = '/home/alex/GitHub/memristive-spinal-cord/data/bio/bio_sci_E_15cms_40Hz_i100_2pedal_no5ht_T_2016-06-12.hdf5'
	sim_path = '/home/alex/GitHub/memristive-spinal-cord/data/neuron/neuron_E_15cms_40Hz_i100_2pedal_no5ht_T.hdf5'
	run(bio_path, sim_path)