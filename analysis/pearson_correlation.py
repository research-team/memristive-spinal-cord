import numpy as np
import pylab as plt


def __corr_coef_2D(A, B):
	# row-wise mean of input arrays & subtract from input arrays themeselves
	A_mA = A - A.mean(axis=1)[:, None]
	B_mB = B - B.mean(axis=1)[:, None]
	# sum of squares across rows
	ssA = (A_mA ** 2).sum(axis=1)[:, None]
	ssB = (B_mB ** 2).sum(axis=1)[:, None]
	# return correlation coefficients
	return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA, ssB.T))


def split_to(array):
	"""
	ToDO add info
	Args:
		array (list or np.ndarray): original data
	Returns:
		np.ndarray: mono answers
		np.ndarray: poly answers
	"""
	array = np.array(array)
	mono_end_step = int(10 / 0.25)

	return array[:, :mono_end_step], array[:, mono_end_step:]


def calc_correlation(array_a, array_b):
	"""
	ToDo add info
	Args:
		array_a (list): data
		array_b (list): data
	Returns:
		tuple: correlation coefficients of mono and poly answers
	"""
	# split bio data by mono and poly
	array_a_mono, array_a_poly = split_to(array_a)
	# split sim data by mono and poly
	array_b_mono, array_b_poly = split_to(array_b)
	# calculate correlations
	mono_corr = np.abs(__corr_coef_2D(array_b_mono, array_a_mono).flatten())
	poly_corr = np.abs(__corr_coef_2D(array_b_poly, array_a_poly).flatten())
	# plot boxplots
	plt.boxplot([mono_corr, poly_corr], showfliers=False, whis=[5, 95])

	for i, data, label in (1, mono_corr, 'mono'), (2, poly_corr, 'poly'):
		plt.plot([i] * len(data), data, '.', label=label)
	plt.legend()
	plt.show()
