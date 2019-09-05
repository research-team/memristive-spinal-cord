import numpy as np
import pylab as plt
from analysis.functions import read_data, calc_boxplots


k_box_high = 1
k_box_low = 2

def compare(data):
	"""
	ToDo add info
	Args:
		data:
	"""
	box_position = 0
	percents = (5, 50, 95)
	prev_boxplot_bd = None
	size = list(range(2, len(D)))
	comparing_sizes = list(zip(size, size[1:]))
	#
	for size1, size2 in comparing_sizes:
		A = data[:size1, :]
		B = data[:size2, :]
		# check previous calculation
		if prev_boxplot_bd is None:
			boxplots_A = np.array([calc_boxplots(dots, percents) for dots in A.T])
			dbA = boxplots_A[:, k_box_high] - boxplots_A[:, k_box_low]
			prev_boxplot_bd = dbA
		# B delta
		boxplots_B = np.array([calc_boxplots(dots, percents) for dots in B.T])
		dbB = boxplots_B[:, k_box_high] - boxplots_B[:, k_box_low]
		#
		delta = dbB / prev_boxplot_bd
		# save previous calculation to save processing time
		prev_boxplot_bd = dbB

		plt.boxplot(delta, positions=[box_position], showfliers=False, widths=1)
		box_position += 1

	plt.xticks(range(1, len(comparing_sizes) + 1))
	plt.show()


if __name__ == "__main__":
	dataset1 = read_data('/home/alex/Downloads/Telegram Desktop/1.hdf5')
	dataset2 = read_data('/home/alex/Downloads/Telegram Desktop/2.hdf5')
	dataset3 = read_data('/home/alex/Downloads/Telegram Desktop/3.hdf5')
	# combine different datasets
	D = np.append(dataset1, dataset2, axis=0)
	D = np.append(D, dataset3, axis=0)
	# shuffle dataset
	np.random.shuffle(D)

	compare(D)
