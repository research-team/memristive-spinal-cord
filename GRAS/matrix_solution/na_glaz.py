import numpy as np
import pylab as plt
import h5py as hdf5

data_folder = "bio_data/"


def read_data(filepath):
	with hdf5.File(filepath) as file:
		data_by_test = np.array([test_values[:] for test_values in file.values()])
	return data_by_test


def run():
	yticks = []
	y_offset = 3
	bio_step = 0.25
	slice_in_ms = 25

	bio_data = read_data(f"{data_folder}/bio_15.hdf5")
	slices_number = int(len(bio_data[0]) / (slice_in_ms / bio_step))

	splitted_per_slice = np.split(np.array(bio_data[0]), slices_number)
	shared_x = np.arange(slice_in_ms / bio_step) * bio_step

	plt.figure(figsize=(16, 9))
	for slice_index, data in enumerate(splitted_per_slice):
		data += slice_index * y_offset # is a link (!)
		plt.plot(shared_x, data, color='r')
		yticks.append(data[0])

	dots = [15.044, 15, 14.32, 16.6, 16.9, 12.1, 16.2, 16.5, 19.7, 24.7, 24.7, 24.7]
	vals = [splitted_per_slice[ind][int(dot / bio_step)] for ind, dot in enumerate(dots)]

	plt.plot(dots, vals, '.', markersize=10, color='k')
	plt.plot(dots, vals, color='b', linewidth=3)

	plt.yticks(yticks, range(1, slices_number+1))
	plt.suptitle("Na glaz: 15cms control rat")
	plt.xlim(0, slice_in_ms)
	plt.show()


if __name__ == "__main__":
	run()
