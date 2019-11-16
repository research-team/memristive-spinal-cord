import numpy as np
import os
import fnmatch
import h5py
import pylab as plt
from analysis.functions import auto_prepare_data
import pickle

#path - папка со всеми папками, где лежат данные
#pattern - строчка,в которой содержатся опознавательные знаки названия файла, например, pattern = '*E*13.5*'
def averaging_data(path, pattern):
	arr = []
	path_f = []
	for d, dirs, files in os.walk(path):
		for f in files:
			path_p = os.path.join(d, f)
			path_f.append(path_p)

	for file in path_f:
		if ( fnmatch.fnmatchcase(file, pattern) ):
			filename = os.path.basename(file)
			folder = os.path.dirname(file)

			prepared_data = auto_prepare_data(folder, filename, dstep_to=0.1)
			shared_x = np.arange(len(prepared_data[0][0])) * 0.1
			for pack in prepared_data:
				for i, slice_data in enumerate(pack):
					plt.plot(shared_x, slice_data + i, color='gray')

			arr += list(prepared_data)

	arr = np.array(arr)

	pack_numbers = arr.shape[0]
	slices_numbers = arr.shape[1]
	dots_numbers = arr.shape[2]

	mean_array = np.zeros((slices_numbers, dots_numbers))

	for slice_index in range(slices_numbers):
		mean_array[slice_index, :] = np.mean(arr[:, slice_index, :], axis=0)

	for i, d in enumerate(mean_array):
		plt.plot(shared_x, d + i, color='g', linewidth=3)
	plt.xlim(0, 25)
	plt.show()
#	print(mean_array)

	return mean_array


