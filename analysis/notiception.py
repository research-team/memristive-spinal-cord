import h5py as hdf5
import numpy as np
from matplotlib import pylab as plt
path = 'small_file_50001920.txt'
path_mat = '../bio-data/notiception/3 ser 2mkM (1).mat'
with open(path) as f:
	floats = list(map(float, f))
print(len(floats))
plt.plot(floats)
plt.show()
# arrays = {}
# f = hdf5.File(path_mat)
# for k, v in f.items():
#     arrays[k] = np.array(v)
# print(len(arrays['data'][0]))