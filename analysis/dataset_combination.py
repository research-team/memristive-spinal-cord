import ntpath
import numpy as np
import h5py as hdf5
from itertools import combinations

path = '/home/alex/bio_article/bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5'

folder = ntpath.dirname(path)
filename = ntpath.basename(path)

with hdf5.File(path, 'r') as file:
	data_by_test = np.array([test_values[:] for test_values in file.values()])
	if not all(map(len, data_by_test)):
		raise Exception("hdf5 has an empty data!")

combs = list(combinations(range(len(data_by_test)), len(data_by_test) // 4))

mid = len(combs) // 2

pairs = [(p1, p2) for p1, p2 in zip(combs[:mid], combs[:mid:-1])]
