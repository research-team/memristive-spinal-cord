import h5py as hdf5
path = '/home/anna/PycharmProjects/LAB/neuron-data/resATP.hdf5'
with hdf5.File(path) as file:
	for data in file.values():
		print(data)
		neuron_means = [d for d in data]
print(neuron_means)
# f = h5py.File('/home/anna/PycharmProjects/LAB/neuron-data/resATP.hdf5', 'r')
# list(f.keys())
# dset = f['test_0']
# print(dset[:100])
# f.close()