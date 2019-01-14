import h5py as hdf5
import scipy.io as sio
import numpy as np
import logging
import os

filename = ""
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('Converter')

def read(mat_path, hdf5_path):
	global filename

	sim_data_dict = {}
	filename = hdf5_path.split("/")[-1].replace(".hdf5", "")

	# read .hdf5 data
	logger.info("Read HDF5")
	with hdf5.File(hdf5_path, 'r') as file:
		for name, data in file.items():
			sim_data_dict[name] = data[:]

	# convert .hdf5 data to 0.25 step
	logger.info("Convert HDF5 data to 0.25 step")
	for k, v in sim_data_dict.items():
		sim_data_dict[k] = [np.mean(v[index:index+10]) for index in range(0, len(v), 10)]

	# read .mat data
	logger.info("Read MAT")
	mat_data = sio.loadmat(mat_path)
	titles = mat_data['titles']
	datas = mat_data['data'][0]
	data_starts = mat_data['datastart']
	data_ends = mat_data['dataend']
	tickrate = mat_data['tickrate'][0][0]

	for index, title in enumerate(titles):
		if "Stim" not in title:
			real_data = datas[int(data_starts[index])-1:int(data_ends[index])]
			return real_data, sim_data_dict
	raise Exception("Data was not read")


def write_data(real_data, sim_data, new_path):
	global filename
	filename = "MERGED" + filename

	abs_path = os.path.join(new_path, filename + ".hdf5" )
	with hdf5.File(abs_path, 'w') as file:
		file.create_dataset("real", data=real_data, compression="gzip")
		for k, data in sim_data.items():
			file.create_dataset(k, data=data, compression="gzip")
		logger.info("Saved {}".format(abs_path))

	# check data
	logger.info("Check data")
	with hdf5.File('{}/{}.hdf5'.format(new_path, filename), 'r') as file:
		for name, data in file.items():
			logger.info("Name: {:<8} \t size: {:<5} \t Data: [{} ... {}]".format(name, len(data), data[0], data[-1]))


def main():
	mat_path = "/home/alex/Downloads/SCI Rat-1_11-22-2016_RMG_40Hz_one_step.mat"
	hdf5_path = "/home/alex/Downloads/sim_healthy_neuron_extensor_eesF40_i100_s21cms_T.hdf5"
	new_path = "/home/alex/Downloads"

	real_data, sim_data = read(mat_path=mat_path, hdf5_path=hdf5_path)
	write_data(real_data, sim_data, new_path=new_path)


if __name__ == "__main__":
	main()
