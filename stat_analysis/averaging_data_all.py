import numpy as np
import h5py
import os
from analysis.functions import auto_prepare_data
save_folder = " "
root = ' '

filepaths = [
	(f'{root}\\  ', " "),
]



for filepath, new_filename in filepaths:
	print(filepath, new_filename)
	# read data
	with h5py.File(filepath, 'r') as f:
		data = [f_values[:] for f_values in f.values()]
	folder = os.path.dirname(filepath)
	filename = os.path.basename(filepath)

	step_size = float(filename.split("_")[-1].replace("step.hdf5", ""))
	prepared_data = auto_prepare_data(folder, filename, dstep_to=step_size)
	pack_numbers = prepared_data.shape[0]
	# combine slices data into one array
	prepared_data = np.reshape(prepared_data, (pack_numbers, -1))
	# get mean of data by line
	mean_data = np.mean(prepared_data, axis=0)
	# save new mean data
	with h5py.File(f"{save_folder}\\{new_filename}.hdf5", 'w') as file:
		file.create_dataset(data=mean_data, name='mean_data')
