import logging
import pylab as plt
import os
import numpy as np
import h5py as hdf5

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()

muscle = "flexor"

save_folder = f"/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights/{muscle}_plots"
result_folder = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights/"

colors = ["k", "g", "gray", "r", "brown", "y", "b"]


def plot_shadows_boxplot(data_per_test):
	step = 0.025
	slices_number = 33
	step_in_slice = 1000
	slice_length_ms = 25
	splitted = np.split(data_per_test, len(data_per_test) // step_in_slice)

	yticks = []
	fig, ax = plt.subplots(figsize=(16, 9))

	for slice_index, slice_data in enumerate(splitted, 1):
		slice_data += slice_index * 30
		yticks.append(slice_data[0])
		plt.plot(np.arange(len(slice_data)) * step, slice_data, color='gray')
		if slice_index % 33 == 0:
			print((slice_index - 1) // 33, len(splitted) // 33)
			ax.set_xlim(0, slice_length_ms)
			ax.set_xticks(range(slice_length_ms + 1))
			ax.set_yticks(yticks)
			ax.set_xticklabels(range(slice_length_ms + 1))
			ax.set_yticklabels(range(1, slices_number + 1))
			fig.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99)
			fig.savefig(f"{save_folder}/{(slice_index - 1) // 33}_{muscle}.png", dpi=250, format="png")
			plt.close()

			fig, ax = plt.subplots(figsize=(16, 9))
			yticks = []


	raise Ellipsis


	for t in range(0, len(data_per_test), slices_number * step_in_slice):
		slice_length_ms = 25
		print("starting ploting")
		yticks = []
		shared_x = np.arange(step_in_slice) * step

		n = 0
		for slice_index, data in enumerate(splitted):
			data += slice_index * 30
			ax.plot(shared_x, data[:, 0], color=(colors[n] if n < len(colors) else colors[n % len(colors)]), linewidth=0.7)
			yticks.append(data[0, 0])
			n+= 1

		ax.set_xlim(0, slice_length_ms)
		ax.set_xticks(range(slice_length_ms + 1))
		ax.set_xticklabels(range(slice_length_ms + 1))
		ax.set_yticks(yticks)
		ax.set_yticklabels(range(1, slices_number + 1))
		fig.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99)
		fig.savefig(f"{save_folder}/{t}_extensor.png", dpi=250, format="png")

		plt.close()
		k += slices_number
		u += slices_number

		logging.info(f"saved file in {save_folder}")

def f():
	for filename in filter(lambda f: f.endswith(".hdf5"), os.listdir(result_folder)):
		title = os.path.splitext(filename)[0]
		print(title)
		logging.info(f"start plotting {filename}")
		with hdf5.File(f"{result_folder}/{filename}") as hdf5_file:
			listed_data = np.array([data[:] for data in hdf5_file.values()][0])
			plot_shadows_boxplot(listed_data)

f()