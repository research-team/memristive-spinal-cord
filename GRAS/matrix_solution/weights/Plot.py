import logging
import pylab as plt
import os
import numpy as np
import h5py as hdf5

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()

save_folder = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights/extensor_plots"
result_folder = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights/"

percents = [25, 50, 75]

colors = ["black", "green", "gray", "red", "brown", "yellow", "blue"]

def calc_boxplots(dots):
	low_box_Q1, median, high_box_Q3 = np.percentile(dots, percents)
	IQR = high_box_Q3 - low_box_Q1
	Q1_15 = low_box_Q1 - 1.5 * IQR
	Q3_15 = high_box_Q3 + 1.5 * IQR
	high_whisker, low_whisker = high_box_Q3, low_box_Q1,

	for dot in dots:
		if high_box_Q3 < dot <= Q3_15 and dot > high_whisker:
			high_whisker = dot
		if Q1_15 <= dot < low_box_Q1 and dot < low_whisker:
			low_whisker = dot

	high_flier, low_flier = high_whisker, low_whisker
	for dot in dots:
		if dot > Q3_15 and dot > high_flier:
			high_flier = dot
		if dot < Q1_15 and dot < low_flier:
			low_filer = dot

	return median, high_box_Q3, low_box_Q1, high_whisker, low_whisker, high_flier, low_flier

def plot_shadows_boxplot(data_per_test):
	step = 0.025
	slice_lenght_ms = 25
	slices_number = 33
	step_in_slice = 1000
	k = 33000
	u = 0
	d_p_t_t = data_per_test.T
	for t in range(2):
		slice_length_ms = 25
		splitted = np.split(np.array([calc_boxplots(dot) for dot in d_p_t_t[u:k]]), slices_number)
		print(d_p_t_t[u:k])
		print("starting ploting")
		yticks = []
		shared_x = np.arange(step_in_slice) * step
		fig, ax = plt.subplots(figsize=(16, 9))

		for i, data in enumerate(splitted):
			data += i * 6
			ax.plot(shared_x, data[:, 0], color='r', linewidth=0.7)
			yticks.append(data[0, 0])

		ax.set_xlim(0, slice_length_ms)
		ax.set_xticks(range(slice_length_ms + 1))
		ax.set_xticklabels(range(slice_length_ms + 1))
		ax.set_yticks(yticks)
		ax.set_yticklabels(range(1, slices_number + 1))
		fig.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99)
		fig.savefig(f"{save_folder}/{t}_extensor.png", dpi=250, format="png")

		plt.close()
		k += k
		u += k

		logging.info(f"saved file in {save_folder}")

def f():
	for filename in filter(lambda f: f.endswith(".hdf5"), os.listdir(result_folder)):
		print("Start")
		title = os.path.splitext(filename)[0]
		logging.info(f"start plotting {filename}")
		with hdf5.File(f"{result_folder}/{filename}") as hdf5_file:
			listed_data = np.array([data[:] for data in hdf5_file.values()])
			plot_shadows_boxplot(listed_data)

f()