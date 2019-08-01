import numpy as np
import os
import matplotlib.pyplot as plt
import os
import logging
import subprocess
import numpy as np
import h5py as hdf5
# from shadows_boxplot import plot_shadows_boxplot

with open("1_MN_E.dat") as file:
	extensor = np.array(list(map(float, file.read().split())))

# with open("1_MN_F.dat") as file:
# 	flexor = np.array(list(map(float, file.read().split())))

logger = logging.getLogger()
colors = ["black", "green", "gray", "red", "brown", "yellow", "blue"]
percents = [25, 50, 75]
save_folder = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights/extensor_plots"
result_folder = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights"
ees_hz = 40
sim_step = 0.025

time_simulation_in_steps = 9600000
len_slice_in_step = 1000
slices_number = 33
plots_number = 10

voltage_array = np.array(extensor[:time_simulation_in_steps])

def calc_boxplots(dots):
	low_box_Q1, median, high_box_Q3 = np.percentile(dots, percents)
	# calc borders
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
			low_flier = dot

	return median, high_box_Q3, low_box_Q1, high_whisker, low_whisker, high_flier, low_flier


def plot_shadows_boxplot(data_per_test, step, save_folder):
	# stuff variables
	slice_length_ms = 25
	slices_number = 33
	steps_in_slice = 1000
	splitted = np.split(np.array([calc_boxplots(dot) for dot in data_per_test.T]), slices_number)

	# build plot
	yticks = []
	shared_x = np.arange(steps_in_slice) * step

	fig, ax = plt.subplots(figsize=(16, 9))

	for j in range(plots_number):
		n = 0
		for i, data in enumerate(splitted):
			data += i * 6
			ax.plot(shared_x, data[:, 0], color=colors[len(colors) % n], linewidth=0.7)
			yticks.append(data[0, 0])
			n += 1

		# plotting stuff
		ax.set_xlim(0, slice_length_ms)
		ax.set_xticks(range(slice_length_ms + 1))
		ax.set_xticklabels(range(slice_length_ms + 1))
		ax.set_yticks(yticks)
		ax.set_yticklabels(range(1, slices_number + 1))
		fig.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99)
		fig.savefig(f"{save_folder}/{j}_plot_of_MN_E")

		plt.close()

		logging.info(f"saved file {j} in {save_folder}")

# with hdf5.File(f"1_MN_E.dat") as hdf5_file:
# 	listed_data = np.array([data[:] for data in hdf5_file.values()])
#
# 	print(type(listed_data))
#
# 	plot_shadows_boxplot(listed_data, ees_hz, sim_step)

with open("1_MN_E.dat") as file:
	extensor = np.array(list(map(float, file.read().split())))
	listed_data = extensor[0:time_simulation_in_steps]

step = 0.025
    slice_length_ms = 25
	slices_number = 33
	steps_in_slice = 1000
	splitted = np.split(np.array([calc_boxplots(dot) for dot in data_per_test.T]), slices_number)

	for t in range(1):
        yticks = []
        shared_x = np.arange(steps_in_slice) * step

        fig, ax = plt.subplots(figsize=(16, 9))

        n = 0
        for i, data in enumerate(splitted):
            data += i * 6
            ax.fill_between(shared_x, data[:, 6], data[:, 5], alpha=0.1, color='r')  # 6 f_low, 5 f_high
            ax.fill_between(shared_x, data[:, 4], data[:, 3], alpha=0.3, color='r')  # 4 w_low, 3 w_high
            ax.fill_between(shared_x, data[:, 2], data[:, 1], alpha=0.6, color='r')  # 2 b_low, 1 b_high
            ax.plot(shared_x, data[:, 0], color=colors[len(colors) % n], linewidth=0.7)  # 0 med
            yticks.append(data[0, 0])
            n += 1

        # plotting stuff
        ax.set_xlim(0, slice_length_ms)
        ax.set_xticks(range(slice_length_ms + 1))
        ax.set_xticklabels(range(slice_length_ms + 1))
        ax.set_yticks(yticks)
        ax.set_yticklabels(range(1, slices_number + 1))
        fig.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.99)
        fig.savefig(f"{save_folder}/{t}_{filename}.png", dpi=250, format="png")

        plt.close()

        logging.info(f"saved file in {save_folder}")

