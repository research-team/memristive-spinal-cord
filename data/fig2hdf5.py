import os
import numpy as np
import pylab as plt
import h5py as hdf5
from scipy.io import loadmat


def fig2png(filename, title, rat, begin, end):
	d = loadmat(filename, squeeze_me=True, struct_as_record=False)
	ax1 = d['hgS_070000'].children

	if np.size(ax1) > 1:
		ax1 = ax1[0]

	plt.figure(figsize=(16, 9))

	yticks = []
	# plt.suptitle(f"{title} [{begin} : {end}] \n {rat}")

	for i, line in enumerate(ax1.children, 1):
		if line.type == 'graph2d.lineseries':
			if begin <= i <= end:
				color = 'r'
			else:
				color = 'gray'

			x = line.properties.XData
			y = line.properties.YData
			yticks.append(y[0])

			plt.plot(x, y, color=color)
		if line.type == 'text':
			break

	plt.xlim(ax1.properties.XLim)
	plt.yticks(yticks, range(1, len(yticks) + 1))

	folder = "/home/alex/r"
	title_for_file = '_'.join(title.split())
	# plt.savefig(f"{folder}/{title_for_file}_{rat.replace('.fig', '')}.png", format="png", dpi=200)
	plt.show()
	plt.close()

# if __name__ == "__main__":
# 	filename =
# 	fig2png(filename, None, None, 1, 99)


def fig2hdf5(filename, title, rat, begin, end):
	d = loadmat(filename, squeeze_me=True, struct_as_record=False)
	ax1 = d['hgS_070000'].children

	if np.size(ax1) > 1:
		ax1 = ax1[0]

	y_data = []

	proper_index = 0
	for i, line in enumerate(ax1.children, 1):
		if line.type == 'graph2d.lineseries':
			if begin <= i <= end:
				y = line.properties.YData - 3 * proper_index
				y_data += list(y)
				proper_index += 1

		if line.type == 'text':
			break

	title = title.lower()
	print(title)
	*mode, muscle, speed, _ = title.split()

	mode = "_".join(mode)

	muscle = "E" if muscle == "extensor" else "F"
	qpz = "" if mode == "qpz" else "no"

	new_filename = f"bio_{muscle}_{speed}_40Hz_i100_2pedal_{qpz}5ht_T.hdf5"

	folder = f"/home/alex/bio_new/{mode}"

	if not os.path.exists(folder):
		os.makedirs(folder)

	with hdf5.File(f"{folder}/{new_filename}", "a") as file:
		file.create_dataset(data=y_data, name=rat)
