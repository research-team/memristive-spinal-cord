import os
import numpy as np
import pylab as plt
import h5py as hdf5
from mmap import mmap
from scipy.io import loadmat


def fig2png(filename, title, rat, begin, end):
	"""
	Args:
		filename:
		title:
		rat:
		begin:
		end:
	"""
	raise NotImplemented

	matfile = loadmat(filename, squeeze_me=True, struct_as_record=False)
	print(matfile)
	data = np.array(matfile['LMG_muscle']).T
	for i, k in enumerate(data):
		plt.plot(np.arange(len(k)) * 0.25, k + i)
	plt.show()
	raise Exception

	ax1 = d['hgS_070000'].children


	if np.size(ax1) > 1:
		ax1 = ax1[0]

	plt.figure(figsize=(16, 9))

	yticks = []
	plt.suptitle(f"{title} [{begin} : {end}] \n {rat}")
	y_data = []

	proper_index = 0
	for i, line in enumerate(ax1.children, 1):
		if line.type == 'graph2d.lineseries':
			if begin <= i <= end:
				x = line.properties.XData
				y = line.properties.YData #- 3 * proper_index
				y_data += list(y)
				proper_index += 1
				plt.plot(x, y)

		if line.type == 'text':
			break
	plt.plot()

	Y = np.array(y_data)
	centered = center_data_by_line(Y)
	normalized_data = normalization(centered, save_centering=True)

	slice_length = int(1000 / 40 / 0.1)
	slice_number = len(normalized_data) // slice_length
	sliced = np.split(normalized_data, slice_number)


	with open(f"{fold}/{new_filename}", 'w') as file:
		for d in sliced:
			file.write(" ".join(map(str, d)) + "\n")

	return
	raise Exception

	# plt.xlim(ax1.properties.XLim)
	plt.yticks(yticks, range(1, len(yticks) + 1))

	folder = "/home/alex/bio_data_png"
	# title_for_file = '_'.join(title.split())
	plt.tight_layout()
	plt.show()
	#plt.savefig(f"{folder}/{title_for_file}_{rat.replace('.fig', '')}.png", format="png", dpi=200)
	plt.close()


def fig2hdf5(filename, title, rat, begin, end):
	raise NotImplemented
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

	print(f"{len(y_data)} rat: {rat} \t title: {title}")

	title = title.lower()
	*mode, muscle, speed, _ = title.split()

	mode = "_".join(mode)

	muscle = "E" if muscle == "extensor" else "F"
	qpz = "" if mode == "qpz" else "no"

	new_filename = f"bio_{muscle}_{speed}_40Hz_i100_2pedal_{qpz}5ht_T.hdf5"

	rat_number = rat.split("_")[0][1:]
	folder = f"/home/alex/bio_data_hdf/{mode}/{rat_number}"

	if not os.path.exists(folder):
		os.makedirs(folder)

	with hdf5.File(f"{folder}/{new_filename}", "a") as file:
		file.create_dataset(data=y_data, name=rat)


def mapcount(filename):
	lines = 0
	with open(filename, "r+") as f:
		buf = mmap(f.fileno(), 0)
		readline = buf.readline
		while readline():
			lines += 1
	return lines


def txt2hdf5(path_source, path_target, buf_size=100000):
	"""
	ToDo add info, use log() instead of print()
	Args:
		path_source:
		path_target:
		buf_size:
	"""
	data_index = 0
	print("Calc a lines number of the file...")
	size = mapcount(path_source)
	print(size)

	print("Start converting...")
	with open(path_source, 'r') as txtfile:
		with hdf5.File(path_target, 'w') as hf:
			hf.create_dataset(name="dataset", shape=(size,), dtype=float, compression='gzip')
			tmp_lines = txtfile.readlines(buf_size)
			show_each = 0
			while tmp_lines:
				data = np.array(tmp_lines).astype(float)
				hf["dataset"][data_index:data_index + len(data)] = data
				data_index += len(data)
				tmp_lines = txtfile.readlines(buf_size)
				show_each += 1
				if show_each == 100:
					print(f"{data_index / size * 100:.2f}%")
					show_each = 0


if __name__ == "__main__":
	txt_path = "/home/alex/data.txt"
	hdf5_path = "/home/alex/new.hdf5"

	txt2hdf5(txt_path, hdf5_path, buf_size=200000)
