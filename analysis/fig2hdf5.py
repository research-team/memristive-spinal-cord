import os
import numpy as np
import pylab as plt
import h5py as hdf5
from scipy.io import loadmat
from analysis.functions import center_data_by_line, normalization

def fig2png(filename, title, rat, begin, end):
	# fl = ntpath.basename(filename)
	# '''e_air_13.5cms_1-5'''
	# meta = fl.replace(".fig", "").split("_")
	# sli = meta[-1].split('-')
	# begin = int(sli[0])
	# end = int(sli[1])
	# fold = ntpath.dirname(filename)
	# new_filename = "_".join(meta[:-1])

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


fig2png("/home/alex/Rat 25 2-14-2018 40 Hz trial 05_extensor.mat",0,0,0, 999)

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
