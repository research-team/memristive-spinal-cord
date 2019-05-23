from analysis.real_data_slices import read_data, trim_myogram

path = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/10_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
path2 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/5_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
path3 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/4_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
path4 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/3_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
path5 = '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/1_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat'
# path6 = '../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz/7Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.mat'
# path7 = '../bio-data/10cms/Week6/7_WEEK6_Rat 40_8-8-2018.mat'
# path8 = '../bio-data/10cms/Week6/8_WEEK6_Rat 40_8-8-2018.mat'
# path9 = '../bio-data/10cms/Week6/9_WEEK6_Rat 40_8-8-2018.mat'
# path10 = '../bio-data/10cms/Week6/10_WEEK6_Rat 40_8-8-2018.mat'

folder = "/".join(path.split("/")[:-1])
align_coef = 0.001

raw_mat_data   = read_data(path)
raw_mat_data2  = read_data(path2)
raw_mat_data3  = read_data(path3)
raw_mat_data4  = read_data(path4)
raw_mat_data5  = read_data(path5)
# raw_mat_data6 = read_data(path6)
# raw_mat_data7 = read_data(path7)
# raw_mat_data8 = read_data(path8)
# raw_mat_data9 = read_data(path9)
# raw_mat_data10 = read_data(path10)

# print(f"mat_data = {mat_data[0]},\n {mat_data[1]}")
# mat_data7 = trim_myogram(raw_mat_data7, folder)
# mat_data8 = trim_myogram(raw_mat_data8, folder)
# mat_data9 = trim_myogram(raw_mat_data9, folder)
# mat_data10 = trim_myogram(raw_mat_data10, folder)


def bio_data_runs():
	data = []
	mat_data = trim_myogram(raw_mat_data, folder)
	mat_data2 = trim_myogram(raw_mat_data2, folder)
	mat_data3 = trim_myogram(raw_mat_data3, folder)
	mat_data4 = trim_myogram(raw_mat_data4, folder)
	mat_data5 = trim_myogram(raw_mat_data5, folder)
	# mat_data6 = trim_myogram(raw_mat_data6, folder)

	data.append([d for d in mat_data[0][0:1200]])    # bipedal control rats 21cm/s ex [0:600] no quipazine bipedal [0:1200]
	data.append([d for d in mat_data2[0][0:1200]])  # no quipazine bipedal [0:1200]
	data.append([d for d in mat_data3[0][1000:2200]])   # no quipazine bipedal [1000:2200]
	data.append([d for d in mat_data4[0][900:2100]])    # no quipazine bipedal [900:2100]
	data.append([d for d in mat_data5[0][600:1800]])    # no quipazine bipedal [600:1800]
	# data.append([d for d in mat_data6[0][:1200]])
	# data.append([d for d in mat_data7[0]])
	# data.append([d for d in mat_data8[0]])
	# data.append([d for d in mat_data9[0]])
	# data.append([d for d in mat_data10[0]])
	# for index, d in enumerate(data):
	# print("data = ", d)
	# plt.plot(d, label=index)
	# plt.legend()
	# plt.show()
	# print("len(data) = ", len(data))
	# for run in data:
	# 	for dot in range(len(run)):
	# 		run[dot] -= align_coef * dot
	return data


# data = bio_data_runs()
# plt.plot(data)
# plt.show()


# def bio_slices(data=data):
	all_bio_slices = []
	# forming list for the plot
	for k in range(len(data)):
		bio_slices = []
		offset = 0
		for i in range(int(len(data[k]) / 100)):
			bio_slices_tmp = []
			for j in range(offset, offset + 100):
				bio_slices_tmp.append(data[k][j])
			bio_slices.append(bio_slices_tmp)
			offset += 100
		all_bio_slices.append(bio_slices)   # list [4][16][100]
	all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]
	return all_bio_slices