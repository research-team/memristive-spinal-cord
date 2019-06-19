from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from matplotlib import pylab as plt
from analysis.cut_several_steps_files import select_slices
from scipy.stats import spearmanr
from analysis.functions import normalization, sim_process
from scipy.stats import pearsonr
import seaborn as sns
from analysis.real_data_slices import read_data, trim_myogram
import matplotlib.patches as mpatches
from matplotlib import pyplot

align_coef = 0.0003
omission_coef = -0.4#5
sim_step = 0.025
step = 0.25
color_bio = '#a6261d'
color_sim = '#472650'
colors = ['#a6261d', '#472650', '#a6261d', '#472650', '#a6261d', '#472650',
          '#a6261d', '#472650', '#a6261d', '#472650', '#a6261d', '#472650']

# path = '../bio-data/quadrupedal control rats 9 m-min/6_0.87 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
# path2 = '../bio-data/quadrupedal control rats 9 m-min/7_1.1 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
# path3 = '../bio-data/quadrupedal control rats 9 m-min/8_1.1 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
# path4 = '../bio-data/quadrupedal control rats 9 m-min/9_1.1 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
# path5 = '../bio-data/quadrupedal control rats 9 m-min/10_1.1 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
#
# folder = "/".join(path.split("/")[:-1])
#
# raw_mat_data = read_data(path)
# raw_mat_data2 = read_data(path2)
# raw_mat_data3 = read_data(path3)
# raw_mat_data4 = read_data(path4)
# raw_mat_data5 = read_data(path5)
#
# bio_data = []
# mat_data = trim_myogram(raw_mat_data, folder)
# mat_data2 = trim_myogram(raw_mat_data2, folder)
# mat_data3 = trim_myogram(raw_mat_data3, folder)
# mat_data4 = trim_myogram(raw_mat_data4, folder)
# mat_data5 = trim_myogram(raw_mat_data5, folder)
#
# bio_data.append([d for d in mat_data[0][1200:2400]])  # bipedal control rats 21cm/s ex [0:600]
# bio_data.append([d for d in mat_data2[0][1200:2400]])  # no quipazine bipedal [0:1200]
# bio_data.append([d for d in mat_data3[0][1200:2400]])  # no quipazine bipedal [1000:2200]
# bio_data.append([d for d in mat_data4[0][1200:2400]])  # no quipazine bipedal [900:2100]
# bio_data.append([d for d in mat_data5[0][1200:2400]])  # no quipazine bipedal [600:1800] quadrupedal [1400:2600]
#
# path6 = '../bio-data/quadrupedal control rats 9 m-min/6_0.87 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
# path7 = '../bio-data/quadrupedal control rats 9 m-min/7_1.1 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
# path8 = '../bio-data/quadrupedal control rats 9 m-min/8_1.1 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
# path9 = '../bio-data/quadrupedal control rats 9 m-min/9_1.1 volts_QuadRat-16_5-09-2017_RMG&RTA_9m-min_one_step.mat'
# folder = "/".join(path.split("/")[:-1])
#
# raw_mat_data = read_data(path6)
# raw_mat_data2 = read_data(path7)
# raw_mat_data3 = read_data(path8)
# raw_mat_data4 = read_data(path9)
#
# control_data = []
# mat_data = trim_myogram(raw_mat_data, folder)
# mat_data2 = trim_myogram(raw_mat_data2, folder)
# mat_data3 = trim_myogram(raw_mat_data3, folder)
# mat_data4 = trim_myogram(raw_mat_data4, folder)
#
# control_data.append([-d for d in mat_data[0][0:1200]])  # bipedal control rats 21cm/s ex [0:600]
# control_data.append([-d for d in mat_data2[0][0:1200]])  # no quipazine bipedal [0:1200]
# control_data.append([-d for d in mat_data3[0][0:1200]])  # no quipazine bipedal [1000:2200]
# control_data.append([-d for d in mat_data4[0][0:1200]])  # no quipazine bipedal [900:2100]

# bio1 = bio_data[0]
# bio1 = normalization(bio1)
# for dot in range(len(bio1)):
# 		bio1[dot] -= omission_coef
# take the []th run of the bio data
# bio2 = bio_data[4]
# and align it by an align_coef
bio_data = bio_data_runs()
for run in bio_data:
	for dot in range(len(run)):
		run[dot] -= align_coef * dot

# for run in bio_data:
	# print("run after= ", run)
	# for slice in bio_data:
	# 	print("len(slice) = ", len(slice))
# calculate the mean data of all bio runs
bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_data)))
# control_bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*control_data)))
zeros_list = []
# for i in range(8):
	# zeros_list.append(control_bio_mean_data[0])

# control_bio_mean_data = zeros_list + control_bio_mean_data

# plt.plot(bio_mean_data, label='SCI')
# plt.plot(control_bio_mean_data, label='control')
# plt.title('bio')
# plt.legend()
# plt.show()
# and take the abs value
# for i in range(len(bio_mean_data)):
# 	bio_mean_data[i] = abs(bio_mean_data[i])
# normalize bio data
bio_mean_data = normalization(bio_mean_data)
# control_bio_mean_data = normalization(control_bio_mean_data)
# and put it down by an omission_coef
for dot in range(len(bio_mean_data)):
		bio_mean_data[dot] -= omission_coef

# plt.hist(bio_mean_data, color='red', edgecolor='black')
# plt.show()
# plot the hist and the line of distribution
bio_slices = []
offset = 0
for i in range(int(len(bio_mean_data) / 100)):
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
		bio_slices_tmp.append(bio_mean_data[j])

	offset += 100
	bio_slices.append(bio_slices_tmp)

control_bio_slices = []
offset = 0
# for i in range(int(len(control_bio_mean_data) / 100)):
# 	bio_slices_tmp = []
# 	for j in range(offset, offset + 100):
# 		bio_slices_tmp.append(control_bio_mean_data[j])
# 	offset += 100
# 	control_bio_slices.append(bio_slices_tmp)
# print("len(bio_slices) = ", len(bio_slices))
for sl in range(len(bio_slices)):
	# print("len(sl) = ", len(bio_slices[sl]))
	bio_slices[sl] = bio_slices[sl]
	# print("---")
	# print("len(sl) = ", len(bio_slices[sl]))

bio_without_ees = []
for sl in bio_slices:
	for s in sl:
		bio_without_ees.append(s)
# print("bio_without_ees = ", len(bio_without_ees), bio_without_ees)

# sns.distplot(bio_mean_data, hist=True, kde=True, bins=int(180/5), color='red', hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth':4})
# plt.title('bio')
# plt.show()

times = [time * step for time in range(len(bio_mean_data))]
# plt.plot(times, bio_mean_data, label='bio')

offset = 0
"""bio_slices = []
for i in range(int(len(bio_mean_data) / 100)):
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
			# print("j = ", j)
		bio_slices_tmp.append(bio_mean_data[j])
	offset += 100
	bio_slices.append(bio_slices_tmp)"""
		# print("len(bio_slices) = ", len(bio_slices))
	# print("bio_slices = ", len(bio_slices), len(bio_slices[0]))

# plot bio slices
"""yticks = []
# bio_slices = bio_slices()
for index, sl in enumerate(bio_slices):
	offset = index * 2
	times = [time * step for time in range(len(bio_slices[0]))]
	# for run in range(len(sl)):
		# print("sl = ", sl[run])
		# plt.plot(times, [s + offset for s in sl], linewidth=1)
	yticks.append(sl[0] + offset)"""
# plt.yticks(yticks, range(1, len(bio_slices) + 1))
# plt.xlim(0, 25)
# plt.show()
# bio_mean_data = normalization(bio_mean_data, -1, 1)
# plt.plot(bio_mean_data, color='blue')
# plt.show()

bio_lat = sim_process(bio_mean_data, step=0.25, inhibition_zero=True)[0]
print("bio_lat= ", bio_lat)

path_neuron = '../../neuron-data/mn_F25tests (5).hdf5'
path_gras = '../../GRAS/F_15cms_40Hz_100%_2pedal_no5ht.hdf5'

neuron_data = select_slices(path_neuron, 17000, 22000)
gras_data = select_slices(path_gras, 0, 10000)

neuron_mean_data = list(map(lambda elements: np.mean(elements), zip(*neuron_data)))
# plt.plot(bio_mean_data, label='bio')
# # plt.plot(control_bio_mean_data, label='control')
# plt.plot(neuron_mean_data, label='neuron')
# plt.legend()
# plt.show()
# # for i in range(len(neuron_mean_data)):
# # 	neuron_mean_data[i] = abs(neuron_mean_data[i])
neuron_mean_data = normalization(neuron_mean_data)

zeros_list = []
for i in range(120):
	zeros_list.append(neuron_mean_data[0])

# neuron_mean_data = zeros_list + neuron_mean_data

gras_mean_data = list(map(lambda elements: np.mean(elements), zip(*gras_data)))
gras_mean_data = normalization(gras_mean_data)

neuron_data_zoomed = []
for i in range(0, len(neuron_mean_data), 10):
	neuron_data_zoomed.append(neuron_mean_data[i])

gras_data_zoomed = []
for i in range(0, len(gras_mean_data), 10):
	gras_data_zoomed.append(gras_mean_data[i])

times = [time * step for time in range(len(bio_mean_data))]  # divide by 10 to convert to ms step
# plt.plot(times, bio_mean_data, color=color_bio, label='bio')
# plt.plot(neuron_data_zoomed, label='neuron')
# plt.legend()
# plt.xlabel("Time, ms", fontsize=56)
# plt.ylabel("Norm voltages, mV", fontsize=56)
# plt.xticks(fontsize=56)
# plt.yticks(fontsize=56)
# plt.show()

neuron_slices = []
offset = 0
for i in range(int(len(neuron_data_zoomed) / 100)):
	# print("i = ", i)
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
		bio_slices_tmp.append(neuron_data_zoomed[j])
	# print("offset = ", offset)

	offset += 100
	neuron_slices.append(bio_slices_tmp)

gras_slices = []
offset = 0
for i in range(int(len(gras_data_zoomed) / 100)):
	# print("i = ", i)
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
		bio_slices_tmp.append(gras_data_zoomed[j])
	offset += 100
	gras_slices.append(bio_slices_tmp)
# print("len(neuron_slices) = ", len(neuron_slices))
for sl in range(len(neuron_slices)):
	# print("len(sl) = ", len(neuron_slices[sl]))
	neuron_slices[sl] = neuron_slices[sl]
	# print("---")
	# print("len(sl) = ", len(neuron_slices[sl]))

neuron_without_ees = []
for sl in neuron_slices:
	for s in sl:
		neuron_without_ees.append(s)
# print("neuron_without_ees = ", len(neuron_without_ees), neuron_without_ees)


# plt.hist(neuron_data_zoomed, color='green', edgecolor='black')
# sns.distplot(neuron_data_zoomed, hist=True, kde=True, bins=int(180/5), color='blue', hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth':4})
# plt.title('neuron')
# plt.show()

yticks = []
x_coor = []
y_coor = []

for index, sl in enumerate(bio_slices):
	offset = index * 0.25
	times = [time * step for time in range(len(sl))]  # divide by 10 to convert to ms step
	yticks.append(sl[0] + offset)
	plt.plot(times, [s + offset for s in sl], color=colors[index], label='SCI')
	coord = int(bio_lat[index] * 4)
	print("coord = ", coord)
	print("times[{}] = ".format(coord), times[coord])
	print("sl[{}] = ".format(coord), sl[coord])
	plt.plot(times[coord], sl[coord] + offset, marker='.', markersize=12, color='red')
	x_coor.append(times[coord])
	y_coor.append(sl[coord] + offset)
	x_2_coors = []
	y_2_coors = []
	if len(x_coor) > 1:
		x_2_coors.append(x_coor[-2])
		x_2_coors.append(x_coor[-1])
		y_2_coors.append(y_coor[-2])
		y_2_coors.append(y_coor[-1])
		plt.plot(x_2_coors, y_2_coors, linestyle='--', color='black')

# for index, sl in enumerate(control_bio_slices):
# 	offset = index
	# plt.plot([s + offset for s in sl], color='green', label='control')
# for index, sl in enumerate(gras_slices):
# 	offset = index * 0.25
# 	times = [time * step for time in range(len(sl))]  # divide by 10 to convert to ms step
	# plt.plot(times, [s + offset for s in sl], color='#472650')
# colors = ["#a6261d", "#472650"]
# texts = ["bio", "neuron"]
# patches = [mpatches.Patch(color=colors[i], label="{:s}".format(texts[i])) for i in range(len(texts))]
# pyplot.legend(handles=patches, loc='best')
ticks = []
labels = []
for i in range(0, len(bio_mean_data), 4):
	ticks.append(i)
	labels.append(int(i / 4))
# plt.xticks(ticks, labels, fontsize=32)
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=32)
plt.yticks(yticks, range(1, len(bio_slices) + 1), fontsize=18)
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()


for i in range(len(neuron_data_zoomed), len(bio_mean_data)):
	neuron_data_zoomed.append(neuron_mean_data[i])

zeros_list_gras = []
for i in range(120):
	zeros_list_gras.append(gras_mean_data[0])

# gras_mean_data = zeros_list_gras + gras_mean_data
gras_data_zoomed = []
for i in range(0, len(gras_mean_data), 10):
	gras_data_zoomed.append(gras_mean_data[i])
for i in range(len(gras_data_zoomed), len(bio_mean_data)):
	gras_data_zoomed.append(gras_mean_data[i])

# print("gras_data_zoomed = ", len(gras_data_zoomed))

# plt.plot(neuron_mean_data)
# plt.show()

# plt.plot(gras_mean_data)
# plt.show()

# plt.plot(gras_data_zoomed, color='red')
# plt.show()
# plt.scatter([range(len(bio_mean_data))], bio_mean_data, c='red', label='bio')
# plt.scatter([range(len(neuron_data_zoomed))], neuron_data_zoomed, c='green', label='neuron')
# plt.legend()
# plt.show()
times_bio = [time * step for time in range(len(bio_mean_data))]
# print("times_bio = ", len(times_bio))
times = [time * step for time in range(len(neuron_data_zoomed))]
# plt.plot(times_bio, bio_mean_data, color=color_bio)
# plt.plot(times, gras_data_zoomed, color=color_sim, label='gras')
# plt.title("Align coefficient = {} \n Omission coefficient = {}".format(align_coef, omission_coef))
# xticks = []
# xlabels = []
# for i in range(0, len(bio_mean_data), 4):
# 	xticks.append(i)
# 	xlabels.append(i / 4)
# plt.xticks([int (i) for i in xticks if i % 5 == 0], xlabels)
# plt.legend()
# plt.xlabel("Time, ms", fontsize=56)
# plt.ylabel("Norm voltages, mV", fontsize=56)
# plt.xticks(fontsize=56)
# plt.yticks(fontsize=56)
# plt.show()
# print("len(bio_mean_data) = ", len(bio_mean_data))
# print("len(neuron_data_zoomed) = ", len(neuron_data_zoomed))
# print("len(gras_data_zoomed) = ", len(gras_data_zoomed))
# print()
corr, _ = spearmanr(neuron_data_zoomed, bio_mean_data)
# print("spearman = ", corr)

# corr, _ = spearmanr(bio_mean_data, gras_data_zoomed)
# print("corr = ", corr)
# print("rank = ", np.rank(neuron_data_zoomed))
pear, _ = pearsonr(neuron_data_zoomed, bio_mean_data)
# print("pearson = ", pear)
cov = np.cov(neuron_data_zoomed, bio_mean_data)
# print("cov = ", cov)