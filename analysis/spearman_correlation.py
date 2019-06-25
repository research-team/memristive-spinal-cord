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

align_coef = -0.0002    # flexors 15 / 21 cm/s bipedal no quipazine
omission_coef = -0.
sim_step = 0.025
step = 0.25
color_bio = '#a6261d'
color_sim = '#472650'
colors = ['#a6261d', '#472650', '#a6261d', '#472650', '#a6261d', '#472650',
          '#a6261d', '#472650', '#a6261d', '#472650', '#a6261d', '#472650']

bio_data = bio_data_runs()
for run in bio_data:
	for dot in range(len(run)):
		run[dot] -= align_coef * dot

print("len(bio_data) = ", len(bio_data))
# calculate the mean data of all bio runs
bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_data)))
# control_bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*control_data)))
zeros_list = []
# for i in range(8):
	# zeros_list.append(control_bio_mean_data[0])

# control_bio_mean_data = zeros_list + control_bio_mean_data

plt.plot(bio_mean_data, label='SCI')
# plt.plot(control_bio_mean_data, label='control')
plt.title('bio')
plt.legend()
plt.show()
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

print("len(bio_slices) = ", len(bio_slices))
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
	# bio_slices[sl] = bio_slices[sl][32:]
	print("---")
	print("len(sl) = ", len(bio_slices[sl]))

bio_without_ees = []
for sl in bio_slices:
	for s in sl:
		bio_without_ees.append(s)
print("bio_without_ees = ", len(bio_without_ees), bio_without_ees)

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
# bio_mean_data = normalization(bio_mean_data, -1, 1)
# plt.plot(bio_mean_data, color='blue')
# plt.show()

bio_lat = sim_process(bio_mean_data, step=0.25, inhibition_zero=True)[0]
print("bio_lat= ", bio_lat)

path_neuron = '../../neuron-data/mn_E25tests (8).hdf5'
path_gras = '../../GRAS/F_15cms_40Hz_100%_2pedal_no5ht.hdf5'

neuron_data = select_slices(path_neuron, 0, 6000)
gras_data = select_slices(path_gras, 0, 12000)

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
plt.plot(gras_mean_data)
plt.show()
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
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
		bio_slices_tmp.append(neuron_data_zoomed[j])
	offset += 100
	neuron_slices.append(bio_slices_tmp)
yticks = []
# bio_slices = bio_slices()

gras_slices = []
offset = 0
for i in range(int(len(gras_data_zoomed) / 100)):
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
		bio_slices_tmp.append(gras_data_zoomed[j])
	offset += 100
	gras_slices.append(bio_slices_tmp)

# for sl in range(len(neuron_slices)):
	# neuron_slices[sl] = neuron_slices[sl][32:]

neuron_without_ees = []
for sl in neuron_slices:
	for s in sl:
		neuron_without_ees.append(s)

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
	# plt.plot(times, [s + offset for s in sl], color=colors[index], label='SCI')
	coord = int(bio_lat[index] * 4)
	print("coord = ", coord)
	# plt.plot(times[coord], sl[coord] + offset, marker='.', markersize=12, color='red')
	# x_coor.append(times[coord])
	# y_coor.append(sl[coord] + offset)
	x_2_coors = []
	y_2_coors = []
	if len(x_coor) > 1:
		x_2_coors.append(x_coor[-2])
		x_2_coors.append(x_coor[-1])
		y_2_coors.append(y_coor[-2])
		y_2_coors.append(y_coor[-1])
		# plt.plot(x_2_coors, y_2_coors, linestyle='--', color='black')

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
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=32)
# plt.yticks(yticks, range(1, len(bio_slices) + 1), fontsize=18)
# plt.xlim(0, 25)
# plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
# plt.show()


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
times = [time * step for time in range(len(neuron_data_zoomed))]
plt.plot(times_bio, bio_mean_data, color=color_bio)
plt.plot(times, neuron_data_zoomed, color=color_sim, label='neuron')
# plt.title("Align coefficient = {} \n Omission coefficient = {}".format(align_coef, omission_coef))
xticks = []
xlabels = []
for i in range(0, len(bio_mean_data), 4):
	xticks.append(i)
	xlabels.append(i / 4)
# plt.xticks([int (i) for i in xticks if i % 5 == 0], xlabels)
# plt.legend()
plt.xlabel("Time, ms", fontsize=56)
plt.ylabel("Norm voltages, mV", fontsize=56)
plt.xticks(fontsize=56)
plt.yticks(fontsize=56)
plt.show()
# print("len(bio_mean_data) = ", len(bio_mean_data))
# print("len(neuron_data_zoomed) = ", len(neuron_data_zoomed))
# print("len(gras_data_zoomed) = ", len(gras_data_zoomed))
# print()
# corr, _ = spearmanr(neuron_without_ees, bio_without_ees)
# print("spearman = ", corr)

corr, _ = spearmanr(bio_mean_data, neuron_data_zoomed)
print("spearman = ", corr)
# print("rank = ", np.rank(neuron_data_zoomed))
pear, _ = pearsonr(neuron_data_zoomed, bio_mean_data)
print("pearson = ", pear)
cov = np.cov(neuron_data_zoomed, bio_mean_data)
# print("cov = ", cov)
print("len(bio_slices[0]) = ", len(bio_slices[0]))
print(len(neuron_slices[0]))
yticks = []
offset = 0
for index, sl in enumerate(bio_slices):
	offset = index * 0.25
	times = [time * step for time in range(len(bio_slices[0]))]
	for run in range(len(sl)):
		plt.plot(times, [s + offset for s in sl], linewidth=1, color=color_bio)
	yticks.append(sl[0] + offset)
for index, sl in enumerate(neuron_slices):
	offset = index * 0.25
	for run in range(len(sl)):
		plt.plot(times, [s + offset for s in sl], linewidth=1, color=color_sim)
# offset = 0
# for index, sl in enumerate(neuron_slices):
	# offset = index * 0.5 + 0.9
	# for run in range(len(sl)):
		# plt.plot(times, [s + offset for s in sl], linewidth=1, color=color_sim)
plt.yticks(yticks, range(1, len(bio_slices) + 1), fontsize=20)
plt.xticks(fontsize=56)
plt.xlabel('Time, ms', fontsize=56)
plt.ylabel('Slices', fontsize=56)
plt.xlim(0, 25)
# plt.title('Correlation = {}'.format(pear))
plt.show()