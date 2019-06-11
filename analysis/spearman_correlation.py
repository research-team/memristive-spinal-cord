from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from matplotlib import pylab as plt
from analysis.cut_several_steps_files import select_slices
from scipy.stats import spearmanr
from analysis.functions import normalization
from scipy.stats import pearsonr
import seaborn as sns

align_coef = -0.001
omission_coef = -0.1
step = 0.25
color_bio = '#ed553b'
color_sim = '#079294'
bio_data = bio_data_runs()
bio1 = bio_data[0]
bio1 = normalization(bio1)
for dot in range(len(bio1)):
		bio1[dot] -= omission_coef
# take the []th run of the bio data
bio2 = bio_data[4]
# and align it by an align_coef
for run in bio_data:
	for dot in range(len(run)):
		run[dot] -= align_coef * dot

# for run in bio_data:
	# print("run after= ", run)
	# for slice in bio_data:
	# 	print("len(slice) = ", len(slice))
# calculate the mean data of all bio runs
bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_data)))
plt.plot(bio_mean_data)
plt.title('bio')
plt.show()
# and take the abs value
# for i in range(len(bio_mean_data)):
# 	bio_mean_data[i] = abs(bio_mean_data[i])
# normalize bio data
bio_mean_data = normalization(bio_mean_data)
# and put it down by an omission_coef
for dot in range(len(bio_mean_data)):
		bio_mean_data[dot] -= omission_coef

# plt.hist(bio_mean_data, color='red', edgecolor='black')
# plt.show()
# plot the hist and the line of distribution
bio_slices = []
offset = 0
for i in range(int(len(bio_mean_data) / 100)):
	# print("i = ", i)
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
		# print("j = ", j)
		bio_slices_tmp.append(bio_mean_data[j])
	# print("offset = ", offset)

	offset += 100
	bio_slices.append(bio_slices_tmp)
print("len(bio_slices) = ", len(bio_slices))
for sl in range(len(bio_slices)):
	# print("len(sl) = ", len(bio_slices[sl]))
	bio_slices[sl] = bio_slices[sl]
	# print("---")
	# print("len(sl) = ", len(bio_slices[sl]))

bio_without_ees = []
for sl in bio_slices:
	for s in sl:
		bio_without_ees.append(s)
print("bio_without_ees = ", len(bio_without_ees), bio_without_ees)

sns.distplot(bio_mean_data, hist=True, kde=True, bins=int(180/5), color='red', hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4})
plt.title('bio')
plt.show()

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
# print(len(bio_mean_data), bio_mean_data)
path_neuron = '../../neuron-data/mn_E25testsspeed15.hdf5'
path_gras = '../../GRAS/E_21cms_40Hz_100%_2pedal_no5ht.hdf5'

neuron_data = select_slices(path_neuron, 0, 12000)
gras_data = select_slices(path_gras, 5000, 11000)

# print("neuron_data = ", neuron_data)
neuron_mean_data = list(map(lambda elements: np.mean(elements), zip(*neuron_data)))
print("neuron_mean_data = ", neuron_mean_data)
plt.plot(neuron_mean_data)
plt.show()
# for i in range(len(neuron_mean_data)):
# 	neuron_mean_data[i] = abs(neuron_mean_data[i])
neuron_mean_data = normalization(neuron_mean_data)
# print("len(neuron_mean_data)  ", len(neuron_mean_data))

zeros_list = []
for i in range(120):
	zeros_list.append(neuron_mean_data[0])
print("len(neuron_mean_data)  ", len(neuron_mean_data))

# neuron_mean_data = zeros_list + neuron_mean_data
print("len(neuron_mean_data)  ", len(neuron_mean_data))

gras_mean_data = list(map(lambda elements: np.mean(elements), zip(*gras_data)))
gras_mean_data = normalization(gras_mean_data)

neuron_data_zoomed = []
for i in range(0, len(neuron_mean_data), 10):
	neuron_data_zoomed.append(neuron_mean_data[i])

neuron_slices = []
offset = 0
for i in range(int(len(neuron_data_zoomed) / 100)):
	print("i = ", i)
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
		# print("j = ", j)
		bio_slices_tmp.append(neuron_data_zoomed[j])
	print("offset = ", offset)

	offset += 100
	neuron_slices.append(bio_slices_tmp)
print("len(neuron_slices) = ", len(neuron_slices))
for sl in range(len(neuron_slices)):
	print("len(sl) = ", len(neuron_slices[sl]))
	neuron_slices[sl] = neuron_slices[sl]
	print("---")
	print("len(sl) = ", len(neuron_slices[sl]))

neuron_without_ees = []
for sl in neuron_slices:
	for s in sl:
		neuron_without_ees.append(s)
print("neuron_without_ees = ", len(neuron_without_ees), neuron_without_ees)


# plt.hist(neuron_data_zoomed, color='green', edgecolor='black')
sns.distplot(neuron_data_zoomed, hist=True, kde=True, bins=int(180/5), color='blue', hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4})
plt.title('neuron')
plt.show()

for index, sl in enumerate(bio_slices):
	offset = index
	plt.plot([s + offset for s in sl], color='red')
for index, sl in enumerate(neuron_slices):
	offset = index
	# plt.plot([s + offset for s in sl], color='green')
plt.show()

print("len(neuron_data_zoomed) = ", len(neuron_data_zoomed))

for i in range(len(neuron_data_zoomed), len(bio_mean_data)):
	neuron_data_zoomed.append(neuron_mean_data[i])
print("len(neuron_data_zoomed) = ", len(neuron_data_zoomed))

zeros_list_gras = []
for i in range(120):
	zeros_list_gras.append(gras_mean_data[0])
print("len(gras_mean_data) = ", len(gras_mean_data))

# gras_mean_data = zeros_list_gras + gras_mean_data
print("len(gras_mean_data) = ", len(gras_mean_data))
gras_data_zoomed = []
for i in range(0, len(gras_mean_data), 10):
	gras_data_zoomed.append(gras_mean_data[i])
print("gras_data_zoomed = ", len(gras_data_zoomed))
print("len(bio_mean_data) = ", len(bio_mean_data))
for i in range(len(gras_data_zoomed), len(bio_mean_data)):
	gras_data_zoomed.append(gras_mean_data[i])

print("gras_data_zoomed = ", len(gras_data_zoomed))

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
print("times_bio = ", len(times_bio))
times = [time * step for time in range(len(neuron_data_zoomed))]
plt.plot(times_bio, bio_mean_data, color=color_bio)
plt.plot(times, neuron_data_zoomed, color=color_sim, label='neuron')
plt.title("Align coefficient = {} \n Omission coefficient = {}".format(align_coef, omission_coef))
# xticks = []
# xlabels = []
# for i in range(0, len(bio_mean_data), 4):
# 	xticks.append(i)
# 	xlabels.append(i / 4)
# plt.xticks([int (i) for i in xticks if i % 5 == 0], xlabels)
plt.legend()
plt.show()
print("len(bio_mean_data) = ", len(bio_mean_data))
print("len(neuron_data_zoomed) = ", len(neuron_data_zoomed))
print("len(gras_data_zoomed) = ", len(gras_data_zoomed))
print()
corr, _ = spearmanr(neuron_data_zoomed, bio_mean_data)
print("spearman = ", corr)

# corr, _ = spearmanr(bio_mean_data, gras_data_zoomed)
# print("corr = ", corr)
# print("rank = ", np.rank(neuron_data_zoomed))
pear, _ = pearsonr(neuron_data_zoomed, bio_mean_data)
print("pearson = ", pear)
cov = np.cov(neuron_data_zoomed, bio_mean_data)
print("cov = ", cov)