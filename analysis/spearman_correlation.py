from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from matplotlib import pylab as plt
from analysis.cut_several_steps_files import select_slices
from scipy.stats import spearmanr
from analysis.functions import normalization

align_coef = 0.001
omission_coef = 0
step = 0.25
bio_data = bio_data_runs()

for run in bio_data:
	# print("run = ", run)
	for dot in range(len(run)):
		run[dot] -= align_coef * dot

# for run in bio_data:
	# print("run after= ", run)
	# for slice in bio_data:
	# 	print("len(slice) = ", len(slice))
bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_data)))
# print("bio_mean_data = ", bio_mean_data)
bio_mean_data = normalization(bio_mean_data)

for dot in range(len(bio_mean_data)):
		bio_mean_data[dot] -= omission_coef

# print("len(bio_mean_data) = ", int(len(bio_mean_data) / 100))

times = [time * step for time in range(len(bio_mean_data))]
plt.plot(times, bio_mean_data, label='bio')
# print("len(bio_mean_data) = ", len(bio_mean_data))
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
path_neuron = '../../neuron-data/3steps_speed15_EX.hdf5'
path_gras = '../../GRAS/E_15cms_40Hz_100%_2pedal_no5ht.hdf5'

neuron_data = select_slices(path_neuron, 18000, 29000)
gras_data = select_slices(path_gras, 10000, 21000)

neuron_mean_data = list(map(lambda elements: np.mean(elements), zip(*neuron_data)))
neuron_mean_data = normalization(neuron_mean_data)
# print("len(neuron_mean_data)  ", len(neuron_mean_data))

zeros_list = []
for i in range(120):
	zeros_list.append(neuron_mean_data[0])
print("len(neuron_mean_data)  ", len(neuron_mean_data))

neuron_mean_data = zeros_list + neuron_mean_data
print("len(neuron_mean_data)  ", len(neuron_mean_data))

gras_mean_data = list(map(lambda elements: np.mean(elements), zip(*gras_data)))
gras_mean_data = normalization(gras_mean_data)

neuron_data_zoomed = []
for i in range(0, len(neuron_mean_data), 10):
	neuron_data_zoomed.append(neuron_mean_data[i])

print("len(neuron_data_zoomed) = ", len(neuron_data_zoomed))

for i in range(len(neuron_data_zoomed), len(bio_mean_data)):
	neuron_data_zoomed.append(neuron_mean_data[i])
print("len(neuron_data_zoomed) = ", len(neuron_data_zoomed))

zeros_list_gras = []
for i in range(120):
	zeros_list_gras.append(gras_mean_data[0])
print("len(gras_mean_data) = ", len(gras_mean_data))

gras_mean_data = zeros_list_gras + gras_mean_data
print("len(gras_mean_data) = ", len(gras_mean_data))
gras_data_zoomed = []
for i in range(0, len(gras_mean_data), 10):
	gras_data_zoomed.append(gras_mean_data[i])
print("gras_data_zoomed = ", len(gras_data_zoomed))
print("len(bio_mean_data) = ", len(bio_mean_data))
for i in range(len(gras_data_zoomed), len(bio_mean_data)):
	gras_data_zoomed.append(gras_mean_data[i])

# print("neuron_data = ", len(neuron_mean_data))
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
times = [time * step for time in range(len(gras_data_zoomed))]
plt.plot(times, gras_data_zoomed, color='green', label='gras')
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
corr, _ = spearmanr(bio_mean_data, gras_data_zoomed)
print("corr = ", corr)
cov = np.cov(bio_mean_data, neuron_data_zoomed)
print("cov = ", cov)