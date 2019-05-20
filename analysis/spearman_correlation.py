from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from matplotlib import pylab as plt
from analysis.cut_several_steps_files import select_slices
from scipy.stats import spearmanr
from analysis.functions import normalization



bio_data = bio_data_runs()
	# for slice in bio_data:
	# 	print("len(slice) = ", len(slice))
bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_data)))

	# print("len(bio_mean_data) = ", int(len(bio_mean_data) / 100))
	# plt.plot(bio_mean_data)
	# plt.show()
	# cut bio data to slices
offset = 0
bio_slices = []
for i in range(int(len(bio_mean_data) / 100)):
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
			# print("j = ", j)
		bio_slices_tmp.append(bio_mean_data[j])
	offset += 100
		# print("---")
	bio_slices.append(bio_slices_tmp)
		# print("len(bio_slices) = ", len(bio_slices))
	# print("bio_slices = ", len(bio_slices), len(bio_slices[0]))

# plot bio slices
yticks = []
step = 0.25
bio_slices = bio_slices()
for index, sl in enumerate(bio_slices):
	offset = index * 2
	times = [time * step for time in range(len(bio_slices[0]))]
	# for run in range(len(sl)):
		# print("sl = ", sl[run])
		# plt.plot(times, [s + offset for s in sl], linewidth=1)
	yticks.append(sl[0] + offset)
plt.yticks(yticks, range(1, len(bio_slices) + 1))
plt.xlim(0, 25)
# plt.show()
# bio_mean_data = normalization(bio_mean_data, -1, 1)
# plt.plot(bio_mean_data, color='blue')
# plt.show()
# print(len(bio_mean_data), bio_mean_data)
path_neuron = '../../neuron-data/3steps_speed15_EX.hdf5'
path_gras = '../../GRAS/E_15cms_40Hz_100%_2pedal_no5ht.hdf5'

neuron_data = select_slices(path_neuron, 32000, 44000)
gras_data = select_slices(path_gras, 10000, 22000)
print("len(gras_data) = ", len(gras_data[0]))
neuron_mean_data = list(map(lambda elements: np.mean(elements), zip(*neuron_data)))
neuron_mean_data = normalization(neuron_mean_data, -1, 1)

gras_mean_data = list(map(lambda elements: np.mean(elements), zip(*gras_data)))
gras_mean_data = normalization(gras_mean_data, -1, 1)

neuron_data_zoomed = []
for i in range(0, len(neuron_mean_data), 10):
	neuron_data_zoomed.append(neuron_mean_data[i])

gras_data_zoomed = []
for i in range(0, len(gras_mean_data), 10):
	gras_data_zoomed.append(gras_mean_data[i])

print("neuron_data = ", len(neuron_mean_data))
print("gras_mean_data = ", gras_mean_data)

# plt.plot(neuron_mean_data)
# plt.show()

# plt.plot(gras_mean_data)
# plt.show()

# plt.plot(gras_data_zoomed, color='red')
# plt.show()
# plt.scatter(bio_mean_data, neuron_data_zoomed, c='red')
# plt.show()
# plt.plot(gras_data_zoomed, color='green')
# plt.show()
# corr, _ = spearmanr(bio_mean_data, neuron_data_zoomed)
# print("corr = ", corr)
# cov = np.cov(bio_mean_data, neuron_data_zoomed)
# print("cov = ", cov)