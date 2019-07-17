from analysis.functions import read_data
from matplotlib import pylab as plt
from analysis.cut_several_steps_files import select_slices
import numpy as np

bio = read_data('../bio-data/hdf5/bio_sci_F_15cms_40Hz_i100_4pedal_no5ht_T_2016-06-12.hdf5')
for index, sl in enumerate(bio):
	offset = index * 16
	# plt.plot([s + offset for s in sl])
# plt.show()

neuron = select_slices('../../neuron-data/mn_E15_speed25tests.hdf5', 11000, 17000)
neuron_mean = list(map(lambda elements: np.mean(elements), zip(*neuron)))

plt.plot(neuron_mean)
plt.show()
for index, sl in enumerate(neuron):
	offset = index * 7
	plt.plot([s + offset for s in sl])
plt.show()