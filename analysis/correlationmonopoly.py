import h5py as hdf5
from GRAS.PCA import prepare_data
from analysis.cut_several_steps_files import select_slices
import numpy as np
from matplotlib import pylab as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr

neuron_data_mono = []
neuron_data_poly = []
bio_data_mono = []
bio_data_poly = []

filepath = '../bio-data/hdf5/bio_sci_E_15cms_40Hz_i100_2pedal_no5ht_T_2016-06-12.hdf5'


def read_data(filepath, sign=1):
    with hdf5.File(filepath) as file:
        data_by_test = [sign * test_values[:] for test_values in file.values()]
    return data_by_test


bio_data = read_data(filepath, sign=1)
bio_data = prepare_data(bio_data)
bio_mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_data)))

neuron_list = np.array(
    select_slices('../bio-data/hdf5/mn_E15_speed25tests.hdf5', 0, 12000))
neuron_list = np.negative(neuron_list)
neuron_list = prepare_data(neuron_list)
neuron_mean = list(map(lambda elements: np.mean(elements), zip(*neuron_list)))
neuron_mean = neuron_mean[::10]

for i in range(12):
    neuron_data_mono.append(neuron_mean[i * 100:i * 100 + 40])

for i in range(12):
    neuron_data_poly.append(neuron_mean[i * 100:i * 100 + 100])

for i in range(12):
    bio_data_mono.append(bio_mean_data[i * 100:i * 100 + 40])

for i in range(12):
    bio_data_poly.append(bio_mean_data[i * 100:i * 100 + 100])

plt.plot(bio_mean_data, label='bio')
plt.plot(neuron_mean, label='neuron')
plt.legend()
plt.show()

neuron_data_mono = sum(neuron_data_mono, [])
bio_data_mono = sum(bio_data_mono, [])

neuron_data_poly = sum(neuron_data_poly, [])
bio_data_poly = sum(bio_data_poly, [])

corr, _ = spearmanr(neuron_data_mono, bio_data_mono)
print("spearman = ", corr)
pear, _ = pearsonr(neuron_data_mono, bio_data_mono)
print("pearson = ", pear)

corr, _ = spearmanr(neuron_data_poly, bio_data_poly)
print("spearman = ", corr)
pear, _ = pearsonr(neuron_data_poly, bio_data_poly)
print("pearson = ", pear)