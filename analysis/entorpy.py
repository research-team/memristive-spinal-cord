import numpy as np
from analysis.patterns_in_bio_data import bio_data_runs
import pandas as pd
from math import e
from analysis.functions import read_nest_data, read_neuron_data


def entropy(labels, base=None):
	vc = pd.Series(labels).value_counts(normalize=True, sort=False)
	base = e if base is None else base
	return -(vc * np.log(vc) / np.log(base)).sum()


bio_runs = bio_data_runs()
mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_runs)))
ent = entropy(mean_data)

# processing of the simulated data
nest = read_nest_data('../../nest-data/sim_extensor_eesF40_i100_s15cms_T.hdf5')
neuron = read_neuron_data('../../neuron-data/15cm.hdf5')
gpu = read_nest_data('../../GPU_extensor_eesF40_inh100_s15cms_T.hdf5')
mean_nest = list(map(lambda elements: np.mean(elements), zip(*nest)))
mean_neuron = list(map(lambda elements: np.mean(elements), zip(*neuron)))
mean_gpu = list(map(lambda elements: np.mean(elements), zip(*gpu)))
nest_ent = entropy(mean_nest)
neuron_ent = entropy(mean_neuron)
gpu_ent = entropy(mean_gpu)
print("entropy = ", gpu_ent)
print("standard deviation = ", np.std(mean_gpu))