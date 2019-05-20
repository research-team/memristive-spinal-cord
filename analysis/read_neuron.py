from analysis.functions import read_neuron_data

path = '../../neuron-data/3steps_speed6_EX.hdf5'

data = read_neuron_data(path)
print("data = ", data)