import pylab as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from analysis.real_data_slices import read_data, trim_myogram
chunks_NEST = []
chunks_NEURON = []
chunks_real = []
deltas_std = []
deltas_std_nest = []
deltas_std_neuron = []
# Common parameters
delta_step_size = 1
sim_time = 125
V_to_uV = 1000000  # convert Volts to micro-Volts (V -> uV)
test_number = 3
neuron_number = 169
cutting_value = 100

print("Extensor")

# collect data for NEST
tests_nest = {k: {} for k in range(test_number)}
for neuron_id in range(test_number):
	nrns_nest = set()
	with open('NEST/{}.dat'.format(neuron_id), 'r') as file:
		for line in file:
			nrn_id, time, volt = line.split("\t")[:3]
			time = float(time)
			if time not in tests_nest[neuron_id].keys():
				tests_nest[neuron_id][time] = 0
			tests_nest[neuron_id][time] += float(volt)
			nrns_nest.add(nrn_id)
	for time in tests_nest[neuron_id].keys():
		tests_nest[neuron_id][time] /= len(nrns_nest)
# calculate mean
nest_tests = []
for k, v in tests_nest.items():
	nest_tests.append(list(v.values()))
tmp = list(map(lambda x: np.mean(x), zip(*nest_tests)))
nest_means = [tmp[0]] + tmp
# calculate mean without EES
nest_without_EES = []
offset = 250
for iter_begin in range(len(nest_means))[::offset]:
	nest_without_EES += [0 for _ in range(cutting_value)] + nest_means[iter_begin + cutting_value:iter_begin + offset]

# collect data for NEURON
neuron_tests = []
for neuron_test_number in range(test_number):
	tmp = []
	for neuron_id in range(neuron_number):
		with open('res2509/volMN{}v{}.txt'.format(neuron_id, neuron_test_number), 'r') as file:
			tmp.append([float(i) * V_to_uV for i in file.read().split("\n")[1:-2]])
	neuron_tests.append([elem * 10**7 for elem in list(map(lambda x: np.mean(x), zip(*tmp)))])
# calculate mean
neuron_means = []
raw_means = list(map(lambda x: np.mean(x), zip(*neuron_tests)))
offset = 4
for iter_begin in range(len(raw_means))[::4]:
	neuron_means.append(np.mean(raw_means[iter_begin:iter_begin+offset]))
# calculate mean without EES
neuron_without_EES = []
offset = 250
for iter_begin in range(len(neuron_means))[::offset]:
	neuron_without_EES += [0 for _ in range(cutting_value)] + neuron_means[iter_begin + cutting_value:iter_begin + offset]
# FixMe neuron_without_EES = [-x for x in neuron_without_EES]
#collect real data

raw_real_data = read_data('../bio-data//SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
processed_real_data = trim_myogram(raw_real_data)
# pl = plot_1d(processed_real_data[0], processed_real_data[1])
real_tests = processed_real_data[0]
# real_data = max(raw_real_data['data'])
# plt.plot(real_data)
# plt.show()
#calculate mean
real_pairs = []
offset_real = 2
for iter_begin_real in range(len(real_tests) - 300)[::offset_real]:
	real_pairs.append(np.mean(real_tests[iter_begin_real:iter_begin_real + offset_real]))
for iter_begin_real in range(len(real_tests) - 300, len(real_tests)):
	real_pairs.append(real_tests[iter_begin_real])
# real_tests = []
# # for k, v in real_data.items():
# for i in range (len(real_data)):
#  	real_tests.append(real_data[i])
# tm = list(map(lambda x: np.mean(x), zip(*real_tests)))
# real_means = [tm[0]] + tm
# normalization
scaler = StandardScaler()
# df = pd.DataFrame({'nest': nest_without_EES, 'neuron': neuron_without_EES})
# df[['nest', 'neuron']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# normalized_nest = nest_without_EES.reshape(1, -1)
# normalized_neuron = neuron_without_EES.reshape(1, -1)
normalized_nest = scaler.fit_transform(nest_means)
normalized_neuron = scaler.transform(neuron_means)
normalized_real = scaler.transform(real_pairs)
# # plt.plot(range(len(lambda x: (x - x.min()) / (x.max() - x.min()))), df)
# plt.plot(range(len(normalized_nest)), normalized_nest, label="NEST")
# plt.plot(range(len(normalized_neuron)), normalized_neuron, label="NEURON")
# plt.legend()
# plt.show()
# raise Exception
step_size_NEST = int(len(normalized_nest) / sim_time * delta_step_size)
step_size_NEURON = int(len(normalized_neuron) / sim_time * delta_step_size)
step_size_real = int(len(normalized_real) / sim_time * delta_step_size)
# fixme split on chunks (by stepsize)
offset = 0
for elem in range(int(len(normalized_nest) / step_size_NEST)):
	chunks_NEST.append(normalized_nest[offset:offset + step_size_NEST])
	offset += step_size_NEST

offset = 0
for elem in range(int(len(normalized_neuron) / step_size_NEURON)):
	chunks_NEURON.append(normalized_neuron[offset:offset + step_size_NEURON])
	offset += step_size_NEURON

offset = 0
for elem in range(int(len(normalized_real) / step_size_real)):
	chunks_real.append(normalized_real[offset:offset + step_size_real])
	offset += step_size_real
# fixme calculate STD for each chunk for NEST
stds_NEST = list(map(lambda x: np.std(x), chunks_NEST))
# fixme calculate STD for each chunk for NEURON
stds_NEURON = list(map(lambda x: np.std(x), chunks_NEURON))
# fixme calculate STD for each chunk for real data
stds_real = list(map(lambda x: np.std(x), chunks_real))
# fixme calculate delta of STD for each zipped STDs of NEST/NEURON
# for std_for_NEST, std_for_NEURON in zip(stds_NEST, stds_NEURON):
# 	deltas_std.append(std_for_NEST - std_for_NEURON)
for std_for_NEST, std_for_real in zip (stds_NEST, stds_real):
	deltas_std_nest.append(std_for_NEST - std_for_real)
for std_for_NEURON, std_for_real in zip (stds_NEURON, stds_real):
	deltas_std_neuron.append(std_for_NEURON - std_for_real)
plt.figure(figsize=(16, 9))
plt.subplot(211)
# plt.plot([i / 15 for i in range(len(normalized_nest))], normalized_nest)
# plt.plot([i / 15 for i in range(len(normalized_real))], normalized_real)
# plt.legend()
# plt.xlim(0, 100)
# plt.ylabel("uV")

# plt.subplot(212)
plt.title("Step size: {} ms".format(delta_step_size))
plt.ylabel("Δ σ (Δ СКО Nest & real)")
plt.xlim(0, len(chunks_NEST))
plt.plot(range(len(deltas_std_nest)), deltas_std_nest)
plt.subplot(212)
plt.title("Step size: {} ms".format(delta_step_size))
plt.ylabel("Δ σ (Δ СКО Neuron & real)")
plt.xlim(0, len(chunks_NEURON))
plt.plot(range(len(deltas_std_neuron)), deltas_std_neuron)
plt.subplot(212)

# plt.savefig("C:/Users/Home/Desktop/results.png", dpi=300)
plt.show()
