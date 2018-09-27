import pylab as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
chunks_NEST = []
chunks_NEURON = []
deltas_std = []
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

print(len(nest_without_EES), nest_without_EES)

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

print(len(neuron_without_EES), neuron_without_EES)

# normalization
scaler = StandardScaler()
#
# print("max_nest: ", max(nest_without_EES))
# print("min_nest: ", min(nest_without_EES))
#
# print("max_neuron: ", max(neuron_without_EES))
# print("min_neuron: ", min(neuron_without_EES))
# df = pd.DataFrame({'nest': nest_without_EES, 'neuron': neuron_without_EES})
# df[['nest', 'neuron']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# normalized_nest = nest_without_EES.reshape(1, -1)
# normalized_neuron = neuron_without_EES.reshape(1, -1)
normalized_nest = scaler.fit_transform(nest_without_EES)
normalized_neuron = scaler.transform(neuron_without_EES)
print("normalized_nest: ", normalized_nest)
print("normalized_neuron: ", normalized_neuron)
# print("df: ", df)
# plt.plot(range(len(lambda x: (x - x.min()) / (x.max() - x.min()))), df)
plt.plot(range(len(normalized_nest)), normalized_nest, label="NEST")
plt.plot(range(len(normalized_neuron)), normalized_neuron, label="NEURON")
plt.legend()
plt.show()


"""
raise Exception
step_size_NEST = int(len(NEST) / sim_time * delta_step_size)
step_size_NEURON = int(len(NEURON) / sim_time * delta_step_size)

# fixme split on chunks (by stepsize)
offset = 0
for elem in range(int(len(NEST) / step_size_NEST)):
	chunks_NEST.append(NEST[offset:offset + step_size_NEST])
	offset += step_size_NEST

offset = 0
for elem in range(int(len(NEURON) / step_size_NEURON)):
	chunks_NEURON.append(NEURON[offset:offset + step_size_NEURON])
	offset += step_size_NEURON

# fixme calculate STD for each chunk for NEST
stds_NEST = list(map(lambda x: np.std(x), chunks_NEST))
# fixme calculate STD for each chunk for NEURON
stds_NEURON = list(map(lambda x: np.std(x), chunks_NEURON))

# fixme calculate delta of STD for each zipped STDs of NEST/NEURON
for std_for_NEST, std_for_NEURON in zip(stds_NEST, stds_NEURON):
	deltas_std.append(std_for_NEST - std_for_NEURON)

pylab.figure(figsize=(16, 9))
pylab.subplot(211)
pylab.plot([i / 40 for i in range(len(NEST))], NEST)
pylab.plot([i / 40 for i in range(len(NEURON))], NEURON)
pylab.xlim(0, 125)
pylab.ylabel("uV")

pylab.subplot(212)
pylab.title("Step size: {} ms".format(delta_step_size))
pylab.ylabel("Δ σ (Δ СКО)")
pylab.xlim(0, len(chunks_NEST))
pylab.plot(range(len(deltas_std)), deltas_std)

pylab.savefig("C:/Users/Home/Desktop/results.png", dpi=300)
pylab.show()
"""
