import pylab
import numpy as np

chunks_NEST = []
chunks_NEURON = []
deltas_std = []
# Common parameters
delta_step_size = 1
sim_time = 125
V_to_uV = 1000000  # convert Volts to micro-Volts (V -> uV)

# fixme collect data for NEST
with open('volMN{}v{}.txt'.format(0, 0), 'r') as file:
    NEST = [float(i) * V_to_uV for i in file.read().split("\n")]

# fixme collect data for NEURON
with open('volMN{}v{}.txt'.format(0, 1), 'r') as file:
    NEURON = [float(i) * V_to_uV for i in file.read().split("\n")]

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

pylab.savefig("results.png", dpi=300)
