import numpy as np

test_number = 25
neuron_number = 169

tests_nest = {k: {} for k in range(test_number)}
for nest_test_number in range(test_number):
    nrns_nest = set()
    with open('dat/{}.dat'.format(nest_test_number), 'r') as file:
        for line in file:
            nrn_id, time, volt = line.split("\t")[:3]
            time = float(time)
            correct = 8.0 <= time <= 174.9
            if time not in tests_nest[nest_test_number].keys() and correct:
                tests_nest[nest_test_number][time] = 0
            if correct:
                tests_nest[nest_test_number][time] += float(volt)
                nrns_nest.add(nrn_id)
    for time in tests_nest[nest_test_number].keys():
        tests_nest[nest_test_number][time] /= len(nrns_nest)
print(tests_nest)

testsNeuron = {k: {} for k in range(test_number)}
for neuron_test_number in range(test_number):
    for neuron_id in range(neuron_number):
        with open('res1309/volMN{}v{}.txt'.format(neuron_id, neuron_test_number), 'r') as file:
            value = [float(i) for i in file.read().split("\n")[1:-2]]
            time_iter = 0.0
            for offset in range(0, len(value), 4):
                if time_iter not in testsNeuron[neuron_test_number]:
                    testsNeuron[neuron_test_number][time_iter] = 0
                testsNeuron[neuron_test_number][time_iter] += np.mean(value[offset:offset + 4])
                time_iter += 0.1
    for time in testsNeuron[neuron_test_number].keys():
        testsNeuron[neuron_test_number][time] /= 169

print(testsNeuron)

