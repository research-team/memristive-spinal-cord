import numpy as np
import pylab
test_number = 25
neuron_number = 21
threads = 8
num_slices = 5
speeds = [25, 50, 125]
frequencies = [20, 40]
slices = {}
yticks = []

testsNeuron = {k: {} for k in range(test_number)}
for neuron_test_number in range(test_number):
    for thread in range(threads):
        for neuron_id in range(neuron_number):
            # for speed in speeds:
            #     if speed == 125: # change to 25 / 50
            for frecuency in frequencies:
                if frecuency == 20:
                    with open('C:/Users/Home/Desktop/Нейролаб/res211/vMN{}r{}fr{}v{}'.format(neuron_id, thread,
                                                                                             frecuency,
                                                                                             neuron_test_number), 'r')\
                            as file:   # change to res3010 in case of s25 and s50
                        value = [float(i) for i in file.read().split("\n")[1:-1]]
                        # print(value)
                        time_iter = 0.0
                        for offset in range(0, len(value)):
                            if time_iter not in testsNeuron[neuron_test_number]:
                                testsNeuron[neuron_test_number][time_iter] = 0
                            testsNeuron[neuron_test_number][time_iter] += value[offset]
                            time_iter += 0.1
            # print(testsNeuron)

for time in testsNeuron[neuron_test_number].keys():
    testsNeuron[neuron_test_number][time] *= 10 ** 13
    testsNeuron[neuron_test_number][time] /= 169
print(testsNeuron)
# every slice keep 10 tests, each test contain voltage results
for test_number, test_data in testsNeuron.items():
    raw_times = sorted(test_data.keys())
    chunk = len(raw_times) / num_slices
    print("chunk = ", chunk)
    for slice_number in range(num_slices):
        start_time = int(slice_number * chunk)
        end_time = int(slice_number * chunk + chunk)
        if slice_number not in slices.keys():
            slices[slice_number] = dict()
        slices[slice_number][test_number] = [test_data[time] for time in raw_times[start_time:end_time]]
print("slices = ", slices)
pylab.figure(figsize=(9, 3))
pylab.suptitle("EES = 20 Hz")   # for s25 spped = 21 cm/s; for s50 speed = 13cm/s
# plot each slice
for slice_number, tests in slices.items():
    offset = slice_number * 10000
    yticks.append(tests[list(tests.keys())[0]][0] + offset)
    # collect mean data: sum values (mV) step by step (ms) per test and divide by test number
    mean_data = list(map(lambda elements: np.mean(elements), zip(*tests.values())))  #
    # collect values
    times = [time / 10 for time in range(len(mean_data))]  # divide by 10 to convert to ms step
    means = [voltage + offset for voltage in mean_data]
    minimal_per_step = [min(a) for a in zip(*tests.values())]
    maximal_per_step = [max(a) for a in zip(*tests.values())]

    # plot mean with shadows.py
    pylab.plot(times, means, linewidth=1, color='gray')
    pylab.fill_between(times,
                                       [i + offset for i in minimal_per_step],
                                       # the minimal values + offset (slice number)
                                       [i + offset for i in maximal_per_step],
                                       # the maximal values + offset (slice number)
                                       alpha=0.35)
# global plot settings
# pylab.xlim(0, 25)
pylab.xticks(range(250), [i if i % 10 == 0 else "" for i in range(250)])
pylab.yticks(yticks, range(1, num_slices + 1))
pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
pylab.subplots_adjust(left=0.03, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
pylab.gca().invert_yaxis()
pylab.show()
