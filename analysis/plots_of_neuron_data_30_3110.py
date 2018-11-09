import pylab as plt
import numpy as np
threads = 8
test_numbers = 25
speeds = [25, 50, 125]
frequencies = [20, 40]
neuron_number = 21
tmp = []
V_to_uV = 1000000  # convert Volts to micro-Volts (V -> uV)
neuron_tests_s125 = []
for neuron_test_number in range(2):
    for thread in range(threads):
        for neuron_id in range(neuron_number):
            for speed in speeds:
                if speed == 125:
                    with open('../neuron-data/res3110/vMN{}r{}s{}v{}'.format(neuron_id, thread,
                                                                                                     speed,
                                                                                                     neuron_test_number),
                                      'r') as file:  # change to res3010 in case of s25 and s50

                        print("opened", 'res3110/volMN{}r{}s{}v{}.txt'.format(neuron_id, thread, speed,
                                                                                                     neuron_test_number))
                        tmp.append([float(i) * V_to_uV for i in file.read().split("\n")[1:-1]])
                    neuron_tests_s125.append([elem * 10 ** 4 for elem in list(map(lambda x: np.mean(x), zip(*tmp)))])
                    tmp.clear()
neuron_means_s125 = list(map(lambda x: np.mean(x), zip(*neuron_tests_s125)))


def find_mins(array):
    min_elems = []
    indexes = []
    for index_elem in range(1, len(array) - 1):
        if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]) and array[index_elem] < -14:
            min_elems.append(array[index_elem])
            indexes.append(index_elem)
    return min_elems, indexes


# ticks = []
# for i in range(len(neuron_means_s125)):
#     if i % 360 == 0:
#         ticks.append(i)
EES_values = find_mins(neuron_means_s125)[0]
EES_indexes = find_mins(neuron_means_s125)[1]
plt.plot(neuron_means_s125)
for index in EES_indexes:
    plt.axvline(x=index, linestyle="--", color="gray")
# for index in ticks:
#     plt.axvline(x=index, linestyle=":", color="red")
# plt.xticks(ticks, [i // 40 for i in ticks])

plt.show()