import os
import sys
import numpy as np
from matplotlib import pylab
from enum import Enum

topologies_path = 'second_level.src.topologies'


class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 1
    RATE = 40
    SIMULATION_TIME = 150.
    INH_COEF = 1.
    PLOT_SLICES_SHIFT = 12.  # ms
frequency_of_discretization = 0.025
duration_of_one_slice = 25

neuron_number = 169
tests_number = 25    # 25
neuron_number = 21
threads = 8
# num_slices = 5
speeds = [25, 50, 125]
frequencies = [20, 40]
# collect data from each test
testsNeuron = {k: {} for k in range(tests_number)}
for neuron_test_number in range(tests_number):
    for speed in speeds:
        if speed == 25:
            sp = '21'
            for thread in range(threads):
                for neuron_id in range(neuron_number):
                    with open(
                            '../neuron-data/res3010/vMN{}r{}s{}v{}'.format(neuron_id, thread, speed,
                                                                                           neuron_test_number),
                            'r') as file:
                        print("opened", 'res3010/vMN{}r{}s{}v{}'.format(neuron_id, thread, speed,
                                                                                           neuron_test_number))
                        value = [float(i) for i in file.read().split("\n")[1:-1]]
                        time_iter = 0.0
                        for offset in range(len(value)):
                            if time_iter not in testsNeuron[neuron_test_number]:
                                testsNeuron[neuron_test_number][time_iter] = 0
                            testsNeuron[neuron_test_number][time_iter] += value[offset]
                            time_iter += 0.1
    for time in testsNeuron[neuron_test_number].keys():
        testsNeuron[neuron_test_number][time] *= 10 ** 13
        testsNeuron[neuron_test_number][time] /= 169
print("len(testsNeuron) = ", len(testsNeuron[0]))
def plot_n_tests():
    """
    Function that draws the plot of shadows by the number of slices: line in each slice is the mean value of each slice by 25 
    runs of Neuron, shadows indicate the scatter between min and max value in each slice 
    Returns
    -------
    Plot
    """
    # test_number = 2
    slices = {}
    yticks = []
    num_slices = int(len(testsNeuron[0]) * frequency_of_discretization / duration_of_one_slice)
    # tests_number = 2

    for test_number, test_data in testsNeuron.items():
        raw_times = sorted(test_data.keys())
        chunk = len(raw_times) / num_slices

        for slice_number in range(num_slices):
            start_time = int(slice_number * chunk)
            end_time = int(slice_number * chunk + chunk)
            if slice_number not in slices.keys():
                slices[slice_number] = dict()
            slices[slice_number][test_number] = [test_data[time] * (-1) for time in raw_times[start_time:end_time]]

    # print(len(slices), "slices = ", slices)
    pylab.figure(figsize=(9, 3))
    pylab.suptitle("Rate {} Hz, inh {}%, speed {} cm/s".format(Params.RATE.value, 100 * Params.INH_COEF.value, sp), fontsize=11)
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

        # print(len(minimal_per_step), "minimal_per_step = ", minimal_per_step)
        # print(len(maximal_per_step), "maximal_per_step = ", maximal_per_step)

        # plot mean with shadows.py
        pylab.plot(times, means, linewidth=1, color='gray')
        pylab.fill_between(times,
                           [i + offset for i in minimal_per_step],  # the minimal values + offset (slice number)
                           [i + offset for i in maximal_per_step],  # the maximal values + offset (slice number)
                           alpha=0.35)
    ticks = []
    for x in range(20,101, 20):
        ticks.append(x)
    labels = []
    for x in range(5,26, 5):
        labels.append(x)
    # global plot settings
    pylab.xlim(0, 100)
    pylab.xticks(ticks, labels)
    pylab.yticks(yticks, range(1, num_slices + 1))
    pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
    pylab.subplots_adjust(left=0.03, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
    pylab.gca().invert_yaxis()
    pylab.show()
    # save the plot
    fig_name = 'slices_mean_{}Hz_{}inh-{}sublevels'.format(Params.RATE.value,
                                                           Params.INH_COEF.value,
                                                           Params.NUM_SUBLEVELS.value)
    pylab.savefig(os.path.join(os.getcwd(), '{}.png'.format(fig_name)), dpi=200)
    pylab.close('all')


if __name__ == "__main__":
    plot_n_tests()