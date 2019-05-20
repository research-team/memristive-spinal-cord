import numpy as np
import math
import pylab as plt
from sklearn.preprocessing import StandardScaler
from analysis.real_data_slices import read_data, trim_myogram
threads = 8
test_numbers = 25
speeds = [25, 50, 125]
neuron_number = 21
tmp = []
V_to_uV = 1000000  # convert Volts to micro-Volts (V -> uV)
neuron_tests_s125 = []
for thread in range(threads):
    for test_number in range(test_numbers):
        for speed in speeds:
            for neuron_id in range(neuron_number):
                if speed == 125:
                    with open('C:/Users/Home/Desktop/Нейролаб/res3110/vMN{}r{}s{}v{}'.format(neuron_id, thread, speed, test_number), 'r') as file:
                        print("opened", 'res3110/volMN{}r{}s{}v{}.txt'.format(neuron_id, thread, speed, test_number))
                        tmp.append([float(i) * V_to_uV for i in file.read().split("\n")[1:-1]])
                    neuron_tests_s125.append([elem * 10 ** 4 for elem in list(map(lambda x: np.mean(x), zip(*tmp)))])
                    tmp.clear()
print("len(neuron_tests_s125) = ", len(neuron_tests_s125))  # 4200
print("len(neuron_tests_s125[0]) = ", len(neuron_tests_s125[0]))
# print("neuron_tests_s125 = ", neuron_tests_s125)
neuron_means_s125 = list(map(lambda x: np.mean(x), zip(*neuron_tests_s125)))
print("len(neuron_means_s125) = ", len(neuron_means_s125))
print("neuron_means_s125 = ", neuron_means_s125)
neuron_pairs_s125 = []
for iter_begin in range(len(neuron_means_s125) - 26400)[::12]:
    neuron_pairs_s125.append(np.mean(neuron_means_s125[iter_begin:iter_begin + 12]))
print("len(neuron_pairs_s125)(middle) = ", len(neuron_pairs_s125))
for iter_begin in range(len(neuron_means_s125) - 26400, len(neuron_means_s125))[::11]:
    neuron_pairs_s125.append(np.mean(neuron_means_s125[iter_begin:iter_begin + 11]))
print("len(neuron_pairs_s125) = ", len(neuron_pairs_s125))

path = "SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat"
data = read_data('C:/Users/Home/LAB/memristive-spinal-cord/bio-data/{}'.format(path))
processed_data = trim_myogram(data)
real_data = processed_data[0]
print("len(real_data) = ", len(real_data))  # 2700
print("real_data = ", real_data)
# scaler = StandardScaler()
# normalized_neuron_pairs_s125 = neuron_pairs_s125.reshape(1, -1)
# normalized_real = real_data.reshape(1, -1)
#
# normalized_neuron_pairs_s125 = scaler.fit_transform(neuron_pairs_s125)
# normalized_real = scaler.transform(real_data)
normalized_neuron_pairs_s125 = [float(i) / sum(neuron_pairs_s125) for i in neuron_pairs_s125]
normalized_real = [float(i) / sum(real_data) for i in real_data]
summ = 0
difference = []
# for i in range(len(real_data)):
#     summ += abs((normalized_real[i] - normalized_neuron_pairs_s125[i]) ** 2)
# standart_deviation = math.sqrt((1/len(real_data)) * summ)
# print("standart_deviation = ", standart_deviation)
for i in range(len(real_data)):
    difference.append(abs(normalized_real[i] - normalized_neuron_pairs_s125[i]))
plt.plot(difference)
plt.title("Difference in values between real data 40 Hz 100% inh 6 cm/s RMG and neuron data with the same parameters")
plt.show()