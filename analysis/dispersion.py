import scipy.io as sio
import numpy as np
import pylab as plt
# from analysis.real_data_slices import read_data, slice_myogram
real_data_step = 0.25
def read_data(file_path):
	global tickrate
	global title
	mat_data = sio.loadmat(file_path)
	tickrate = int(mat_data['tickrate'][0][0])
	title = mat_data['titles'][0]
	return mat_data


def slice_myogram(raw_data, slicing_index ='Stim'):
	"""
	The function to slice the data from the matlab file of myogram.
	:param dict raw_data:  the myogram data loaded from matlab file.
	:param str slicing_index: the index to be used as the base for slicing, default 'Stim'.
	:return: list volt_data: the voltages array
	:return: list slices_begin_time: the staring time of each slice array.
	"""
	# Collect data
	volt_data = []
	stim_data = []
	slices_begin_time = []

	# data processing
	for index, data_title in enumerate(raw_data['titles']):
		data_start = int(raw_data['datastart'][index]) - 1
		data_end = int(raw_data['dataend'][index])
		float_data = [round(float(x), 3) for x in raw_data['data'][0][data_start:data_end]]
		if slicing_index not in data_title:
			volt_data = float_data
		else:
			stim_data = float_data

	# find peaks in stimulations data
	for index in range(1, len(stim_data) - 1):
		if stim_data[index - 1] < stim_data[index] > stim_data[index + 1] and stim_data[index] > 4:
			slices_begin_time.append(index * real_data_step)  # division by 4 gives us the normal 1 ms step size

	# remove unnecessary data, use only from first stim, and last stim
	volt_data = volt_data[int(slices_begin_time[0] / real_data_step):int(slices_begin_time[-1] / real_data_step)]

	# move times to the begin (start from 0 ms)
	slices_begin_time = [t - slices_begin_time[0] for t in slices_begin_time]

	return volt_data, slices_begin_time




# read real data
path = "SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat"
data = read_data('../bio-data/{}'.format(path))
processed_data = slice_myogram(data)
real_data = processed_data[0]
slices_begin_time = processed_data[1]
mat_data = sio.loadmat('../bio-data/{}'.format(path))
tickrate = int(mat_data['tickrate'][0][0])
# for i in range(slices_begin_time[0]:slices_begin_time[-1]):

print("len(real_data) = ", len(real_data))  # 2700
# normalization
oldmin = min(real_data)
oldmax = max(real_data)
oldrange = oldmax - oldmin
newmin = 0.
newmax = 1.
newrange = newmax - newmin
if oldrange == 0:
	if oldmin < newmin:
		newval = newmin
	elif oldmin > newmax:
		newval = newmax
	else:
		newval = oldmin
	normal = [newval for v in real_data]
else:
	scale = newrange / oldrange
	normal_real = [(v - oldmin) * scale + newmin for v in real_data]
print("normal real = ", normal_real)
print("len(normal real) = ", len(normal_real))
# print("real_data = ", real_data)
# plot real data
# plt.plot(real_data)
# x = [i / tickrate * 1000 for i in range(len(real_data))]
# plt.plot(x, real_data, label=title)
# plt.plot(slices_begin_time, [0 for _ in slices_begin_time], ".", color='r')

# for kk in slices_begin_time:
# 	plt.axvline(x=kk, linestyle="--", color="gray")
# plt.xticks(np.arange(0, slices_begin_time[-1] + 1, 25), np.arange(0, slices_begin_time[-1] + 1, 25))

# plt.show()
# read neuron data
threads = 8
test_numbers = 25
speeds = [25, 50, 125]
neuron_number = 21
tmp = []
V_to_uV = 1000000  # convert Volts to micro-Volts (V -> uV)
neuron_tests_s125 = []
for thread in range(threads):
    for test_number in range(2):
        for speed in speeds:
            for neuron_id in range(neuron_number):
                if speed == 125:
                    with open('../neuron-data/res3110/vMN{}r{}s{}v{}'.format(neuron_id, thread, speed, test_number), 'r') as file:
                        print("opened", 'res3110/volMN{}r{}s{}v{}.txt'.format(neuron_id, thread, speed, test_number))
                        tmp.append([float(i) * V_to_uV for i in file.read().split("\n")[1:-1]])
                    neuron_tests_s125.append([elem * 10 ** 4 for elem in list(map(lambda x: np.mean(x), zip(*tmp)))])
                    del tmp[:]
print("len(neuron_tests_s125) = ", len(neuron_tests_s125))  # 4200
print("len(neuron_tests_s125[0]) = ", len(neuron_tests_s125[0]))

neuron_means_s125 = list(map(lambda x: np.mean(x), zip(*neuron_tests_s125)))
print("len(neuron_means_s125) = ", len(neuron_means_s125))
# print("neuron_means_s125 = ", neuron_means_s125)
# normalization
oldmin = min(neuron_means_s125)
oldmax = max(neuron_means_s125)
oldrange = oldmax - oldmin
newmin = 0.
newmax = 1.
newrange = newmax - newmin
if oldrange == 0:
	if oldmin < newmin:
		newval = newmin
	elif oldmin > newmax:
		newval = newmax
	else:
		newval = oldmin
	normal = [newval for v in neuron_means_s125]
else:
	scale = newrange / oldrange
	normal_neuron = [(v - oldmin) * scale + newmin for v in neuron_means_s125]
# plt.plot(range(len(neuron_means_s125)), normal_neuron)
# print("normal = ", normal)
neuron_dots_arrays = []
tmp = []
offset = 0
for iter_begin in range(300):
	for chunk_len in range(offset, offset + 12):
		tmp.append(normal_neuron[chunk_len])
	# print("iter_begin1 = ", iter_begin)
	# print("len(tmp) in cycle1 = ", len(tmp))
	copy = list(tmp)
	neuron_dots_arrays.append(copy)
	del tmp[:]
	offset += 12
	# print("len(neuron_dots_arrays) in cycle = ", len(neuron_dots_arrays))
offset = 0
for iter_begin in range(2400):
	for chunk_len in range(offset, offset + 11):
		tmp.append(normal_neuron[chunk_len])
	# print("iter_begin2 = ", iter_begin)
	# print("len(tmp) in cycle2 = ", len(tmp))
	copy = list(tmp)
	neuron_dots_arrays.append(copy)
	del tmp[:]
	offset += 11
print("len(neuron_dots_arrays) = ", len(neuron_dots_arrays))
# print("neuron_dots_arrays2 = '\n'", neuron_dots_arrays)
print("len(neuron_dots_arrays[0]) = ", len(neuron_dots_arrays[0]))
print("len(neuron_dots_arrays[-1]) = ", len(neuron_dots_arrays[-1]))
maxes = []
mins = []
for inner_list in neuron_dots_arrays:
	maxes.append(max(inner_list))
	mins.append(min(inner_list))
print("len(maxes) = ", len(maxes))
# print("maxes = ", maxes[0])
# print("maxes = ", maxes[15])
# print("maxes = ", maxes[30])
# print("maxes = ", maxes[100])
# print("maxes = ", maxes[150])
# print("maxes = ", maxes[300])
# print("maxes = ", maxes[492])
# print("maxes = ", maxes[516])
# print("maxes = ", maxes[720])
# print("maxes = ", maxes[814])
print("len(mins) = ", len(mins))
# print("mins = ", mins[0])
# print("mins = ", mins[15])
# print("mins = ", mins[30])
# print("mins = ", mins[100])
# print("mins = ", mins[150])
# print("mins = ", mins[300])
# print("mins = ", mins[492])
# print("mins = ", mins[516])
# print("mins = ", mins[720])
# print("mins = ", mins[814])
plt.plot(normal_real, linewidth=1, color='gray', label="real")
# plt.plot(maxes, label="maxes")
# plt.plot(mins, label="mins")
plt.fill_between(normal_real, mins, maxes,  alpha=0.35)
plt.legend()
plt.show()