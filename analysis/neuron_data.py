from analysis.functions import read_neuron_data, find_ees_indexes, normalization
from matplotlib import pylab as plt

data = read_neuron_data('../../neuron-data/3steps_speed15_EX.hdf5')
# print(type(data))
# print(len(data))
# print(len(data[0]), data[0])
# print(data[1])
minimum = []
maximum = []

for d in data:
	if(len(d) > 0):
		minimum.append(min(d))
		maximum.append(max(d))
	# print("d = ", d)

data_list = []
for j in data:
	data_list_tmp = []
	for i in j:
		data_list_tmp.append(i)

	data_list.append(data_list_tmp)
all_neuron_slices = []
for k in range(len(data_list)):
	neuron_slices = []
	offset = 0
	for  i in range(int(len(data_list[k]) / 1000)):
		neuron_slices_tmp = []
		for j in range(offset, offset + 1000):
			neuron_slices_tmp.append(data_list[k][j])
		neuron_slices.append(neuron_slices_tmp)
		offset += 1000
	all_neuron_slices.append(neuron_slices)
	# print("all_neuron_slices = ", len(all_neuron_slices))
# print('---')
# print("all_neuron_slices = ", len(all_neuron_slices))
# print("all_neuron_slices = ", len(all_neuron_slices[0]))
# print("all_neuron_slices = ", len(all_neuron_slices[0][0])) # [0][17][1000]
all_neuron_slices = list(zip(*all_neuron_slices))
# print("all_neuron_slices = ", all_neuron_slices)

# for dat in data_list:
# 	print("dat = ", dat)
# print("len(data_list) = ", len(data_list))
# print(len(data_list[0]))
# minimum = min(data_list)
# maximum = max(data_list)
# print("minimum = ", minimum)
# print("maximum = ", maximum)
# print(data_list)

stimulations = []
for i in range(0, len(data_list) + 1, 1000):
	stimulations.append(i)
# plt.plot(data_list)
	# for s in stimulations:
	# 	plt.axvline(x=s, linestyle='--', color='gray')
# plt.xlim(0, len(data_list))
# plt.show()

slices = []


def neuron_20_runs():
	return data_list


def neuron_data():
	offset = 0
	for sl in range(int(len(data_list) / 1000)):
		slices_tmp= []
		for j in range(offset, offset + 1000):
			slices_tmp.append(data_list[j])
		slices.append(slices_tmp)
		offset += 1000
	# print("len(slices) = ", len(slices))
	return slices, minimum, maximum

neuron = neuron_data()
slices= neuron[0]
# print(len(slices))
# print(len(slices[0]))
yticks = []
times = []
step = 0.025
for index, sl in enumerate(slices):
	offset = index * 17
	yticks.append(sl[0] + offset)
	times = [time * step for time in range(len(sl))]
	# plt.plot(times, [s + offset for s in sl])
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)])
# plt.yticks(yticks, range(1, len(slices) + 1))
# plt.xlim(0, 25)
# plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
# plt.show()