from matplotlib import pylab as plt
from matplotlib import pyplot
# import csv
# import matplotlib.cbook as cbook
# import numpy as np
#from analysis.boxplot_custom_quartiles import my_boxplot_stats
from analysis.functions import read_neuron_data
import numpy as np

path = '../bio-data/notiception/atp/atp_75002880.txt'
path_neuron = '/home/anna/PycharmProjects/LAB/neuron-data/resATP3.hdf5'
# path_spike_times = '../../neuron-data/st2.txt'
# path_neuron = '../../neuron-data/resATP.hdf5'
# additional_path_neuron = '../../neuron-data/resATP.hdf5'
neuron_data = read_neuron_data(path_neuron)
# additional_neuron_data = read_neuron_data(additional_path_neuron)
# print("neuron_data = ", type(neuron_data), len(neuron_data), len(neuron_data[0]))
# print("additional_neuron_data = ", type(additional_neuron_data), len(additional_neuron_data),
#       len(additional_neuron_data[0]))
# raise Exception
neuron = []
for i in range(len(neuron_data)):
	for j in range(len(neuron_data[i])):
		neuron.append(neuron_data[i][j])
# plt.plot(neuron, color='red', label='neuron')
# print("neuron = ", neuron[:100])

# additional_neuron = []
# for i in range(len(additional_neuron_data)):
# 	for j in range(len(additional_neuron_data[i])):
# 		additional_neuron.append(additional_neuron_data[i][j])

# print("len(neuron) = ", len(neuron))
# raise Exception

# print("len(additional_neuron) =  ", len(additional_neuron))

# for n in range(len(neuron)):
# 	neuron[n] += additional_neuron[n]
# print("neuron = ", neuron[:100])

# plt.plot(additional_neuron, color='green', label='additional_neuron')
# plt.plot(neuron)
# plt.show()
# print(neuron[0:100])
# print(neuron[240000:240100])
neuron_by_1min = []
offset = 0
for i in range(5):
	neuron_by_1min_tmp = []
	for j in range(offset, offset + 2400000):
		# print("j = ", j)
		neuron_by_1min_tmp.append(neuron[j])
	neuron_by_1min.append(neuron_by_1min_tmp)
	offset += 2400000

# plt.plot(neuron_by_1min[0])
# plt.title('neuron')
# plt.show()

# print("neuron_by_1min = ", len(neuron_by_1min))
# for m  in neuron_by_1min:
	# plt.plot(m)
	# plt.title('neuron')
	# plt.show()
# 	print("len(m) = ", m[0:100])
# raise Exception
# plt.show()
with open(path) as f:
	floats = list(map(float, f))
# print("len(floats) = ", len(floats))    # 75002880
# raise Exception
# floats = floats[int(len(floats) / 2):]
# plt.plot(floats)
# plt.title('bio')
# plt.ylim(-0.68, 0.95)
# plt.show()
# print("len(floats) = ", len(floats))

# with open(path_neuron) as f:
# 	neuron = list(map(float, f))

# with open(path_spike_times) as f:
# 	spike_times = sorted(list(map(float, f)))
# print("spike_times = ", spike_times)
# spike_times_dots = [int(s * 40) for s in spike_times]
# print("spike_times_dots = ", spike_times_dots)
spikes_by_minutes = []
for minute in neuron_by_1min:
	spikes = []
	# print("minute = ", len(minute))
	for i in range(1, len(minute) - 1):
		if minute[i - 1] < minute[i] > minute[i + 1] and minute[i] > 0.015:
			spikes.append(i)
	spikes_by_minutes.append(spikes)
# print("spikes_by_minutes[-1] = ", spikes_by_minutes[-1])

# pyplot.vlines(spikes_by_minutes[-1], ymin=0, ymax=1)
# pyplot.ylim(0, 2)
# pyplot.title("Spike times neuron ATP")
# plt.show()

# for m in neuron_by_1min:
# 	spikes_in_minute = []
# 	for i in range(1, len(m) - 1):
# 		if m[i - 1] < m[i] > m[i + 1]:
# 			spikes_in_minute.append(i)
# 	spikes.append(spikes_in_minute)
# print("len(spikes) = ", len(spikes))
# for spike in spikes:
# 	print("len(spike)= ", len(spike))
# spikes_by_minutes.append(spikes)
# print("len(spikes_by_minutes) = ", len(spikes_by_minutes))
# for minute in spikes_by_minutes:
# 	print("len(minute) = ", len(minute))
# raise Exception
# print("spikes = ", spikes)

spikes = []
for f in range(1, len(neuron) - 1):
	if neuron[f - 1] < neuron[f] > neuron[f + 1] and neuron[f] > 0.015:
		spikes.append(f)
# plt.plot(neuron)
# for s in spikes:
# 	plt.plot(s, neuron[s], marker='.', markerfacecolor='red')
# plt.show()

spikes_bio = [0]
for f in range(1, len(floats) - 1):
	if floats[f - 1] < floats[f] > floats[f + 1] and floats[f] > 0.3:
		if f - spikes_bio[-1] > 60:
		# print("floats[{}] = ".format(f), floats[f])
			spikes_bio.append(f)
del spikes_bio[0]
# for s in spikes_bio:
# 	plt.plot(s, floats[s], marker='.', color='red')
# plt.show()
# pyplot.vlines(spikes_bio, ymin=0, ymax=1)
# pyplot.ylim(0, 2)
# pyplot.title("Spike times bio ATP")
# plt.show()
# print("spikes_bio = ", spikes_bio)
# spikes_neuron = []
# for f in range(1, len(neuron) - 1):
# 	if neuron[f - 1] < neuron[f] > neuron[f + 1]:
# 		spikes_neuron.append(f)

# plt.plot(neuron)
# for i in spike_times_dots:
# 	plt.plot(i, neuron[i], marker='.', color='red', label=i)
# plt.show()

# print(len(floats))
# print("spikes = ", spikes)
# plt.plot(floats)
# for i in spikes_bio:
# 	plt.plot(i, floats[i], marker='.', color='red')
# plt.show()
print("spikes = ", spikes)
print("spikes_bio = ", spikes_bio)

spikes = [s / 40 for s in spikes]
spikes_bio = [s / 125 for s in spikes_bio]
print("spikes = ", spikes)
print("spikes_bio = ", spikes_bio)

# for minu in spikes_by_minutes:
	# minu = [s / 40 for s in spikes]
	# print("minu = ", minu)
	# print("len(minu) = ", len(minu))
	# del minu[-2:-1]
	# print("minu = ", len(minu))
	# pyplot.vlines(minu, ymin=0, ymax=1)
	# pyplot.ylim(0, 2)
	# pyplot.title("Spike times neuron ATP")
	# plt.show()
# spikes_neuron = [s / 40 for s in spikes_neuron]

# pyplot.vlines(spikes, ymin=0, ymax=1)
# pyplot.ylim(0, 2)
# pyplot.title("Spike times neuron ATP")
# plt.show()

pyplot.vlines(spikes_bio, ymin=0, ymax=1)
pyplot.ylim(0, 2)
pyplot.title("Spike times bio ATP")
plt.show()

intervals = []
for s in range(1, len(spikes)):
	if spikes[s] - spikes[s - 1]:
		intervals.append(spikes[s] - spikes[s - 1])
print("intervals = ", len(intervals))

intervals_bio = []
for s in range(1, len(spikes_bio)):
	if spikes_bio[s] - spikes_bio[s - 1]:
		intervals_bio.append(spikes_bio[s] - spikes_bio[s - 1])
# plt.plot(intervals_bio, color='blue', marker='.', markersize=10, markerfacecolor='red')
# plt.show()
# plt.plot(spikes, neuron_data[0][spikes], marker='.', color='red')
# plt.show()
"""start = 0
finish = start + 1000
count_array = []
while finish < len(neuron_data[0]):
	count = 0
	for s in spikes:
		if s > start and s < finish:
			count += 1
	start += 1000
	finish = start + 1000

	if count > 0 and count < 2000:
		count_array.append(count)
start = 0
finish = start + 1000
count_array_bio = []
while finish < len(floats): # 37 501 440
	count = 0
	for s in spikes_bio:
		if s > start and s < finish:
			count += 1
	start += 1000
	finish = start + 1000

	if count > 0:
		count_array_bio.append(count)
"""
# final_count_array = []
# for c in range(1, len(count_array) - 1):
# 	if count_array[c] != 0 and count_array[c] != count_array[c - 1]:
# 		final_count_array.append(count_array[c])
names = ['bio intervals', 'neuron intervals']
data = [intervals_bio, intervals]
median_neuron = np.median(intervals)
median_bio = np.median(intervals_bio)
print("median_neuron = ", median_neuron)
print("median_bio = ", median_bio)
fig, ax = pyplot.subplots()
ax.boxplot(data)
ax.set_xticklabels(names)
pyplot.show()
