from matplotlib import pylab as plt
from matplotlib import pyplot
# import csv
# import matplotlib.cbook as cbook
# import numpy as np
#from analysis.boxplot_custom_quartiles import my_boxplot_stats
from analysis.functions import read_neuron_data

path = '../bio-data/notiception/atp/atp_half_75002880.txt'
# path_neuron = '../../neuron-data/all2.txt'
# path_spike_times = '../../neuron-data/st2.txt'
path_neuron = '../../neuron-data/5HT_res_newest.hdf5'
neuron_data = read_neuron_data(path_neuron)
# print("neuron_data = ", type(neuron_data), len(neuron_data), len(neuron_data[0]))
# raise Exception
neuron = []
# for i in range(len(neuron_data)):
# 	for j in range(len(neuron_data[i])):
# 		neuron.append(neuron_data[i][j])

print("len(neuron) = ", len(neuron))
# print(neuron[0:100])
# print(neuron[240000:240100])
# neuron_by_1min = []
# offset = 0
# for i in range(5):
# 	neuron_by_1min_tmp = []
# 	for j in range(offset, offset + 2400000):
# 		print("j = ", j)
		# neuron_by_1min_tmp.append(neuron[j])
	# neuron_by_1min.append(neuron_by_1min_tmp)
	# offset += 2400000
# print("neuron_by_1min = ", len(neuron_by_1min))
# for m  in neuron_by_1min:
# 	print("len(m) = ", m[0:100])
# raise Exception
# plt.plot(neuron_data[0])
# plt.show()
with open(path) as f:
	floats = list(map(float, f))
print("len(floats) = ", len(floats))    # 75002880
# raise Exception
# floats = floats[int(len(floats) / 2):]
# plt.plot(floats)
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
# spikes_by_minutes = []
# for minute in neuron_by_1min:
# 	spikes = []
	# print("minute = ", len(minute))
	# for i in range(1, len(minute) - 1):
	# 	if minute[i - 1] < minute[i] > minute[i + 1]:
	# 		spikes.append(i)
	# spikes_by_minutes.append(spikes)
# print("spikes = ", spikes[-10:])
# plt.plot(spikes, marker='.', markercolor='red')
# plt.show()
# spikes_in_minute = []
# offset = 0
# for i in range(len(spikes)):
# 	spikes_in_minute_tmp = []
# 	for j in range(offset, offset + 2400000):
# 		spikes_in_minute_tmp.append(spikes[j])
# 	offset += 2400000
# 	spikes_in_minute.append(spikes_in_minute_tmp)

# for sp in spikes_in_minute:
# 	print("len(sp) = ", len(sp))
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
	if neuron[f - 1] < neuron[f] > neuron[f + 1]:
		spikes.append(f)

spikes_bio = []
for f in range(1, len(floats) - 1):
	if floats[f - 1] < floats[f] > floats[f + 1] and floats[f] > 0.35:
		print("floats[{}] = ".format(f), floats[f])
		spikes_bio.append(f)

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

spikes = [s / 40 for s in spikes]
spikes_bio = [s / 125 for s in spikes_bio]
print("spikes = ", spikes)

# for minu in spikes_by_minutes:
# 	print("minu = ", minu[0:100])
# 	pyplot.vlines(minu, ymin=0, ymax=1)
# 	pyplot.ylim(0, 2)
# 	pyplot.title("Spike times neuron 5HT")
# 	plt.show()
# spikes_neuron = [s / 40 for s in spikes_neuron]

pyplot.vlines(spikes_bio, ymin=0, ymax=1)
pyplot.ylim(0, 2)
pyplot.title("Spike times bio ATP")
plt.show()

intervals = []
for s in range(1, len(spikes)):
	if spikes[s] - spikes[s - 1] < 1000:
		intervals.append(spikes[s] - spikes[s - 1])
print("intervals = ", len(intervals))

intervals_bio = []
for s in range(1, len(spikes_bio)):
	if spikes_bio[s] - spikes_bio[s - 1] < 1000:
		intervals_bio.append(spikes_bio[s] - spikes_bio[s - 1])
# print("intervals_bio = ", intervals_bio)
# plt.plot(intervals_bio, color='blue', marker='.', markersize=10, markerfacecolor='red')
# plt.show()
# plt.plot(spikes, neuron_data[0][spikes], marker='.', color='red')
# plt.show()
start = 0
finish = start + 1000
count_array = []
while finish < len(neuron_data[0]):
	print("finish = ", finish)
	count = 0
	for s in spikes:
		if s > start and s < finish:
			count += 1
	# 		print("count = ", count)
	start += 1000
	finish = start + 1000

	if count > 0 and count < 2000:
		count_array.append(count)
print("count_array = ", len(count_array), count_array)
start = 0
finish = start + 1000
count_array_bio = []
while finish < len(floats): # 37 501 440
	print("finish = ", finish)
	count = 0
	for s in spikes_bio:
		if s > start and s < finish:
			count += 1
	# 		print("count = ", count)
	start += 1000
	finish = start + 1000

	if count > 0:
		count_array_bio.append(count)
print("count_array_bio = ", len(count_array_bio), count_array_bio)
print("max(count_array) = ", max(count_array))
# final_count_array = []
# for c in range(1, len(count_array) - 1):
# 	if count_array[c] != 0 and count_array[c] != count_array[c - 1]:
# 		final_count_array.append(count_array[c])
# print("final_count_array = ", len(final_count_array), final_count_array)
# print("len(spikes) = ", len(spikes))
# pyplot.boxplot(intervals)#, count_array)
# pyplot.show()
# stats = {}
# stats['interspikes intervals'] = my_boxplot_stats(intervals,
#                                                   percents=[10, 99.9])
# stats['number of spikes'] = my_boxplot_stats(count_array)

# fig, ax = pyplot.subplots(1, 1)
# ax.bxp([#stats['interspikes intervals'][0],
# 	 stats['number of spikes'][0]])
# stats = cbook.boxplot_stats(intervals, )
names = ['neuron intervals', 'number of neuron spikes', 'bio intervals', 'number of bio spikes']
data = [intervals, count_array, intervals_bio, count_array_bio]
# pyplot.boxplot(data)
fig, ax = pyplot.subplots()
ax.boxplot(data)
ax.set_xticklabels(names)
pyplot.show()
