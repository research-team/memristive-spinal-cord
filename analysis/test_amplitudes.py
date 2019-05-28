from analysis.patterns_in_bio_data import bio_data_runs
from matplotlib import pylab as plt
from analysis.functions import normalization, calc_max_min, calc_amplitudes, find_latencies

sim_step = 0.025
bio_step = 0.25
offset = 0
all_bio_slices = []

bio_data = bio_data_runs()

for k in range(len(bio_data)):
	bio_slices = []
	offset = 0
	for i in range(int(len(bio_data[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio_data[k][j])
		bio_slices.append(bio_slices_tmp)
		offset += 100
	all_bio_slices.append(bio_slices)   # list [4][16][100]
print("all_bio_slices = ", all_bio_slices)
all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]

instant_mean = []
for slice in range(len(all_bio_slices)):
	instant_mean_sum = []
	for dot in range(len(all_bio_slices[slice][0])):
		instant_mean_tmp = []
		for run in range(len(all_bio_slices[slice])):
			instant_mean_tmp.append(abs(all_bio_slices[slice][run][dot]))
		instant_mean_sum.append(sum(instant_mean_tmp))
	instant_mean.append(instant_mean_sum)
for sl in range(len(instant_mean)):
	instant_mean[sl] = normalization(instant_mean[sl], -1, 1)

print("instant_mean = ", instant_mean)
# creating the lists of voltages
volts = []
for i in instant_mean:
	for j in i:
		volts.append(j)

stim_indexes = list(range(0, len(volts), int(25 / bio_step)))
mins_maxes = calc_max_min(stim_indexes, volts)
print("len(mins_maxes) = ", len(mins_maxes))
for l in mins_maxes:
	print("l = ", l)

max_times = mins_maxes[0]
max_values = mins_maxes[1]
min_times = mins_maxes[2]
min_values = mins_maxes[3]

print("len(max_times) = ", len(max_times[0]), max_times)
print("len(max_values) = ", len(max_values[0]), max_values)
latencies = find_latencies(mins_maxes, bio_step, norm_to_ms=True, inhibition_zero=True)
amplitudes = calc_amplitudes(mins_maxes, latencies, bio_step, after_latencies =False)
print("latencies = ", latencies)
index = 1
yticks = []
yticks_placeholders = []
for slice_mean in instant_mean:
	# print("index = ", index)
	offset = index
	yticks_placeholders.append(index)
	plt.plot([sl + offset for sl in slice_mean])
	for i in range(len(max_times[index - 1])):
		plt.plot(max_times[index - 1][i], max_values[index - 1][i] + offset, marker=".", color='red')
	for i in range(len(min_times[index - 1])):
		plt.plot(min_times[index - 1][i], min_values[index - 1][i] + offset, marker=".", color='green')
	plt.plot(latencies[index - 1], offset - 1, marker='D', color='black')
	yticks.append([sl + offset for sl in slice_mean])
	index += 1
plt.show()
