from analysis.patterns_in_bio_data import bio_data_runs
from matplotlib import pylab as plt
from math import sqrt

# importing the list of all runs of the bio  data from the function 'bio_data_runs'
bio_runs = bio_data_runs()
offset = 0
all_bio_slices = []
step = 0.25

# forming list for the plot
for k in range(len(bio_runs)):
	bio_slices= []
	offset= 0
	for i in range(int(len(bio_runs[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio_runs[k][j])
		bio_slices.append(bio_slices_tmp)
		offset += 100
	all_bio_slices.append(bio_slices)   # list [4][16][100]
all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]

# calculating the instant dispersion
instant_mean = []
for slice in range(len(all_bio_slices)):
	instant_mean_sum = []
	for dot in range(len(all_bio_slices[slice][0])):
		instant_mean_tmp = []
		for run in range(len(all_bio_slices[slice])):
			instant_mean_tmp.append(abs(all_bio_slices[slice][run][dot]))
		instant_mean_sum.append(sum(instant_mean_tmp))
	instant_mean.append(instant_mean_sum)

print("instant_mean[0][0] = ", instant_mean[0][0])

# plotting the slices
for index, sl in enumerate(all_bio_slices):
	offset = index * 5
	times = [time * step for time in range(len(all_bio_slices[0][0]))]
	for run in range(len(sl)):
		plt.plot(times, [s + offset for s in sl[run]], color='gray', linewidth=1)

# plotting the dispersion
yticks = []
for index, sl in enumerate(instant_mean):
	offset = index * 5
	yticks.append(sl[0] + offset)
	times = [time * step for time in range(len(sl))]
	plt.plot(times, [s + offset for s in sl], linewidth=2)
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)])
plt.yticks(yticks, range(1, len(instant_mean) + 1))
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()