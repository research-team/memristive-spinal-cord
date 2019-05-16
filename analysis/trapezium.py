from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from matplotlib import pylab as plt
from analysis.functions import bio_process

# importing the list of all runs of the bio  data from the function 'bio_data_runs'
bio_runs = bio_data_runs()
# calculating the mean value of all runs
mean_data = list(map(lambda elements: np.mean(elements), zip(*bio_runs)))
# number of slices
num_slices = int(len(mean_data) / 100)
slices = []
offset = 0
yticks = []
step = 0.25
# forming list for shadows plotting
all_bio_slices = []
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

# creating the list of lists (dots per slice)
offset = 0
for sl in range(num_slices):
	slices_tmp = []
	for dot in range(offset, offset + 100):
		slices_tmp.append(mean_data[dot])
	slices.append(slices_tmp)
	offset += 100

# creating the list of dots of stimulations
stimulations = []
for stim in range(0, 1601, 100):
	stimulations.append(stim)
print(stimulations)

# creating the list of lists (volts and stimulations)
volts_and_stims = []
volts_and_stims.append(mean_data)
volts_and_stims.append(stimulations)

# calculating the latencies
lat_amp = bio_process(volts_and_stims, 16)
latencies = lat_amp[0]
print("latencies = ", latencies)

# creating the list of x & y coordinates of the lines
x_coor = []
y_coor = []

# plotting the slices
for index, sl in enumerate(all_bio_slices):
	offset = index * 10
	times = [time * step for time in range(len(all_bio_slices[0][0]))]
	for run in range(len(sl)):
		plt.plot(times, [s + offset for s in sl[run]], color='gray')

for index, run in enumerate(slices):
	offset = index * 10
	yticks.append(run[0] + offset)
	times = [time * step for time in range(len(run))]
	plt.plot(times, [s + offset for s in run], linewidth=3)
	# plotting of the dots
	plt.plot(latencies[index], run[int(latencies[index] / step)] + offset, marker='.', markersize=8)
	# pltting of the lines
	x_coor.append(latencies[index])
	y_coor.append(run[int(latencies[index] / step)] + offset)
	x_2_coors = []
	y_2_coors = []
	if len(x_coor) > 1:
		x_2_coors.append(x_coor[-2])
		x_2_coors.append(x_coor[-1])
		y_2_coors.append(y_coor[-2])
		y_2_coors.append(y_coor[-1])
		plt.plot(x_2_coors, y_2_coors, linestyle='--', color='black')
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
plt.yticks(yticks, range(1, len(slices) + 1), fontsize=14)
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()
