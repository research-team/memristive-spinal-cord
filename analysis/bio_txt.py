from analysis.functions import read_bio_data
from matplotlib import pylab as plt

data = read_bio_data('/home/anna/Desktop/data/bio/Different Frequencies_20_40_100_250 Hz/250Hz/RMG/2_SCI Rat-1_11-22-2016_RMG_250Hz_one_step.txt')

freq = 250
slice_duration = int(1 / freq * 1000)
slice_duration_in_dots = slice_duration * 4

print(slice_duration)
print(len(data[0]))

slices = []
offset = 0
for k in range(int(len(data[0]) / slice_duration_in_dots)):
	slices_tmp = []
	for i in range(offset, offset + slice_duration_in_dots):
		slices_tmp.append(data[0][i])
	offset += slice_duration_in_dots
	slices.append(slices_tmp)

print(len(slices), len(slices[0]))

yticks = []
for index, sl in enumerate(slices):
	offset = index * 0.1
	times = [time * 0.25 for time in range(len(sl))]
	plt.plot(times, [s + offset for s in sl])
	yticks.append(sl[0] + offset)
plt.xticks(range(slice_duration + 1), [i if i % 1 == 0 else "" for i in range(slice_duration + 1)], fontsize=14)
plt.yticks(yticks, range(1, int(len(data[0]) / slice_duration_in_dots) + 1), fontsize=14)
plt.xlim(0, slice_duration)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()
