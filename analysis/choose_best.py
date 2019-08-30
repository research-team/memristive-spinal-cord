from analysis.functions import read_data
from matplotlib import pylab as plt
from analysis.functions import center_data_by_line,  normalization
import numpy as np

step = 0.25
split_coef = 3

bio = read_data('/home/anna/Desktop/data/4pedal/bio_F_15cms_40Hz_i100_4pedal_no5ht_T_0.25step.hdf5')
y_data = bio[3]

Y = np.array(y_data)
centered = center_data_by_line(Y)
normalized_data = normalization(centered, save_centering=True)

slice_length = int(1000 / 40 / 0.25)
slice_number = len(normalized_data) // slice_length
sliced = np.split(normalized_data, slice_number)

with open("/home/anna/Desktop/data/4pedal/f_15cms", 'w') as file:
	for d in sliced:
		file.write(" ".join(map(str, d)) + "\n")
raise Exception
print(best_run)
# raise Exception

# best_slices = best_run.split(maxsplit=6)
offset = 0
best_slices = []
for run in range(int(len(best_run) / 100)):
	best_slices_tmp = []
	for i in range(offset, offset + 100):
		best_slices_tmp.append(best_run[i])
	offset += 100
	best_slices.append(best_slices_tmp)

for s in range(len(best_slices)):
	# best_slices[s] = str(best_slices[s])
	print(best_slices[s])

# best_run = ', '.join(best_run)
print(len(best_run), best_run)

with open("/home/anna/Desktop/data/4pedal/f_21cms", "w") as file:
	for b in best_slices:
		file.write(" ".join(map(str, b)) + '\n')

bio_slices = []
for k in range(len(bio)):
	bio_slices_t = []
	offset = 0
	for i in range(int(len(bio[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio[k][j])
		offset += 100
		bio_slices_t.append(bio_slices_tmp)
	bio_slices.append(bio_slices_t)
bio_slices = list(zip(*bio_slices))
print(len(bio_slices))
print(len(bio_slices[0]))
print(len(bio_slices[0][0]))

yticks = []

for index, run in enumerate(bio_slices):
	best_slice_border = (index + 1) * 100
	offset = index * split_coef
	mean_data = best_run[best_slice_border - 100:best_slice_border]
	times = [time * step for time in range(len(mean_data))]
	means = [voltage + offset for voltage in mean_data]
	yticks.append(means[0])
	minimal_per_step = [min(a) for a in zip(*run)]
	maximal_per_step = [max(a) for a in zip(*run)]
	plt.plot(times, means, linewidth=0.5, color='k')
	plt.fill_between(times, [mini + offset for mini in minimal_per_step],
	                 [maxi + offset for maxi in maximal_per_step], alpha=0.35)
plt.xlim(0, 25)
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
plt.yticks(yticks, range(1, len(bio_slices) + 1), fontsize=14)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()
yticks = []
for run in bio_slices:
	for index, i in enumerate(run):
		offset = index
		times = [time * 0.25 for time in range(len(i))]  # divide by 10 to convert to ms step
		plt.plot(times, [s+ offset for s  in i])
	plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
	plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
	plt.xlim(0, 25)
	plt.show()
