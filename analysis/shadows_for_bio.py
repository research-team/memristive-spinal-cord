from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np
from matplotlib import pylab as plt
from cycler import cycler
import matplotlib.patches as mpatches

# importing bio runs from the function 'bio_data_runs'
bio_runs = bio_data_runs()

# forming list for shadows plotting
all_bio_slices = []
step = 0.25
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
yticks = []
times = [time * step for time in range(len(all_bio_slices[0][0]))]
colors = ['black', 'saddlebrown', 'firebrick', 'sandybrown', 'olivedrab']
texts = ['1', '2', '3', '4', '5']
patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]

plt.rc('axes', prop_cycle=cycler(color=colors))
for index, sl in enumerate(all_bio_slices):
	offset = index * 6
	# for run in range(len(sl)):
		# plt.plot(times, [s + offset for s in sl[run]])
# black_patches = mpatches.Patch(color='black', label='1')
# silver_patches = mpatches.Patch(color='silver', label='2')
# firebrick_patches = mpatches.Patch(color='firebrick', label='3')
# sandybrown_patches = mpatches.Patch(color='sandybrown', label='4')
# gold_patches = mpatches.Patch(color='gold', label='5')
# plt.legend(handles=patches, loc='upper right')
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
# plt.yticks(yticks, range(1, len(all_bio_slices) + 1), fontsize=14)
# plt.xlim(0, 25)
# plt.show()
# plot shadows
for index, run in enumerate(all_bio_slices):
	print(len(run), run)
	offset = index * 6
	# plt.plot([r + offset for r in run ])
	mean_data = list(map(lambda elements: np.mean(elements), zip(*run)))
	times = [time * step for time in range(len(mean_data))]
	means = [voltage + offset for voltage in mean_data]
	yticks.append(means[0])
	minimal_per_step = [min(a) for a in zip(*run)]
	maximal_per_step = [max(a) for a in zip(*run)]
	plt.plot(times, means, linewidth=0.5, color='k')
	plt.fill_between(times, [mini + offset for mini in minimal_per_step],
	                 [maxi + offset for maxi in maximal_per_step], alpha=0.35)
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
plt.yticks(yticks, range(1, len(all_bio_slices) + 1), fontsize=14)
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()