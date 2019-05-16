import os
import numpy as np
import pylab as plt
from collections import defaultdict

step = 0.025
sim_time = 275

path = '/home/alex/res/'

groups = defaultdict(list)

for filename in sorted([filename for filename in os.listdir(path) if filename.endswith('v0')]):
	with open(f'{path}/{filename}') as file:
		groups[filename.split('r')[0]].append(list(map(float, file.readlines())))
	print(f"done {filename}")

for group_name, group_data in groups.items():
	print(group_name)

	y = list(map(lambda d: -np.mean(d) * 10**9, zip(*group_data)))
	x = [t * step for t in range(len(y))]

	plt.figure(figsize=(16, 9))
	plt.subplot(311)
	plt.plot(x, y)
	plt.suptitle(group_name)

	for t in range(0, sim_time, 25):
		plt.axvline(x=t, color='k')

	plt.axvline(x=150, linewidth=3, color='k')


	plt.grid()
	plt.xlim(0, sim_time)
	plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')

	plt.subplot(312)
	plt.grid()
	plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
	for t in range(0, sim_time, 25):
		plt.axvline(x=t, color='k')

	plt.subplot(313)
	plt.grid()
	plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25),
	           ["{}\n{}".format(global_time - 25 * (global_time // 25), global_time)
	            for global_time in range(0, sim_time + 1, 5 if sim_time <= 275 else 25)],
	           fontsize=8)
	for t in range(0, sim_time, 25):
		plt.axvline(x=t, color='k')

	plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.08)
	plt.savefig(f"{path}/{group_name}.png", format="png", dpi=200)
	plt.close()

