import os
from datetime import datetime
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# time_index:
# 0 -'до',
# 1 -'после',
# 2- '30 мин',
# 3- '60 мин',
# 4- '6ч',
# 5- '24ч'
color_l = '#89cc76'
color_r = '#ffe042'
muscles = ('Ext Car Uln', 'Bic Br c l', 'Deltoideus', 'Rect Femoris', 'Tibialis Ant', 'Flex Car Uln', 'Tric Br c l', 'Bic Fem c l', 'Gastr c m', 'Soleus m', 'Achilles t')

path = 'C:/Users/exc24/PycharmProjects/test/data/'

params = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]
times_range = range(6)

dict_data = dict()
k_freq, k_stiff, k_decr, k_relax, k_creep = params

filenames = [name for name in os.listdir(f"{path}/csv") if name.endswith(".csv")]

def merge(list_of_list):
	return sum(list_of_list, [])

for filename in filenames:
	with open(f"{path}/csv/{filename}", encoding='windows-1251') as file:
		# remove header
		header = file.readline().strip().split(";")[-5:]
		assert header == params
		# get data
		time_index = 0
		prev_time = None
		for index, line in enumerate(file):
			line = line.strip().replace(",", ".").split(";")
			name, time, pattern, muscle, side, *values = line   # *data = freq, stiff, decr, relax, creep
			try:
				time = datetime.strptime(time, "%d.%m.%Y %H:%M:%S")
			except ValueError:
				time = datetime.strptime(time, "%d.%m.%Y %H:%M")

			if prev_time is None:
				prev_time = time

			if (time - prev_time).seconds / 60 > 20:
				time_index += 1
			if time_index >= 6:
				break

			if name not in dict_data:
				dict_data[name] = {}
			if muscle not in dict_data[name]:
				dict_data[name][muscle] = {}
			if side not in dict_data[name][muscle]:
				dict_data[name][muscle][side] = {t: {} for t in range(6)}
				for t in range(6):
					dict_data[name][muscle][side][t] = {p: [] for p in params}

			for p, v in zip(params, map(float, values)):
				if len(dict_data[name][muscle][side][time_index][p]) >= 6:
					dict_data[name][muscle][side][time_index][p] = []
				dict_data[name][muscle][side][time_index][p].append(v)

			prev_time = time

for muscle in muscles:
	all_freq_left = []
	for i in times_range:
		tmp = []
		for v in dict_data.values():
			tmp.append(v[muscle]['Left'][i][k_freq])
		all_freq_left.append(tmp)

	all_freq_right = []
	for i in times_range:
		tmp = []
		for v in dict_data.values():
			tmp.append(v[muscle]['Right'][i][k_freq])
		all_freq_right.append(tmp)

	# print(*all_freq_left, sep='\n')


	freq_mean_left = [np.mean(merge(all_freq_left[time])) for time in times_range]
	freq_se_left = [stats.sem(merge(all_freq_left[time])) for time in times_range]

	freq_mean_right = [np.mean(merge(all_freq_right[time])) for time in times_range]
	freq_se_right = [stats.sem(merge(all_freq_right[time])) for time in times_range]

	bar_names = ['before', 'after', '30min', '60min', '6h', '24h']

	# left
	bar_indicies = range(len(bar_names))
	plt.figure(figsize=(6, 6))
	plt.bar(bar_indicies, freq_mean_left, yerr=freq_se_left, error_kw={'ecolor': '0.1', 'capsize': 6}, color=color_l)
	plt.xticks(bar_indicies, bar_names)
	plt.savefig(f'{path}/{muscle}_left.png', format='png')
	plt.tight_layout()
	# plt.show()
	plt.close()

	# right
	plt.figure(figsize=(6, 6))
	plt.bar(bar_indicies, freq_mean_right, yerr=freq_se_right, error_kw={'ecolor': '0.1', 'capsize': 6}, color=color_r)
	plt.xticks(bar_indicies, bar_names)
	plt.tight_layout()
	plt.savefig(f'{path}/{muscle}_right.png', format='png')
	# plt.show()
	plt.close()

	# left + right
	plt.figure(figsize=(6, 6))
	x = np.arange(len(bar_names))
	width = 0.35
	rects1 = plt.bar(x - width / 2, freq_mean_left, width, yerr=freq_se_left, error_kw={'ecolor': '0.1', 'capsize': 6},
	                 label='L', color=color_l)
	rects2 = plt.bar(x + width / 2, freq_mean_right, width, yerr=freq_se_right, error_kw={'ecolor': '0.1', 'capsize': 6},
	                 label='R', color=color_r)
	plt.xticks(bar_indicies, bar_names)
	plt.legend()
	plt.tight_layout()
	plt.savefig(f'{path}/{muscle}_combo.png', format='png')
	# plt.show()
	plt.close()
