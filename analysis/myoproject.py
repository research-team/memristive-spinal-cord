import os
import numpy as np
import logging as log
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)

color_l = '#89cc76'
color_r = '#ffe042'
muscles = ('Ext Car Uln', 'Bic Br c l', 'Deltoideus', 'Rect Femoris', 'Tibialis Ant', 'Flex Car Uln', 'Tric Br c l',
           'Bic Fem c l', 'Gastr c m', 'Soleus m', 'Achilles t')
params = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]
bar_names = ['before', 'after', '30min', '60min', '6h', '24h']
times_range = range(len(bar_names))
bar_indicies = times_range

dict_data = dict()


def merge(list_of_list):
	return sum(list_of_list, [])


def read_data(datapath):
	filenames = [name for name in os.listdir(f"{datapath}/csv") if name.endswith(".csv")]
	for filename in filenames:
		log.info(f"Обработан файл {filename}")
		with open(f"{datapath}/csv/{filename}", encoding='windows-1251') as file:
			# remove header
			header = file.readline().strip().split(";")[-5:]
			assert header == params, 'Проверь кол-во столбцов в файле'
			# get data
			time_index = 0
			prev_time = None
			for index, line in enumerate(file):
				# читает строку, раскидывает по переменным
				line = line.strip().replace(",", ".").split(";")
				name, time, pattern, muscle, side, *values = line  # *values = freq, stiff, decr, relax, creep
				# парсит время в 2ух форматах
				try:
					time = datetime.strptime(time, "%d.%m.%Y %H:%M:%S")
				except ValueError:
					time = datetime.strptime(time, "%d.%m.%Y %H:%M")

				# если разница м-у предыдущим и нынешним временем >20 мин => новый временной индекс
				if prev_time is None:
					prev_time = time
				if (time - prev_time).seconds / 60 > 20:
					time_index += 1
				if time_index >= 6:
					break

				# заполнение словаря
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


def plot(mean, err, side=None, param=None, muscle=None, show=False, save_to=None):
	color = color_l if side == "Left" else color_r
	plt.figure(figsize=(4, 3))
	plt.bar(bar_indicies, mean, yerr=err, error_kw={'ecolor': '0.1', 'capsize': 6}, color=color)
	plt.xticks(bar_indicies, bar_names)
	plt.savefig(f'{save_to}/{muscle}_{param}_{side}.png', format='png')
	plt.tight_layout()
	if show:
		plt.show()
	plt.close()


def plot_combo(mean_l, err_l, mean_r, err_r, param=None, muscle=None, show=False, save_to=None):
	plt.figure(figsize=(4, 3))
	x = np.arange(len(bar_names))
	width = 0.35
	plt.bar(x - width / 2, mean_l, width, yerr=err_l,  error_kw={'ecolor': '0.1', 'capsize': 3}, label='L', color=color_l)
	plt.bar(x + width / 2, mean_r, width, yerr=err_r, error_kw={'ecolor': '0.1', 'capsize': 3}, label='R', color=color_r)
	plt.xticks(bar_indicies, bar_names)
	plt.legend()
	plt.tight_layout()
	plt.savefig(f'{save_to}/{muscle}_{param}_combo.png', format='png')
	if show:
		plt.show()
	plt.close()


def plotting(savepath):
	# заполнение списков значениями показателей, взятыми у каждого человека за определенный период времени
	for muscle in muscles:
		for param in params:
			all_left = []
			all_right = []
			for time in times_range:
				all_left.append([v[muscle]["Left"][time][param] for v in dict_data.values()])
				all_right.append([v[muscle]['Right'][time][param] for v in dict_data.values()])
			############ WARNING ############
			array_times = [merge(all_left[t]) for t in times_range]
			print(*zip(bar_names, array_times), sep='\n')
			exit()
			############
			mean_left = [np.mean(merge(all_left[t])) for t in times_range]
			se_left = [stats.sem(merge(all_left[t])) for t in times_range]
			#
			mean_right = [np.mean(merge(all_right[t])) for t in times_range]
			se_right = [stats.sem(merge(all_right[t])) for t in times_range]
			#
			plot(mean_left, se_left, side="Left", param=param, muscle=muscle, save_to=savepath)
			plot(mean_right, se_right, side="Right", param=param, muscle=muscle, save_to=savepath)
			plot_combo(mean_left, se_left, mean_right, se_right, param=param, muscle=muscle, save_to=savepath)

			log.info(f"Отрисован {param}_{muscle}")


def main():
	datapath = 'C:/Users/exc24/PycharmProjects/test/data/'
	read_data(datapath)
	plotting(datapath)


if __name__ == '__main__':
	main()

"""
# заполнение списков значениями показателей, взятыми у каждого человека за определенный период времени
for muscle in muscles:
	for side in sides:
		for time in times_range:
			all_left = []
			all_right = []
			for param in params:
				tmp = []
				for v in dict_data.values():
					tmp.append(v[muscle][side][time][param])
				if side == sides[0]:
					if param == params[0]:
						all_freq_left.append(tmp)
					if param == params[1]:
						all_stif_left.append(tmp)
					if param == params[2]:
						all_decr_left.append(tmp)
					if param == params[3]:
						all_relax_left.append(tmp)
					if param == params[4]:
						all_creep_left.append(tmp)
				else:
					if param == params[0]:
						all_freq_right.append(tmp)
					if param == params[1]:
						all_stif_right.append(tmp)
					if param == params[2]:
						all_decr_right.append(tmp)
					if param == params[3]:
						all_relax_right.append(tmp)
					if param == params[4]:
						all_creep_right.append(tmp)
#
# for i in range(all_list):
# 	for m, er in zip(all_mean_list, all_se_list):
# 		m = [np.mean(merge(i[time])) for time in times_range]
# 		er = [stats.sem(merge(i[time])) for time in times_range]
# freq_mean_right = [np.mean(merge(all_freq_right[time])) for time in times_range]
# freq_se_right = [stats.sem(merge(all_freq_right[time])) for time in times_range]
#
# stif_mean_left = [np.mean(merge(all_stif_left[time])) for time in times_range]
# stif_se_left = [stats.sem(merge(all_stif_left[time])) for time in times_range]
# stif_mean_right = [np.mean(merge(all_stif_right[time])) for time in times_range]
# stif_se_right = [stats.sem(merge(all_stif_right[time])) for time in times_range]
#
# decr_mean_left = [np.mean(merge(all_param[time])) for time in times_range]
# decr_se_left = [stats.sem(merge(all_param[time])) for time in times_range]
# decr_mean_right = [np.mean(merge(all_param[time])) for time in times_range]
# decr_se_right = [stats.sem(merge(all_param[time])) for time in times_range]

# freq_mean_right = [np.mean(merge(all_freq_right[time])) for time in times_range]
# freq_se_right = [stats.sem(merge(all_freq_right[time])) for time in times_range]

	# # графики
	#
	# # left
	# plt.figure(figsize=(4, 3))
	# plt.bar(bar_indicies, freq_mean_left, yerr=freq_se_left, error_kw={'ecolor': '0.1', 'capsize': 6}, color=color_l)
	# plt.xticks(bar_indicies, bar_names)
	# plt.savefig(f'{path}/{muscle}_{params[0]}_left.png', format='png')
	# plt.tight_layout()
	# # plt.show()
	# plt.close()
	#
	# # right
	# plt.figure(figsize=(4, 3))
	# plt.bar(bar_indicies, freq_mean_right, yerr=freq_se_right, error_kw={'ecolor': '0.1', 'capsize': 6}, color=color_r)
	# plt.xticks(bar_indicies, bar_names)
	# plt.tight_layout()
	# plt.savefig(f'{path}/{muscle}_{params[0]}_right.png', format='png')
	# # plt.show()
	# plt.close()
"""