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


def plot(mean, err, side=None, param=None, muscle=None, show=False, save_to=None, pval_dict=None):
	color = color_l if side == "Left" else color_r
	plt.figure(figsize=(4, 3))
	plt.bar(bar_indicies, mean, yerr=err, error_kw={'ecolor': '0.1', 'capsize': 6}, color=color)
	plt.xticks(bar_indicies, bar_names)

	if pval_dict:
		pass

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


def check_norm_dist(list_for_check):
	stat, p = stats.shapiro(list_for_check)
	if p < 0.05:
		return False
	else:
		return True


def plotting(savepath):
	# заполнение списков значениями показателей, взятыми у каждого человека за определенный период времени
	for muscle in muscles:
		for param in params:
			mean_left, se_left, mean_right, se_right = [None] * 4

			for side in "Left", "Right":
				all_data = []
				stat_dict = {}
				for time in times_range:
					all_data.append([v[muscle][side][time][param] for v in dict_data.values()])

				array_times = [merge(all_data[t]) for t in times_range]
				mean = [np.mean(array_times[t]) for t in times_range]
				if side == "Left":
					mean_left = mean
				else:
					mean_right = mean

				se = [stats.sem(array_times[t]) for t in times_range]
				if side == "Left":
					se_left = se
				else:
					se_right = se

				for index, t in enumerate(times_range[:-1]):
					if t == 0:
						if check_norm_dist(array_times[t]):  # если распределение первой выборки нормальное
							for next_t in times_range[index + 1:]:
								if check_norm_dist(array_times[next_t]):  # если распределение второй выборки нормальное
									_, p = stats.ttest_rel(array_times[t], array_times[next_t])
								else:
									_, p = stats.wilcoxon(array_times[t], array_times[next_t])
									stat_key = (t, next_t)
									stat_dict[stat_key] = p
						else:  # если распределение первой выборки НЕнормальное
							for next_t in times_range[index + 1:]:
								_, p = stats.wilcoxon(array_times[t], array_times[next_t])
								stat_key = (t, next_t)
								stat_dict[stat_key] = p
					else:
						for next_t in times_range[index + 1:]:
							if check_norm_dist(array_times[next_t]):  # если распределение второй выборки нормальное
								_, p = stats.ttest_ind(array_times[t], array_times[next_t])
							else:
								_, p = stats.mannwhitneyu(array_times[t], array_times[next_t])
							stat_key = (t, next_t)
							stat_dict[stat_key] = p

				plot(mean, se, side=side, param=param, muscle=muscle, save_to=savepath, pval_dict=stat_dict)

			plot_combo(mean_left, se_left, mean_right, se_right, param=param, muscle=muscle, save_to=savepath)
			log.info(f"Отрисован {param}_{muscle}")


def main():
	datapath = 'C:/Users/exc24/PycharmProjects/test/data/'
	read_data(datapath)
	plotting(datapath)


if __name__ == '__main__':
	main()
