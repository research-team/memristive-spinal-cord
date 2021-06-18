import os
import numpy as np
import logging as log
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)

color_bef = '#cfcfcf'
color_aft = '#8a8a8a'

# muscles = ['Ext Car Uln', 'Bic Br c l', 'Deltoideus', 'Rect Femoris', 'Tibialis Ant',
#            'Flex Car Uln', 'Tric Br c l', 'Bic Fem c l', 'Gastr c m', 'Soleus m', 'Achilles tendon']

muscles = []
params = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]

bar_names_ba = ['Left', 'Right']
bar_indicies = range(len(bar_names_ba))

dict_data = dict()

times_range = range(2)


def merge(list_of_list):
	return sum(list_of_list, [])


def read_data(datapath):
	filenames = [name[:-4] for name in os.listdir(f"{datapath}") if name.endswith(".xlsx")]
	for filename in filenames:
		filename = '01 feb.xlsx'
		with open(f"{datapath}/{filename}", encoding='windows-1251') as file:
			# remove header
			header = file.readline().strip().split(".")[-5:]
			assert header == params, 'Проверь кол-во столбцов в файле'
			# get data
			time_index = 0
			prev_time = None
			for index, line in enumerate(file):
				# читает строку, раскидывает по переменным
				line = line.strip().replace(",", ".").split(";")
				name, time, pattern, muscle, side, *values = line  # *values = freq, stiff, decr, relax, creep

				if muscle not in muscles:
					muscles.append(muscle)

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
				if time_index >= 2:
					break

				# заполнение словаря
				if name not in dict_data:
					dict_data[name] = {}
				if muscle not in dict_data[name]:
					dict_data[name][muscle] = {}
				if side not in dict_data[name][muscle]:
					dict_data[name][muscle][side] = {t: {} for t in times_range}
					for t in times_range:
						dict_data[name][muscle][side][t] = {p: [] for p in params}

				# перезапись повторяющихся мышц
				for p, v in zip(params, map(float, values)):
					if len(dict_data[name][muscle][side][time_index][p]) >= 6:
						dict_data[name][muscle][side][time_index][p] = []
					dict_data[name][muscle][side][time_index][p].append(v)

				prev_time = time


def check_norm_dist(list_for_check):
	stat, p = stats.shapiro(list_for_check)
	if p < 0.05:
		return False
	else:
		return True


def plotting(datapath, muscles, filename):
	# заполнение списков значениями показателей, взятыми у каждого человека за определенный период времени
	for muscle in muscles:
		for param in params:
			# хранит: k - индексы сравниваемых пар, v - p value
			stat_dict = {'Left': None,
			             'Right': None}

			mean_dict = {'Left': [],
			             'Right': []}

			se_dict = {'Left': [],
			           'Right': []}

			for side in "Left", "Right":

				all_data = []

				for time in times_range:
					# добавляет
					try:
						all_data.append([v[muscle][side][time][param] for v in dict_data.values()])
					# если мышцы нет в прочитаном файле, добавить и заполнить 0
					except Exception as e:
						print(f"Кажется, не хватает какой-то мышцы: {e}")
						print(f"Заполнено нулями")
						all_data.append([[0, 0, 0, 0, 0, 0]])

				array_times = [merge(all_data[t]) for t in times_range]
				mean = [np.mean(array_times[t]) for t in times_range]

				if side == 'Left':
					mean_dict['Left'].append(mean[0])
					se_before = stats.sem(array_times[0])
					se_dict['Left'].append(se_before)
					mean_dict['Left'].append(mean[1])
					se_after = stats.sem(array_times[1])
					se_dict['Left'].append(se_after)
				if side == 'Right':
					mean_dict['Right'].append(mean[0])
					se_before = stats.sem(array_times[0])
					se_dict['Right'].append(se_before)
					mean_dict['Right'].append(mean[1])
					se_after = stats.sem(array_times[1])
					se_dict['Right'].append(se_after)

				# statistic
				for index, t in enumerate(times_range[:-1]):
					if check_norm_dist(array_times[t]):  # если распределение первой выборки нормальное
						for next_t in times_range[index + 1:]:
							if check_norm_dist(array_times[next_t]):  # если распределение второй выборки нормальное
								_, stat_val = stats.ttest_rel(array_times[t], array_times[next_t])
								stat_key = (t, next_t)
								if side == 'Left':
									stat_dict['Left'] = {stat_key: stat_val}
								if side == 'Right':
									stat_dict['Right'] = {stat_key: stat_val}
							else:
								_, p = stats.wilcoxon(array_times[t], array_times[next_t])
								stat_key = (t, next_t)
								stat_val = p
								if side == 'Left':
									stat_dict['Left'] = {stat_key: stat_val}
								if side == 'Right':
									stat_dict['Right'] = {stat_key: stat_val}
					else:  # если распределение первой выборки НЕнормальное
						for next_t in times_range[index + 1:]:
							_, p = stats.wilcoxon(array_times[t], array_times[next_t])
							stat_key = (t, next_t)
							stat_val = p
							if side == 'Left':
								stat_dict['Left'] = {stat_key: stat_val}
							if side == 'Right':
								stat_dict['Right'] = {stat_key: stat_val}

			plot(mean_dict, se_dict, param=param, muscle=muscle, save_to=datapath,
			             pval_dict=stat_dict, filename=filename)

			log.info(f"Отрисован {param}_{muscle}")


def render_stat(pval_dict, mean, axis=None):
	if axis is None:
		axis = plt

	for side, pair_pval in pval_dict.items():
		side = 0 if side == 'Left' else 1

		pairs = [pair for pair, pval in pair_pval.items() if pval < 0.05]

		def calc_line_height(pair):
			before_bar, after_bar = pair
			return max(*mean[before_bar],
			           *mean[after_bar]) + line_upper

		line_upper = max(mean[side]) * 0.08
		serif_size = line_upper / 5
		bar_shift = 1 / 2.5
		line_height = list(map(calc_line_height, pairs))

		# plot text and lines
		if pairs:
			left_bar = side - 0.25
			right_bar = side + 0.25
			hline = line_height[0]
			# line
			if axis:
				bar_shift = 1 / 50
				line_x1, line_x2 = left_bar + bar_shift, right_bar - bar_shift
				serif_x1, serif_x2 = left_bar + bar_shift, right_bar - bar_shift
			else:
				line_x1, line_x2 = left_bar + bar_shift, right_bar - bar_shift
				serif_x1, serif_x2 = left_bar + bar_shift, right_bar - bar_shift
			serif_y1, serif_y2 = hline - serif_size, hline

			axis.plot([line_x1, line_x2], [hline, hline], color='k')
			axis.plot([serif_x1, serif_x1], [serif_y1, serif_y2], color='k')
			axis.plot([serif_x2, serif_x2], [serif_y1, serif_y2], color='k')

			axis.text((left_bar + right_bar) / 2, hline + line_upper / 5, "*", ha='center')
		# axis.plot([(left_bar + right_bar) / 2], [hline + line_upper / 5], '.', color='r', ms=15)


def near_round(x, base=5.0):
	return base * np.ceil(x / base)


def plot(mean_dict, se_dict, param=None, muscle=None, show=False, save_to=None, pval_dict=None, filename=None):
	plt.close()
	fig, ax = plt.subplots(figsize=(4, 3))
	# styles
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	for label in ax.get_xticklabels():
		label.set_fontsize(15)
	for label in ax.get_yticklabels():
		label.set_fontsize(15)
	for axis in 'bottom', 'left':
		ax.spines[axis].set_linewidth(1.5)
	ax.xaxis.set_tick_params(width=1.5)
	ax.yaxis.set_tick_params(width=1.5)

	# plot data
	x = np.arange(len(bar_names_ba))
	mean_before = [mean_dict['Left'][0], mean_dict['Right'][0]]
	mean_after = [mean_dict['Left'][1], mean_dict['Right'][1]]
	se_before = [se_dict['Left'][0], se_dict['Right'][0]]
	se_after = [se_dict['Left'][1], se_dict['Right'][1]]
	mean_for_stat_render = mean_before, mean_after

	# ticks
	max_val = max(max(mean_before), max(mean_after))
	if max_val <= 2:
		step = 0.5
	elif 2 < max_val <= 10:
		step = 1
	elif 10 < max_val <= 25:
		step = 5
	elif 25 < max_val <= 100:
		step = 10
	else:
		step = 100
	max_nearby = near_round(max_val, step)

	# set limits and ticks
	width = 0.35
	ax.bar(x - width / 2, mean_before, width, yerr=se_before, error_kw={'ecolor': '0.1', 'capsize': 3}, label='before',
	       color=color_bef)
	ax.bar(x + width / 2, mean_after, width, yerr=se_after, error_kw={'ecolor': '0.1', 'capsize': 3}, label='after',
	       color=color_aft)

	# mean_for_render_stat = [mean_b, mean_aft]

	# render_stat(pval_dict=pval_dict, mean=mean_for_render_stat)

	ax.set_xticks(bar_indicies)
	ax.set_xticklabels(bar_names_ba)
	ax.set_ylabel(param, fontdict={'size': 15})
	if max_nearby <= 2:
		ax.set_yticks(np.arange(int(0), max_nearby + 0.01, step))
		ax.set_yticklabels(np.arange(int(0), max_nearby + 0.01, step))
	else:
		ax.set_yticks(range(0, int(max_nearby) + 1, step))
		ax.set_yticklabels(range(0, int(max_nearby) + 1, step))
	ax.set_ylim(0, max_nearby)
	ax.set_xlim(-0.5, 1.5)

	axins = ax.inset_axes([0.0, 0.1, 1.0, 1.0])
	axins.set_xticks(bar_indicies)
	axins.patch.set_alpha(0)
	axins.axis('off')
	axins.set_xlim(-0.5, 1.5)
	axins.set_ylim(0, max_nearby * 1.1)

	render_stat(pval_dict=pval_dict, mean=mean_for_stat_render, axis=axins)

	# saving
	plt.tight_layout()
	plt.legend(loc="lower right")
	folder_name = filename[:-4]
	save_folder = f'{save_to}/{folder_name}'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	plt.savefig(f'{save_folder}/{muscle}_{param}_BA.png', format='png')
	if show:
		plt.show()
	plt.close()


def main():
	path = '/home/b-rain/SCC'
	folder = 'Suleimanov'
	datapath = os.path.join(path, folder)
	read_data(datapath=datapath)
	print('Done')


if __name__ == '__main__':
	main()
