import os
import numpy as np
import logging as log
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)

color_l = '#89cc76'
color_r = '#ffe042'
color_bef = '#cfcfcf'
color_aft = '#8a8a8a'

patterns = {'DryImmersion': ['Ext Car Uln', 'Bic Br c l', 'Deltoideus', 'Rect Femoris', 'Tibialis Ant',
                             'Flex Car Uln', 'Tric Br c l', 'Bic Fem c l', 'Gastr c m', 'Soleus m', 'Achilles tendon']
            }
params = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]

bar_names_lr = ['Before', 'After']
bar_names_ba = ['Left', 'Right']
bar_indicies = range(len(bar_names_ba))

dict_data = dict()

plotting_types = ("BA", "DYN", "ALL")


def merge(list_of_list):
	return sum(list_of_list, [])


def set_time_range(t_range):
	times_range = range(t_range)
	return times_range


def read_data(datapath, filename, t_range):
	with open(f"{datapath}/{filename}", encoding='windows-1251') as file:
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
			if time_index >= 2:
				break

			times_range = set_time_range(t_range)

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
	# print(dict_data)
	# exit()

def read_data_by_plt_type(datapath, filename, type_for_plotting, t_range):
	if type_for_plotting == "ALL":
		filenames = [name for name in os.listdir(f"{datapath}") if name.endswith(".csv")]
		for filename in filenames:
			log.info(f"Обработан файл {filename}")
			read_data(datapath, filename, t_range)

	if type_for_plotting == "BA":
		read_data(datapath, filename, t_range)

	if type_for_plotting == "DYN":
		pass


def check_norm_dist(list_for_check):
	stat, p = stats.shapiro(list_for_check)
	if p < 0.05:
		return False
	else:
		return True


def plotting(savepath, pattern, type_for_plotting, t_range, filename):
	array_times = []
	mean = None
	muscles = []
	times_range = set_time_range(t_range)

	# проверка паттерна и выбор группы мышц
	if pattern == "DryImmersion":
		muscles = patterns["DryImmersion"]

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

				if type_for_plotting == "ALL":
					array_times = [merge(all_data[t]) for t in times_range]
					mean = [np.mean(array_times[t]) for t in times_range]

				if type_for_plotting == "BA":
					array_times = [merge(all_data[t]) for t in times_range]
					mean = [np.mean(array_times[t]) for t in times_range]

				if type_for_plotting == "DYN":
					pass

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

				# if side == "Left":
				# 	mean_left = mean
				# else:
				# 	mean_right = mean

				# se = [stats.sem(array_times[t]) for t in times_range]
				# if side == "Left":
				# 	se_left = se
				# else:
				# 	se_right = se

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

			# plot(mean, se, side=side, param=param, muscle=muscle, save_to=savepath, pval_dict=stat_dict)

			plot_bef_aft(mean_dict, se_dict, param=param, muscle=muscle, save_to=savepath,
			             pval_dict=stat_dict, filename=filename)
			# plot_combo(mean_left, se_left, mean_right, se_right, param=param, muscle=muscle, save_to=savepath)
			log.info(f"Отрисован {param}_{muscle}")


def render_stat(pval_dict, mean, axis=None):
	"""

	Args:
		pval_dict: dfdsfdfds wtf
		mean (list of list): dfdfdfdfd

	Returns:

	"""
	if axis is None:
		axis = plt

	for side, pair_pval in pval_dict.items():
		side = 0 if side == 'Left' else 1

		pairs = [pair for pair, pval in pair_pval.items() if pval < 0.05]

		# text = {
		# 	'*': (1e-2, 5e-2),
		# 	'**': (1e-3, 1e-2),
		# 	'***': (1e-4, 1e-3),
		# 	'****': (0, 1e-4)
		# }

		def calc_line_height(pair):
			before_bar, after_bar = pair
			return max(*mean[before_bar],
			           *mean[after_bar]) + line_upper

		# sorted_pairs = sorted(pairs, key=lambda pair: pair[1] - pair[0])
		line_upper = max(mean[side]) * 0.08
		serif_size = line_upper / 5
		bar_shift = 1 / 2.5


		# def diff(h1, h2):
		# 	return abs(h2 - h1) < line_upper / 2

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
				# serifs
				serif_x1, serif_x2 = left_bar + bar_shift, right_bar - bar_shift
			serif_y1, serif_y2 = hline - serif_size, hline

			axis.plot([line_x1, line_x2], [hline, hline], color='k')
			axis.plot([serif_x1, serif_x1], [serif_y1, serif_y2], color='k')
			axis.plot([serif_x2, serif_x2], [serif_y1, serif_y2], color='k')

			# check the next lines and move them upper if need
			# for i1 in range(index + 1, len(sorted_pairs)):
			# 	if diff(line_height[i1], line_height[index]):
			# 		line_height[i1] += line_upper
			# pvalue = pair_pval[(0, 1)]

			axis.text((left_bar + right_bar) / 2, hline + line_upper / 5, "*", ha='center')
			# axis.plot([(left_bar + right_bar) / 2], [hline + line_upper / 5], '.', color='r', ms=15)


def near_round(x, base=5.0):
	return base * np.ceil(x / base)


def plot(mean, err, side=None, param=None, muscle=None, show=False, save_to=None, pval_dict=None):
	color = color_l if side == "Left" else color_r
	plt.figure(figsize=(4, 3))

	plt.bar(bar_indicies, mean, yerr=err, error_kw={'ecolor': '0.1', 'capsize': 6}, color=color)
	plt.xticks(bar_indicies, bar_names_lr)

	render_stat(pval_dict=pval_dict, mean=mean)

	plt.savefig(f'{save_to}/{muscle}_{param}_{side}.png', format='png')
	if show:
		plt.show()
	plt.close()


def plot_bef_aft(mean_dict, se_dict, param=None, muscle=None, show=False, save_to=None, pval_dict=None, filename=None):
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



def plot_lr(mean_l, err_l, mean_r, err_r, param=None, muscle=None, show=False, save_to=None):
	plt.figure(figsize=(4, 3))
	x = np.arange(len(bar_names_lr))
	width = 0.35
	plt.bar(x - width / 2, mean_l, width, yerr=err_l, error_kw={'ecolor': '0.1', 'capsize': 3}, label='L',
	        color=color_l)
	plt.bar(x + width / 2, mean_r, width, yerr=err_r, error_kw={'ecolor': '0.1', 'capsize': 3}, label='R',
	        color=color_r)
	plt.xticks(bar_indicies, bar_names_ba)
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(f'{save_to}/{muscle}_{param}_combo.png', format='png')
	if show:
		plt.show()
	plt.close()


def main():
	filename = '02feb2021.csv'
	datapath = 'C:/Users/exc24/PycharmProjects/test/SCC/Suleimanov/data'
	savepath = 'C:/Users/exc24/PycharmProjects/test/SCC/Suleimanov'
	plt_type = 'BA'
	time_range = 2

	read_data_by_plt_type(datapath=datapath, filename=filename, type_for_plotting=plt_type, t_range=time_range, )
	plotting(savepath=savepath, pattern="DryImmersion", type_for_plotting=plt_type, t_range=time_range,
	         filename=filename)


if __name__ == '__main__':
	main()
