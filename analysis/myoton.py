import os
import numpy as np
import logging as log
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)

color_bef = '#cfcfcf'
color_aft = '#8a8a8a'

muscles = []
params = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]

bar_names = ['Left', 'Right']
bar_indicies = range(len(bar_names))

dict_data = dict() # словарь с значениями

times_range = range(2) # "до" - 0;  "после" - 1

# суммация для списков
def merge(list_of_list):
	return sum(list_of_list, [])

# опредедение нормальности распределения данных
def check_norm_dist(list_for_check):
	stat, p = stats.shapiro(list_for_check)
	if p < 0.05:
		return False
	else:
		return True

# округление
def near_round(x, base=5.0):
	return base * np.ceil(x / base)

# отрисовка статистики
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

# считывание данных с файла
def read_data(datapath):
	filenames = [name for name in os.listdir(f"{datapath}") if name.endswith(".csv")]
	for filename in filenames:
		# подсчет кол-во тапов
		side_counter = 0
		# filename = '02feb2021.csv'
		with open(f"{datapath}/{filename}", encoding='windows-1251') as file:
			# повторное открытие файла для подсчета общего кол-ва строк
			with open(f"{datapath}/{filename}", encoding='windows-1251') as file_for_row_count:
				row_in_file = sum(1 for row in file_for_row_count) - 1  # - 1 without header
			# удаление первой строки с заголовком
			header = file.readline().strip().split(";")[-5:]
			assert header == params, 'Проверь кол-во столбцов в файле'

			time_index = 0  # если равен 0 -> "до", 1 -> "после"
			row_count = 1  # подсчет прочтенных строк
			prev_time = None  # время предыдущей строки
			prev_side = None  # сторона предыдущей строки
			prev_muscle = None  # мышца предыдущей строки
			prev_muscle_from_list = None  # предыдущая мышца непосредственно из списка

			# считываение построчно
			for index, line in enumerate(file):
				# разделение строки по переменным
				line = line.strip().replace(",", ".").split(";")
				name, time, pattern, muscle, side, *values = line  # *values = freq, stiff, decr, relax, creep

				# замена названия в словаре
				if muscle == "Achilles tendon":
					muscle = "Achilles t"
					print('"Achilles tendon" переименован в "Achilles t"')

				# создание списка мышц
				if muscle not in muscles:
					muscles.append(muscle)

				if prev_muscle is None:
					prev_muscle = muscle

				if prev_side is None:
					prev_side = side
				if prev_side == side:
					side_counter += 1

				# парсит время в 2ух форматах
				try:
					time = datetime.strptime(time, "%d.%m.%Y %H:%M:%S")
				except ValueError:
					time = datetime.strptime(time, "%d.%m.%Y %H:%M")

				# если разница м-у предыдущим и нынешним временем >20 мин => новый временной индекс
				# только два временных индекса "до"(0) и "после"(1)
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

				# заполнение словаря значениями параметров (строго 6 значений)
				for p, v in zip(params, map(float, values)):
					# если больше 6 значений -> обнуление списка
					if len(dict_data[name][muscle][side][time_index][p]) >= 6:
						dict_data[name][muscle][side][time_index][p] = []
					# если меньше 6 значений -> дублировние последнего значения
					if prev_side != side and side_counter < 6:
						if len(dict_data[name][prev_muscle][prev_side][time_index][p]) != 0:
							last_value = dict_data[name][prev_muscle][prev_side][time_index][p][-1]
							dict_data[name][prev_muscle][prev_side][time_index][p].append(last_value)
						print(
							f"В списке {prev_muscle}, {prev_side} не хватает значений, продублировано последнее значение")

					# определение предыдущей мышцы из списка (на случай, если в измерениях "до" или "после" не будет мышцы)
					if muscles.index(muscle) != 0:
						prev_muscle_from_list = muscles[muscles.index(muscle) - 1]

					# если мышца в списке, но нет измерениий -> заполнение нулями
					if prev_muscle_from_list:
						if prev_muscle_from_list != muscle and len(
								dict_data[name][prev_muscle_from_list][prev_side][time_index][p]) == 0:
							while len(dict_data[name][prev_muscle_from_list][prev_side][time_index][p]) < 6:
								dict_data[name][prev_muscle_from_list][prev_side][time_index][p].append(0)

					# добавление значения в словарь
					dict_data[name][muscle][side][time_index][p].append(v)

					# проверка на завершение считывания файла
					# если у последней мышцы не хватает значений -> дублирование последнего
					if row_count == row_in_file:
						for s in range(len(bar_names)):
							for i in times_range:
								if len(dict_data[name][muscle][bar_names[s]][i][p]) < 6:
									last_value = dict_data[name][muscle][bar_names[s]][i][p][-1]
									dict_data[name][muscle][bar_names[s]][i][p].append(last_value)
									print(
										f"В списке {muscle}, {bar_names[s]}, не хватает значений, продублировано последнее значение")

				row_count += 1
				if prev_side != side:
					side_counter = 1
				prev_time = time
				prev_side = side
				prev_muscle = muscle

			log.info(f"\n Файл {filename} \n")

			# создание нового словаря, подсчет статистики и отрисовка
			plotting(datapath=datapath, muscles=muscles, filename=filename)

# перераспределение данных
def plotting(datapath, muscles, filename):
	# заполнение списков значениями показателей, взятыми у каждого человека за определенный период времени
	for muscle in muscles:
		for param in params:
			# хранит: k - индексы сравниваемых пар, v - p value
			stat_dict = {'Left': None,
			             'Right': None}

			# словарь значений среднего
			mean_dict = {'Left': [],
			             'Right': []}

			# словарь ошибки среднего
			se_dict = {'Left': [],
			           'Right': []}

			for side in "Left", "Right":
				all_data = []

				# сортировка по времени
				for time in times_range:
					all_data.append([v[muscle][side][time][param] for v in dict_data.values()])

				# array_times = [ [до], [после] ]
				array_times = [merge(all_data[t]) for t in times_range]

				# на случай, если значений всё же меньше 6 (если появится - плохо)
				# дублирование последнего значения
				if len(array_times[0]) != 0 and len(array_times[1]) != 0:
					if len(array_times[0]) < len(array_times[1]):
						while len(array_times[0]) != len(array_times[1]):
							print(
								f"ВНИМАНИЕ, в списке 'до' {len(array_times[0])} значений, продублировано последнее значение ({muscle}, {side})")
							array_times[0].append(array_times[0][-1])
					else:
						while len(array_times[1]) != len(array_times[0]):
							print(
								f"ВНИМАНИЕ, в списке 'после' {len(array_times[1])} значений, продублировано последнее значение ({muscle}, {side})")
							array_times[1].append(array_times[1][-1])
				else:
					if len(array_times[0]) == 0:
						while len(array_times[0]) < len(array_times[1]):
							print(f"ВНИМАНИЕ, в списке 'до' нет значений, заполнено нулями ({muscle}, {side})")
							array_times[0].append(0)
					else:
						while len(array_times[1]) < len(array_times[0]):
							print(f"ВНИМАНИЕ, в списке 'после' нет значений, заполнено нулями ({muscle}, {side})")
							array_times[1].append(0)

				# расчет среднего значения
				mean = [np.mean(array_times[t]) for t in times_range]

				# формирование словаря среднего значения и ошибки среднего
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

				# расчет статистики
				for index, t in enumerate(times_range[:-1]):
					if check_norm_dist(array_times[t]):  # если распределение первой выборки нормальное
						for next_t in times_range[index + 1:]:
							if check_norm_dist(array_times[next_t]):
								# если распределение второй выборки нормальное
								_, stat_val = stats.ttest_rel(array_times[t], array_times[next_t])
								stat_key = (t, next_t)

								# заполнение словаря "до/после" : "p value"
								if side == 'Left':
									stat_dict['Left'] = {stat_key: stat_val}
								if side == 'Right':
									stat_dict['Right'] = {stat_key: stat_val}
							else:
								# если распределение второй выборки НЕнормальное
								_, p = stats.wilcoxon(array_times[t], array_times[next_t])
								stat_key = (t, next_t)
								stat_val = p

								# заполнение словаря "до/после" : "p value"
								if side == 'Left':
									stat_dict['Left'] = {stat_key: stat_val}
								if side == 'Right':
									stat_dict['Right'] = {stat_key: stat_val}
					else:  # если распределение первой выборки НЕнормальное
						for next_t in times_range[index + 1:]:  # распределение второй выборки не важно
							_, p = stats.wilcoxon(array_times[t], array_times[next_t])
							stat_key = (t, next_t)
							stat_val = p

							# заполнение словаря "до/после" : "p value"
							if side == 'Left':
								stat_dict['Left'] = {stat_key: stat_val}
							if side == 'Right':
								stat_dict['Right'] = {stat_key: stat_val}

			# отрисовка
			plot(mean_dict, se_dict, param=param, muscle=muscle, save_to=datapath,
			     pval_dict=stat_dict, filename=filename)

			log.info(f"Отрисован {param} {muscle}")

# отрисовка
def plot(mean_dict, se_dict, param=None, muscle=None, show=False, save_to=None, pval_dict=None, filename=None):
	# закрытие предыдущего изображения
	plt.close()
	# зсоздание нового
	fig, ax = plt.subplots(figsize=(4, 3))
	# оси
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

	# перераспределение данных
	x = np.arange(len(bar_names))
	mean_before = [mean_dict['Left'][0], mean_dict['Right'][0]]
	mean_after = [mean_dict['Left'][1], mean_dict['Right'][1]]
	se_before = [se_dict['Left'][0], se_dict['Right'][0]]
	se_after = [se_dict['Left'][1], se_dict['Right'][1]]
	mean_for_stat_render = mean_before, mean_after

	# отметки на осях
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

	# толщина баров
	width = 0.35

	# отрисовка баров
	ax.bar(x - width / 2, mean_before, width, yerr=se_before, error_kw={'ecolor': '0.1', 'capsize': 3}, label='before',
	       color=color_bef)
	ax.bar(x + width / 2, mean_after, width, yerr=se_after, error_kw={'ecolor': '0.1', 'capsize': 3}, label='after',
	       color=color_aft)

	# подписи к осям
	ax.set_xticks(bar_indicies)
	ax.set_xticklabels(bar_names)
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

	# сохранение
	plt.tight_layout()
	plt.legend(loc="lower right")
	folder_name = filename[:-4]
	folder_name_muscle = muscle
	save_folder = f'{save_to}/{folder_name}/{folder_name_muscle}'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
	plt.savefig(f'{save_folder}/{muscle}_{param}.png', format='png')
	if show:
		plt.show()
	plt.close()


def main():
	# абсолютный путь
	path = 'C:/MYO/'
	# папка с файлами
	folder = 'Suleimanov/data'

	# формирование пути к файлу
	datapath = os.path.join(path, folder)
	# вызов функции чтения и отрисовки
	read_data(datapath=datapath)

	print('Done')


if __name__ == '__main__':
	main()
