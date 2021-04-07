import os
import numpy as np
import logging as log
from scipy import stats
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)

color_l = '#cfcfcf'
color_r = '#8a8a8a'
muscles = ('Longis Cap', 'Longis Cerv', 'Sternocleid', 'Trapez up p')
params = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]
# bar_names = ['before', 'after', '30min', '60min', '6h', '24h']
# times_range = range(len(bar_names))
# bar_indicies = times_range
time_index = 0

dict_data = dict()
names = ['AllaP', 'DimaZh', 'ElenaS', 'ElenaYa', 'KatyaM', 'MaksimM', 'MarselV', 'VictoriaYa']

def merge(list_of_list):
	return sum(list_of_list, [])


def read_data(datapath):
	filenames = [name for name in os.listdir(f"{datapath}/") if name.endswith(".csv")]
	subjects = range(len(filenames)-1)
	for filename in filenames:
		log.info(f"Обработан файл {filename}")
		with open(f"{datapath}/{filename}", encoding='windows-1251') as file:
			# remove header
			header = file.readline().strip().split(";")[-5:]
			assert header == params, 'Проверь кол-во столбцов в файле'
			# get data
			for index, line in enumerate(file):
				# читает строку, раскидывает по переменным
				line = line.strip().replace(",", ".").split(";")
				name, time, pattern, muscle, side, *values = line  # *values = freq, stiff, decr, relax, creep

				# заполнение словаря
				if name not in dict_data:
					dict_data[name] = {}
				if muscle not in dict_data[name]:
					dict_data[name][muscle] = {}
				if side not in dict_data[name][muscle]:
					dict_data[name][muscle][side] = {time_index: {p: [] for p in params}}

				for p, v in zip(params, map(float, values)):
					if len(dict_data[name][muscle][side][time_index][p]) >= 6:
						dict_data[name][muscle][side][time_index][p] = []
					dict_data[name][muscle][side][time_index][p].append(v)
	return subjects


def plotting(savepath, subjects):
	# заполнение списков значениями показателей, взятыми у каждого человека за определенный период времени
	for param in params:
		all_data_left_mean = []
		all_data_right_mean = []
		all_data_left_se = []
		all_data_right_se = []
		for muscle in muscles:
			for side in "Left", "Right":
				if side == "Left":
					all_patients = [v[muscle][side][time_index][param] for v in dict_data.values()]
					all_data_left_mean.append(np.mean(all_patients))
					error = stats.sem(all_patients, axis=None)
					all_data_left_se.append(error)
				if side == "Right":
					all_patients = [v[muscle][side][time_index][param] for v in dict_data.values()]
					all_data_right_mean.append(np.mean(all_patients))
					error = stats.sem(all_patients, axis=None)
					all_data_right_se.append(error)

		plot_combo(all_data_left_mean, all_data_left_se, all_data_right_mean, all_data_right_se,
		           param=param, show=False, save_to=savepath, subjects=subjects)



def ANNplotting(savepath, subjects):
	# заполнение списков значениями показателей, взятыми у каждого человека за определенный период времени
	for param in params:
		all_data_left = []
		all_data_right = []
		for muscle in muscles:
			mean_left, se_left, mean_right, se_right = [None] * 4

			for side in "Left", "Right":
				if side == "Left":
					all_data_left.append([v[muscle][side][time_index][param] for v in dict_data.values()])
				if side == "Right":
					all_data_right.append([v[muscle][side][time_index][param] for v in dict_data.values()])

			mean_left = [np.mean(all_data_left[time_index][subject]) for subject in subjects]
			se_left = [stats.sem(all_data_left[time_index][subject]) for subject in subjects]
			mean_right = [np.mean(all_data_right[time_index][subject]) for subject in subjects]
			se_right = [stats.sem(all_data_right[time_index][subject]) for subject in subjects]


			# print(f"{muscle}, {side}, {param}, mean_left = {mean_left}, se_left = {se_left}, mean_right = {mean_right}, se_right = {se_right}")
			# plot(mean_left , se_left, mean_right, se_right, param=param, muscle=muscle, show=False,
			#            save_to=savepath, subjects=subjects)

			#for plotting by muscles
			print(len(all_data_left))
			exit()
			all_left_m = merge(all_data_left[0])
			all_right_m = merge(all_data_right[0])
			mean_left_m = np.mean(all_left_m[0])
			se_left_m = stats.sem(all_left_m[0], axis=1)
			print(se_left_m)
			mean_right_m = np.mean(all_right_m[0])
			print(mean_right_m)
			exit()
			se_right_m = stats.sem(all_right_m[0])
		plot_combo(mean_left_m, se_left_m, mean_right_m, se_right_m, param=param, muscle=muscle, show=False,
		                save_to=savepath, subjects=subjects)

		print(f"{muscle}, {side}, {param}, all_left_m = {all_left_m}, mean_left = {mean_left_m}, se_left = {se_left_m},"
		      f" mean_right = {mean_right_m}, se_right = {se_right_m}")

def merge(list_of_list):
	return sum(list_of_list, [])


def near_round(x, base=5.0):
	return base * np.ceil(x / base)

def plot2(mean_left, se_left, mean_right, se_right,
               param=None, muscle=None, show=False, save_to=None, subjects=None):
	"""
	Args:
		mean_left:
		se_left:
		mean_right:
		se_right:
		param:
		muscle:
		show:
		save_to:
		subjects:
	"""
	fig, ax = plt.subplots(figsize=(4, 3))
	# styles
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	for label in ax.get_xticklabels():
		label.set_fontsize(15)
	for label in ax.get_yticklabels():
		label.set_fontsize(15)
	for axis in 'bottom', 'left':
		ax.spines[axis].set_linewidth(1.5)
	ax.xaxis.set_tick_params(width=1.5)
	ax.yaxis.set_tick_params(width=1.5)

	# ticks
	max_val = max(max(mean_left), max(mean_right))
	if max_val <= 2:
		step = 0.5
	elif 2 < max_val <= 15:
		step = 1
	elif 15 < max_val <= 25:
		step = 5
	elif 25 < max_val <= 100:
		step = 10
	else:
		step = 100
	max_nearby = near_round(max_val, step)

	# plot data
	x = np.arange(1)
	width = 0.35
	ax.bar(x - width / 2, mean_left, width,
	       yerr=se_left, error_kw={'ecolor': '0.1', 'capsize': 3}, label='L', color=color_l)
	ax.bar(x + width / 2, mean_right, width,
	       yerr=se_right, error_kw={'ecolor': '0.1', 'capsize': 3}, label='R', color=color_r)
	last_ind = len(subjects)+1
	bar_names = range(1, last_ind, 1)

	# set limits and ticks
	ax.set_xticks(1)
	ax.set_xticklabels(1)
	if max_nearby <= 2:
		ax.set_yticks(np.arange(int(0), max_nearby + 0.01, step))
		ax.set_yticklabels(np.arange(int(0), max_nearby + 0.01, step))
	else:
		ax.set_yticks(range(0, int(max_nearby) + 1, step))
		ax.set_yticklabels(range(0, int(max_nearby) + 1, step))
	ax.set_ylim(0, max_nearby)

	# saving
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(f'{save_to}/{muscle}_{param}.png', format='png')
	if show:
		plt.show()
	plt.close()



def plot_combo(mean_left, se_left, mean_right, se_right,
               param=None, muscle=None, show=False, save_to=None, subjects=None):
	"""
	Args:
		mean_left:
		se_left:
		mean_right:
		se_right:
		param:
		muscle:
		show:
		save_to:
		subjects:
	"""
	fig, ax = plt.subplots(figsize=(4, 3))
	# styles
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	for label in ax.get_xticklabels():
		label.set_fontsize(15)
	for label in ax.get_yticklabels():
		label.set_fontsize(15)
	for axis in 'bottom', 'left':
		ax.spines[axis].set_linewidth(1.5)
	ax.xaxis.set_tick_params(width=1.5)
	ax.yaxis.set_tick_params(width=1.5)

	# ticks
	max_val = max(max(mean_left), max(mean_right))
	if max_val <= 2:
		step = 0.5
	elif 2 < max_val <= 15:
		step = 1
	elif 15 < max_val <= 25:
		step = 5
	elif 25 < max_val <= 100:
		step = 10
	else:
		step = 100
	max_nearby = near_round(max_val, step)

	# plot data
	x = np.arange(len(muscles))
	width = 0.35
	ax.bar(x - width / 2, mean_left, width,
	       yerr=se_left, error_kw={'ecolor': '0.1', 'capsize': 3}, label='L', color=color_l)
	ax.bar(x + width / 2, mean_right, width,
	       yerr=se_right, error_kw={'ecolor': '0.1', 'capsize': 3}, label='R', color=color_r)

	# set limits and ticks
	ax.set_xticks(range(len(muscles)))
	ax.set_xticklabels(muscles, rotation=90)
	if max_nearby <= 2:
		ax.set_yticks(np.arange(int(0), max_nearby + 0.01, step))
		ax.set_yticklabels(np.arange(int(0), max_nearby + 0.01, step))
	else:
		ax.set_yticks(range(0, int(max_nearby) + 1, step))
		ax.set_yticklabels(range(0, int(max_nearby) + 1, step))
	ax.set_ylim(0, max_nearby)

	# saving
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(f'{save_to}/ALL_{param}.png', format='png')
	if show:
		plt.show()
	plt.close()



def ANNplot_combo(mean_left, se_left, mean_right, se_right,
               param=None, muscle=None, show=False, save_to=None, subjects=None):
	"""
	Args:
		mean_left:
		se_left:
		mean_right:
		se_right:
		param:
		muscle:
		show:
		save_to:
		subjects:
	"""
	fig, ax = plt.subplots(figsize=(4, 3))
	# styles
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	for label in ax.get_xticklabels():
		label.set_fontsize(15)
	for label in ax.get_yticklabels():
		label.set_fontsize(15)
	for axis in 'bottom', 'left':
		ax.spines[axis].set_linewidth(1.5)
	ax.xaxis.set_tick_params(width=1.5)
	ax.yaxis.set_tick_params(width=1.5)

	# ticks
	max_val = max(max(mean_left), max(mean_right))
	if max_val <= 2:
		step = 0.5
	elif 2 < max_val <= 15:
		step = 1
	elif 15 < max_val <= 25:
		step = 5
	elif 25 < max_val <= 100:
		step = 10
	else:
		step = 100
	max_nearby = near_round(max_val, step)

	# plot data
	x = np.arange(len(subjects))
	width = 0.35
	ax.bar(x - width / 2, mean_left, width,
	       yerr=se_left, error_kw={'ecolor': '0.1', 'capsize': 3}, label='L', color=color_l)
	ax.bar(x + width / 2, mean_right, width,
	       yerr=se_right, error_kw={'ecolor': '0.1', 'capsize': 3}, label='R', color=color_r)
	last_ind = len(subjects)+1
	bar_names = range(1, last_ind, 1)

	# set limits and ticks
	ax.set_xticks(subjects)
	ax.set_xticklabels(bar_names)
	if max_nearby <= 2:
		ax.set_yticks(np.arange(int(0), max_nearby + 0.01, step))
		ax.set_yticklabels(np.arange(int(0), max_nearby + 0.01, step))
	else:
		ax.set_yticks(range(0, int(max_nearby) + 1, step))
		ax.set_yticklabels(range(0, int(max_nearby) + 1, step))
	ax.set_ylim(0, max_nearby)

	# saving
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(f'{save_to}/{muscle}_{param}.png', format='png')
	if show:
		plt.show()
	plt.close()


def main():
	datapath = 'C:/Users/exc24/PycharmProjects/test/neck'
	subjects = read_data(datapath)
	# savepath = 'C:/Users/exc24/PycharmProjects/test/neck/plot'
	savepath = 'C:/Users/exc24/PycharmProjects/test/neck/plot_with_names'

	plotting(savepath, subjects=subjects)


# print(dict_data, sep= '\n' )

if __name__ == '__main__':
	main()
