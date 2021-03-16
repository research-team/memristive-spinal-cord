import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

with open("../../../../Users/exc24/PycharmProjects/test/Dima.csv", encoding='utf-8') as file:
	file.readline()
	muscles = set()
	dict_data = dict()
	params = ["freq", "stiff", "decr", "relax", "creep"]
	k_freq, k_stiff, k_decr, k_relax, k_creep = "freq", "stiff", "decr", "relax", "creep"
	#    times = ['до', 'после', '30 мин', '60 мин', '6ч', '24ч']

	for index, line in enumerate(file):
		time_index = index // 132
		line = line.replace(",", ".").split(";")
		name, time, pattern, muscle, side, freq, stiff, decr, relax, creep = line
		muscles.add(muscle)
		if name not in dict_data:
			dict_data[name] = {}
		if muscle not in dict_data[name]:
			dict_data[name][muscle] = {}
		if side not in dict_data[name][muscle]:
			dict_data[name][muscle][side] = {t: {} for t in range(6)}
			for t in range(6):
				dict_data[name][muscle][side][t] = {p: [] for p in params}
		dict_data[name][muscle][side][time_index][k_freq].append(float(freq))
		dict_data[name][muscle][side][time_index][k_stiff].append(float(stiff))
		dict_data[name][muscle][side][time_index][k_decr].append(float(decr))
		dict_data[name][muscle][side][time_index][k_relax].append(float(relax))
		dict_data[name][muscle][side][time_index][k_creep].append(float(creep))

fr_mean = []
for i in range(7):
	fr = dict_data['Dima']['Achilles t']['Left'][i]['freq']
	fr_mean.append(fr)
fr_se = []
for i in range(7):
	frSE = stats.sem(fr)
	fr_se.append(frSE)

index = ['before', 'after', '30min', '60min', '6h', '24h']
plt.bar(index, [fr_mean], yerr=[fr_se], error_kw={'ecolor': '0.1', 'capsize': 6})
# plt.xticks(index+0.4,['A'])
plt.legend(loc=2)
plt.show()

# print(muscle)
# for time in range(6):
#     pass
# l np.mean()
# r np.mean()
# plt.plot(.....)
# plt.show()
# print(dict_data['Dima']['Achilles t']['Left'][1])

# print(*data, sep='\n')
# print(dict_data)
