import numpy as np
import pylab as plt
import scipy.io as sio

datas = {}
mat_data = sio.loadmat(
	'Lavrov_data/SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat')
tickrate = int(mat_data['tickrate'][0][0])

# Collect data
for index, data_title in enumerate(mat_data['titles']):
	data_start = int(mat_data['datastart'][index])-1
	data_end = int(mat_data['dataend'][index])
	# if "Stim" not in data_title:
	datas[data_title] = mat_data['data'][0][data_start:data_end]
for data_title, mat_data in datas.items():
	mat_data = mat_data[188:]
	index = 1
	for offs_iter in range(len(mat_data))[::100]:
		plt.plot([x * 0.00025 for x in range(len(mat_data[offs_iter + 30:offs_iter + 100]))],
		         [y - 1 * index for y in mat_data[offs_iter + 30:offs_iter + 100]],
		         label=data_title, color='gray')
		index += 1
for kek in [x * 0.00025 for x in range(len(mat_data))][::100]:
	plt.axvline(x=kek, linestyle="--", color="gray")
plt.xlim(0, len(mat_data) * 0.00025)

# print(datas.items())
# Plot data
# for data_title, data in datas.items():
#   x = [i / tickrate * 1000 for i in range(len(data))]
# plt.plot(x, data, label=data_title)
# plt.xlim(0, x[-1])

plt.legend()
plt.show()