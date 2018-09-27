import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import warnings
fig = plt.figure()
Axes3D(fig)

datas = {}
mat_data = sio.loadmat('C:/Users/Home/Desktop/учебники/Нейролаб/данные Лаврова/SCI Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
tickrate = int(mat_data['tickrate'][0][0])
for index, data_title in enumerate(mat_data['titles']):
	data_start = int(mat_data['datastart'][index])-1
	data_end = int(mat_data['dataend'][index])

	if "Stim" not in data_title:
		datas[data_title] = mat_data['data'][0][data_start:data_end]

# Plot data
for data_title, data in datas.items():
	x = [i / tickrate for i in range(len(data))]
	plt.plot(x, data, label=data_title)
	plt.xlim(0, x[-1])

plt.show()