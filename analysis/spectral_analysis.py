import pylab as plt
from analysis.real_data_slices import read_data
raw_real_data = read_data('../bio-data//SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
real_data = max(raw_real_data['data'])
datas = {}
FD = 4000 #frequency of discretization, counts per second
N = len(real_data) #lenght of the input
tickrate = int(raw_real_data['tickrate'][0][0])
for index, data_title in enumerate(raw_real_data['titles']):
	data_start = int(raw_real_data['datastart'][index])-1
	data_end = int(raw_real_data['dataend'][index])

	#if "Stim" not in data_title:
	datas[data_title] = raw_real_data['data'][0][data_start:data_end]

# Plot data
for data_title, data in datas.items():
	x = [i / tickrate for i in range(len(data))]
	plt.plot(x, data, label=data_title)
	plt.xlim(0, x[-1])
for kek in [x / tickrate for x in range(len(datas[raw_real_data['titles'][0]]))][188::100]:
	plt.axvline(x=kek, linestyle="--", color="gray")
plt.legend()
plt.show()