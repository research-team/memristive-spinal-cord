import pylab as plt
import scipy.io as sio

datas = {}
mat_data = sio.loadmat('C:/Users/Home/LAB/memristive-spinal-cord/analysis/Lavrov_data/SCI Rat-1_11-22-2016_RMG_40Hz_one_step')
tickrate = int(mat_data['tickrate'][0][0])
datas_max = []
datas_min = []
# Collect data
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
# if "Stim" not in data_title:
# 	datas[data_title] = mat_data['data'][0][data_start:data_end]
values = max(datas.values())
offset = 100
sliced_values = []
start = 188
kekdata = [x / tickrate for x in range(len(datas[mat_data['titles'][0]]))][188::100]
counter = len(kekdata)
for kek in kekdata:
	plt.axvline(x=kek, linestyle="--", color="gray")
print("counter = ", counter)
for j in range(counter - 1):
	for i in range(start, start + offset, 1):
		sliced_values.append(values[i])
	datas_max.append(max(sliced_values))
	datas_min.append(min(sliced_values))
	start += 100
	sliced_values.clear()
print("max = ", datas_max, len(datas_max))
print("min = ", datas_min, len(datas_min))

plt.legend()
plt.show()
