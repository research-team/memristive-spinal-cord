import pylab as plt
import scipy.io as sio

datas = {}
mat_data = sio.loadmat('/home/alex/Downloads/fff.mat')

# Collect data
for index, data_title in enumerate(mat_data['titles']):
	data_start = int(mat_data['datastart'][index])-1
	data_end = int(mat_data['dataend'][index])

	if "Stim" not in data_title:
		datas[data_title] = mat_data['data'][0][data_start:data_end]

# Plot data
for data_title, mat_data in datas.items():
	plt.plot(range(len(mat_data)), mat_data, label=data_title)
	plt.xlim(0, len(mat_data))

plt.legend()
plt.show()
	
