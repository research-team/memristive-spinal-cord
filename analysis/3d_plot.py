import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import logging

logger = logging
logging.basicConfig(level=logging.DEBUG)
fig = plt.figure()
Axes3D(fig)
datas = {}

logger.debug('Completed setup')
mat_data = sio.loadmat('../../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
tick_rate = int(mat_data['tickrate'][0][0])

logger.info('Loaded data')
for index, data_title in enumerate(mat_data['titles']):
	data_start = int(mat_data['datastart'][index])-1
	data_end = int(mat_data['dataend'][index])
	# if "Stim" not in data_title:
	logger.debug('data title ' + data_title + " " + str(index))
	datas[data_title] = mat_data['data'][0][data_start:data_end]

logger.info("Plot data")
for data_title, data in datas.items():
	logger.debug('data title ' + data_title)
	x = [i / tick_rate for i in range(len(data))]
	plt.plot(x, data, label=data_title)
	plt.xlim(0, x[-1])
plt.show()

logger.info("End of processing")
