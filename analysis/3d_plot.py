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
'''
RTA right  tibialis anterior (flexor)
RMG right adductor magnus (extensor)
'''
mat_data = sio.loadmat('../../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
# mat_data = sio.loadmat('../../bio-data/SCI_Rat-1_11-22-2016_RMG_20Hz_one_step.mat')
# mat_data = sio.loadmat('../../bio-data/SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat')
# mat_data = sio.loadmat('../../bio-data/SCI_Rat-1_11-22-2016_RTA_20Hz_one_step.mat')

tick_rate = int(mat_data['tickrate'][0][0])

logger.info('Loaded data')
for index, data_title in enumerate(mat_data['titles']):
    data_start = int(mat_data['datastart'][index]) - 1
    data_end = int(mat_data['dataend'][index])
    # if "Stim" not in data_title:
    datas[data_title] = mat_data['data'][0][data_start:data_end]
    logger.debug('Collected data ' + data_title)

logger.info("Plot data")
for data_title, data in datas.items():
    logger.debug('Ploting data ' + data_title)
    x = [i / tick_rate for i in range(len(data))]
    plt.plot(x, data, label=data_title)
    plt.xlim(0, x[-1])
#plt.show()

logger.info('Slicing')
max_stim = max(datas['Stim'])
number_of_maxs = sum(d > max_stim-.1 for d in datas['Stim'])
logger.debug('number of maxs ' + str(number_of_maxs) + ' ' + str(max_stim))

logger.info('End of processing')
