import numpy
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import logging

def slice_ees(data_array, slicing_index = 'Stim', data_index = 'RMG ', epsilon = .001) :
    logger.debug('Slicing')
    max_stim = max(data_array[slicing_index])
    list_of_maxs = [i for i, x in enumerate(data_array[slicing_index]) if x > max_stim-epsilon]
    logger.debug('number of maxs ' + str(len(list_of_maxs)))
    res = numpy.split(datas[data_index], list_of_maxs)
    logger.debug('number of slices ' + str(len(res)))
    return res


logger = logging
logging.basicConfig(level=logging.DEBUG)
fig = plt.figure()
Axes3D(fig)
datas = {}
plt.matplotlib.pyplot.viridis()

logger.debug('Completed setup')
'''
RTA right  tibialis anterior (flexor)
RMG right adductor magnus (extensor)
'''
mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
# mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RMG_20Hz_one_step.mat')
# mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat')
# mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RTA_20Hz_one_step.mat')

tick_rate = int(mat_data['tickrate'][0][0])

logger.info('Loaded data')
for index, data_title in enumerate(mat_data['titles']):
    data_start = int(mat_data['datastart'][index]) - 1
    data_end = int(mat_data['dataend'][index])
    #if "RMG" not in data_title:
    datas[data_title] = mat_data['data'][0][data_start:data_end]
    logger.debug('Collected data ' + data_title)

logger.info("Plot data")
#for data_title, data in datas.items():
    #logger.debug('Plotting title ' + data_title)
    #x = [i / tick_rate for i in range(len(data))]
    #plt.plot(x, data, label=data_title)
    #plt.xlim(0, x[-1])
#plt.show()

slices = slice_ees(datas)

for s in slices[:-1]:
    logger.debug('Plotting slices ' + str(len(s)))
    x = [i for i in range(len(s))]
    x = [i / tick_rate * 1000 for i in range(len(s))]
    plt.plot(x, s, label='slice')
    plt.xlim(0, x[-1])

#x_axes = [i / tick_rate * 1000 for i in range(len(s))]
#plt.plot(x_axes, slices)
plt.show()

logger.info('End of processing')

