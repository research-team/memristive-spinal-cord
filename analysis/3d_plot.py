import numpy
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import logging
from matplotlib.collections import PolyCollection
from analysis.eeslib import slice_ees


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
mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
#mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RMG_20Hz_one_step.mat')
#mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat')
#mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RTA_20Hz_one_step.mat')

tick_rate = int(mat_data['tickrate'][0][0])

logger.info('Loaded data')

for index, data_title in enumerate(mat_data['titles']):
    data_start = int(mat_data['datastart'][index]) - 1
    data_end = int(mat_data['dataend'][index])
    datas[data_title] = mat_data['data'][0][data_start:data_end]
    logger.debug('Collected data ' + data_title)

logger.info("Plotting data")

slices = slice_ees(datas, sorted(datas)[1], sorted(datas)[0])[1:5]
ax = fig.gca(projection='3d')

xs = plt.arange(0, 100/tick_rate*1000, (100/tick_rate*1000)/100)
zs = range(len(slices))
verts = []
for z in zs:
    ys = slices[z]
    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts, facecolors=(0.0, 0.0, 0.0, 0.0), edgecolors=['k'], closed=False)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X ms')
ax.set_xlim3d(0, len(xs)/tick_rate*1000)
ax.set_ylabel('Y slice #')
ax.set_ylim3d(-1, len(zs))
ax.set_zlabel('Z mV')
ax.set_zlim3d(-10, 5)
ax.set_title("Sliced EES EMG")

plt.show()

logger.info('End of processing')

