import numpy
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import logging
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors


def slice_ees(data_array, slicing_index = 'Stim', data_index = 'RMG ', epsilon = .001) :
    logger.debug('Slicing')
    max_stim = max(data_array[slicing_index])
    list_of_maxs = [i for i, x in enumerate(data_array[slicing_index]) if x > max_stim-epsilon]
    logger.debug('number of maxs ' + str(len(list_of_maxs)))
    res = numpy.split(datas[data_index], list_of_maxs)
    logger.debug('number of slices ' + str(len(res)))
    return res

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)


logger = logging
logging.basicConfig(level=logging.DEBUG)
fig = plt.figure()
Axes3D(fig)
datas = {}
plt.matplotlib.pyplot.jet()

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
    #plt.plot(x, s, label='slice')
    #plt.xlim(0, x[-1])

#x_axes = [i / tick_rate * 1000 for i in range(len(s))]
#plt.plot(x_axes, slices)
#plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

xs = range(100)
zs = range(len(slices))
verts = []
for z in zs:
    ys = slices[z]
    verts.append(list(zip(xs, ys)))


poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'), cc('y')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, 10)
ax.set_ylabel('Y')
ax.set_ylim3d(-1, 4)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 1)

plt.show()


logger.info('End of processing')

