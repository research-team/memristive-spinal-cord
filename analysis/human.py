import numpy as np
import h5py
import logging
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import stats
import scipy.io as sio

def draw_channels(start, end, titles, k = 0, yticks = []):
    logger.info("channels")
    for s, e, t in zip(start, end, titles):
        # channels
        d = data[int(s):int(e)] + 5 * k
        if len(d) == 0:
            d = np.array([0] * 200) + 5 * k
        plt.plot(np.arange(len(d)) * 0.25, d)
        yticks.append(d[0])
        k += 1
    plt.yticks(yticks, titles)
    plt.show()

def draw_slices(start, end, titles, time, period, muscle):
    logger.info("slices")
    for s, e, t in zip(start, end, titles):
        # slices
        if t == muscle:
            logger.info("muscle is here")
            d = data[int(s):int(e)] # + 2 *k
            logger.info(len(d))
            f = 0
            for i in range(12):
                p = d[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
                plt.plot(np.arange(len(p)) * 0.25, p)
        plt.show()

#Start it up!
slice_height = 0.1
logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()
#mat_contents = sio.loadmat('../../RITM 14Ch + GND.mat')
mat_contents = sio.loadmat('../../04.29-07-R22-R-2+4-13+15-20Hz-4.mat')

for i in sorted(mat_contents.keys()):
    logger.info(i)
    logger.info(mat_contents[i])
    logger.info(len(mat_contents[i]))

starts = mat_contents['datastart']
ends = mat_contents['dataend']
logger.info(ends - starts)
data = mat_contents['data'][0]
titles = mat_contents['titles']
logger.info(len(data))

# constants
#start_time = 5005
#start_time = 8810
start_time = 14110
period = 50
#muscle_channel = "SOL L     "
#muscle_channel = "SOL L    "
#muscle_channel = "SOL R    "
#muscle_channel = 'TA L     '
muscle_channel = 'TA R     '
#muscle_channel = "Art short"

#for i in range(14, 16):
for i in range(1):
    start = starts[:, i]
    end = ends[:, i]
    # plt.subplot(len(starts), 1, (i+1))
    k = 0
    yticks = []
    draw_channels(start, end, titles)
    #draw_slices(start, end, titles, start_time, period, muscle_channel)
    # plt.savefig('./graphs/05.29-07-R23-R-AS{}.png'.format(i))
    plt.clf()
