import numpy as np
import h5py
import logging
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import stats
import scipy.io as sio

# constants
time = 5010
period = 50
#muscle = "SOL L     "
muscle = "SOL L    "
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

#for i in range(14, 16):
for i in range(1):
    start = starts[:, i]
    end = ends[:, i]
    # plt.subplot(len(starts), 1, (i+1))
    k = 0
    yticks = []
    for s, e, t in zip(start, end, titles):
        if t == muscle:
        	logger.info("YES")
        	d = data[int(s):int(e)] # + 2 *k
        	logger.info(len(d))
        	f = 0
        	for i in range(12):
        		p = d[time*4+i*period*4:time*4+(i+1)*period*4] + 0.2 *i
        		plt.plot(np.arange(len(p)) * 0.25, p)
        plt.show()
    #     d = data[int(s):int(e)] + 5 * k
    #     if len(d) == 0:
    #         d = np.array([0] * 200) + 5 * k
    #     plt.plot(np.arange(len(d)) * 0.25, d)
    #     yticks.append(d[0])
    #     k += 1
    # plt.yticks(yticks, titles)
    plt.show()
    # plt.savefig('./graphs/05.29-07-R23-R-AS{}.png'.format(i))
    plt.clf()
