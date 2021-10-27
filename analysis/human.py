import numpy as np
import h5py
import logging
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import stats
from scipy.signal import detrend
import scipy.io as sio
import pywt
import scipy
import math

from scipy.signal import butter, lfilter, filtfilt

fs = 5000.0
lowcut = 40.0
highcut = 1000.0

v='bior4.4'
thres=[0.1,0.4,0.6]

def lowpassfilter(signal, thresh=0.4, wavelet=v):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, level=8,mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode='soft' ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def lowfilter(signal, N = 2, Wn = 0.1):
    B, A = butter(N, Wn, output='ba')
    signalf = filtfilt(B, A, signal)
    return signalf

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def draw_channels(start, end, titles, k = 0, yticks = [], filer = False):
    logger.info("channels")
    titl = []
    for s, e, t in zip(start, end, titles):
        # channels
        if "ACC" not in t and "GYRO" not in t and "MAG" not in t:
            d = data[int(s+50*40):int(e)] + 2 * k
            if len(d) == 0:
                d = np.array([0] * 200) + 2 * k

            if filter:
                # plt.plot(np.arange(len(d)) * 0.25, d)
                # d_f = lowfilter(np.array(d))
                # plt.plot(np.arange(len(d_f)) * 0.25, d_f)
                # plt.plot(np.arange(len(d_f)) * 0.25, d_f-d)
                d = butter_bandpass_filter(np.array(d), lowcut, highcut, fs) + 2 * k
            plt.plot(np.arange(len(d)) * 0.25, d)
            titl.append(t)
            yticks.append(d[0])
            k += 1
    plt.yticks(yticks, titl)
    plt.show()

def draw_slices(start, end, titles, time, period, muscle, filer = False):
    logger.info("slices")
    for s, e, t in zip(start, end, titles):
        # slices
        # if t == muscle:
        logger.info("muscle is here")
        d = data[int(s):int(e)] # + 2 *k
        if filter:
            d_b = butter_bandpass_filter(np.array(d), lowcut, highcut, fs)
            d_f = lowfilter(np.array(d_b))

            # d_b = detrend(d_f)

        # d_f = d
        logger.info(len(d))
        f = 0
        plt.clf()
        for i in range(12):
            # p = d[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
            # plt.plot(np.arange(len(p)) * 0.25, p)
            # p = d_f[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
            # plt.plot(np.arange(len(p)) * 0.25, p)
            # p = d_p[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
            # plt.plot(np.arange(len(p)) * 0.25, p)
            p = d_f[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
            plt.plot(np.arange(len(p)) * 0.25, p)
            # plt.legend(['Original','Filtered', 'R', 'Bandpass'])
        plt.savefig(f'/Users/sulgod/Desktop/graphs/new/{t}_time{time}_f.png')
        # plt.show()

#Start it up!
slice_height = 0.025
logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()
mat_contents = sio.loadmat('/Users/sulgod/Downloads/humandata_new/7v/Vertical walk SS.mat')
# mat_contents = sio.loadmat('/Users/sulgod/Downloads/131219.mat')

for i in sorted(mat_contents.keys()):
    logger.info(i)
    logger.info(mat_contents[i])
    logger.info(len(mat_contents[i]))

starts = mat_contents['datastart']
print(len(starts))
ends = mat_contents['dataend']
logger.info(ends - starts)
data = mat_contents['data'][0]
titles = mat_contents['titles']
logger.info(len(data))

# constants
#start_time = 5005
#start_time = 8810
start_time = 1470
period = 50
muscle_channel = "SOL L    "
# muscle_channel = "RF R     "
# muscle_channel = "BF L     "
#muscle_channel = 'TA L     '
# muscle_channel = 'TA R     '
#muscle_channel = "Art short"

#for i in range(14, 16):
for i in range(1):
    start = starts[:, i]
    end = ends[:, i]
    # plt.subplot(len(starts), 1, (i+1))
    k = 0
    yticks = []
    # draw_channels(start, end, titles, filer = False)
    draw_slices(start, end, titles, start_time, period, muscle_channel, filer = False)
    # plt.savefig('./graphs/05.29-07-R23-R-AS{}.png'.format(i))
    # plt.clf()

print(len(starts))
