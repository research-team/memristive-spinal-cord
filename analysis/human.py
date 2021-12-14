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
lowcut = 30.0
highcut = 1000.0

v='bior4.4'
thres=[0.1,0.4,0.6]

def lowpassfilter(signal, thresh=0.4, wavelet=v):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, level=8,mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode='soft' ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def lowfilter(signal, N = 2, Wn = 0.08):
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

def get_indexes(start, end, titles, filter = False):
    k=0
    for s, e, t in zip(start, end, titles):
        if "TA L" in t:
            d = data[int(s+50*40):int(e)] + 2 * k
            if len(d) == 0:
                d = np.array([0] * 200)

            if filter:
                # plt.plot(np.arange(len(d)) * 0.25, d)
                # d_f = lowfilter(np.array(d))
                # plt.plot(np.arange(len(d_f)) * 0.25, d_f)
                # plt.plot(np.arange(len(d_f)) * 0.25, d_f-d)
                d = butter_bandpass_filter(np.array(d), lowcut, highcut, fs) + 2 * k
                d = lowfilter(np.array(d))
            indexes = argrelextrema(d, np.greater)[0]
            values = d[indexes]
            indexes = indexes[values > max(values)*0.745]
            values = values[values > max(values)*0.745]

            diff_steps = []
            diff_steps.append(True)
            for i in range (1, len(indexes)):
                if indexes[i] - indexes[i-1] > 1000/0.25:
                    diff_steps.append(True)
                else:
                    diff_steps.append(False)

            # diff_steps = np.diff(indexes, n=1)
            # diff_ms = np.append(diff_steps * 0.25, 10)

            indexes = indexes[diff_steps]
            values = values[diff_steps]
        k += 1
    return indexes, values

def draw_channels(start, end, titles, k = 0, filter = False):
    logger.info("channels")
    yticks = []
    titl = []
    indexes, values = get_indexes(start, end, titles, True)
    for s, e, t in zip(start, end, titles):
        # channels
        if "ACC" not in t and "GYRO" not in t and "MAG" not in t and "Art" not in t and "Channel" not in t:
            height = (np.max(data[int(s+50*40):int(e)]) - np.min(data[int(s+50*40):int(e)]))*0.3
            d = data[int(s+50*40):int(e)] + k*0.7
            if len(d) == 0:
                d = np.array([0] * 200) + k

            if filter:
                # plt.plot(np.arange(len(d)) * 0.25, d)
                # d_f = lowfilter(np.array(d))
                # plt.plot(np.arange(len(d_f)) * 0.25, d_f)
                # plt.plot(np.arange(len(d_f)) * 0.25, d_f-d)
                d = butter_bandpass_filter(np.array(d), lowcut, highcut, fs) + k*0.4
                d = lowfilter(np.array(d))
            print(len(indexes))

            # plt.scatter(indexes * 0.25, values)
            for i in range(1, len(indexes)):
                plt.plot(i*20000+np.arange(len(d[indexes[i-1]:indexes[i]])) * 0.25, d[indexes[i-1]:indexes[i]])
            # plt.plot(np.arange(len(d)) * 0.25, d)

            titl.append(t)
            yticks.append(d[0])
            k += 1
    plt.yticks(yticks, titl)
    plt.show()

def draw_slices(start, end, titles, time, period, muscle, filter = False):
    logger.info("slices")
    indexes, values = get_indexes(start, end, titles, True)
    plt.figure(figsize=(10, 20))

    starts = []
    #
    for j in range(0, len(indexes)-1):
        for s, e, t in zip(start, end, titles):
            if "Art 2" in t:
                d = data[int(s+50*40):int(e)]
                d = d[indexes[j]:indexes[j+1]]
                s = argrelextrema(d, np.greater)[0]# + 2 *k
                values = d[s]
                s = s[values > max(values)*0.745]
                starts.append(s[0]+indexes[j])
                print(f'ind - {indexes[j]}, start {s[0]} {starts[j]}')

    for j in range(0, len(indexes)-1):
        for s, e, t in zip(start, end, titles):
            # slices
            if "ACC" not in t and "GYRO" not in t and "MAG" not in t and "Channel" not in t:
                logger.info("muscle is here")
                d = data[int(s+50*40):int(e)] # + 2 *k
                if filter:
                    d = lowfilter(np.array(d))

                    d = butter_bandpass_filter(np.array(d), lowcut, highcut, fs)

                    # d_b = detrend(d_f)

                # d_f = d
                logger.info(len(d))
                f = 0
                plt.clf()
                slice_height = (np.max(d[indexes[j]:indexes[j+1]]) - np.min(d[indexes[j]:indexes[j+1]]))*0.25
                step_len = int((indexes[j+1]-indexes[j])*0.25/period)
                print(f'step len {step_len}')
                if step_len > 80:
                    step_len = 80
                for i in range(step_len):
                    # p = d[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
                    # plt.plot(np.arange(len(p)) * 0.25, p)
                    # p = d_f[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
                    # plt.plot(np.arange(len(p)) * 0.25, p)
                    # p = d_p[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
                    # plt.plot(np.arange(len(p)) * 0.25, p)
                    p = d[starts[j]+i*period*4:starts[j]+(i+1)*period*4] + slice_height *i
                    plt.plot(np.arange(len(p)) * 0.25, p)
                    # plt.legend(['Original','Filtered', 'R', 'Bandpass'])
                plt.savefig(f'/Users/sulgod/Desktop/graphs/new/{t}_time{starts[j]*0.25}_f.png')
        # plt.show()

#Start it up!
slice_height = 0.02
logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()
mat_contents = sio.loadmat('/Users/sulgod/Downloads/humandata_new/6v/17062021 2+5- 210ms 20hz 10.2ma  walk 3.mat')
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
start_time = 5395
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
    # draw_channels(start, end, titles, filter = True)
    draw_slices(start, end, titles, start_time, period, muscle_channel, filter = True)
    # plt.savefig('./graphs/05.29-07-R23-R-AS{}.png'.format(i))
    # plt.clf()

print(len(starts))
