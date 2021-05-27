import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter


fs = 5000.0
lowcut = 20.0
highcut = 1000.0



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


def moving_average(data, weight):
    return np.convolve(data, np.ones(weight), 'valid') / weight


def read_data(datapath):
    filenames = [name[:-4] for name in os.listdir(f"{datapath}") if name.endswith(".mat")]
    for filename in filenames:
        # filename = 'on the left side SS1'
        dict_data = sio.loadmat(f'{datapath}/{filename}')
        save_folder = f'{datapath}/render/{filename}'

        raw_data = dict_data['data'][0]

        starts = [int(d[0]) for d in dict_data['datastart']]
        ends = [int(d[0]) for d in dict_data['dataend']]
        titles = dict_data['titles']
        dx = 1 / dict_data['samplerate'][0][0]

        muscles = {}
        for t, s, e in zip(titles, starts, ends):
            muscles[t] = raw_data[s:e]

        for title, data in muscles.items():
            # original_render(title, data, save_folder, dx)
            # original_render_html(title, data, save_folder, dx)
            smoothed_render(title, data, save_folder, dx)
            # smoothed_render_html(title, data, save_folder, dx)


def draw_slices(zip_file, frequency, dx, save_folder, title, show=False):
    d = np.array(zip_file)[:, 0]
    shift_max = max(d)
    shift_min = abs(min(d))
    shift = 0.01  # max(shift_min, shift_max)

    for i in range(100):
        start = int(frequency * i / dx)
        end = int(frequency * (i + 1) / dx)
        plt.plot(d[start:end] + (i * shift))
    plt.savefig(f'{save_folder}/{title}_slices.png', format='png')
    if title == 'Art 1' or title == 'Art 2':
        plt.show()
    plt.close()


def smoothed_render(title, data, save_folder, dx, show=False):
    plt.suptitle(f'{title} (smoothed)')
    data = butter_bandpass_filter(np.array(data), lowcut, highcut, fs)
    x = np.arange(len(data)) * dx
    zip_data_x = list(zip(data, x))

    plt.plot(x, data, color='g')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/{title}_smoothed.png', format='png')
    if title == 'Art 1' or title == 'Art 2':
        plt.show()
    plt.close()



    N = int(4000 * (len(data)) * dx)
    yf = rfft(data)
    xf = rfftfreq(N, 1 / 4000)
    dict_val = dict(zip(yf, xf))
    max_val = max(dict_val.keys())
    max_x = dict_val[max_val]

    plt.plot(xf, np.abs(yf))
    plt.show()
    print(max_x)
    exit()



    draw_slices(zip_file=zip_data_x, dx=dx, frequency=0.0255, save_folder=save_folder, title=title)


def main():
    datapath = '/home/b-rain/rhythmic'
    read_data(datapath)


if __name__ == '__main__':
    main()
