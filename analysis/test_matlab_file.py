import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly

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
            smoothed_render(title, data, save_folder, dx, show=True)
            # smoothed_render_html(title, data, save_folder, dx)


# def original_render_html(title, data, save_folder, dx):
#     trace1 = go.Scatter(x=np.arange(len(data)) * dx,
#                         y=data,
#                         mode="lines",
#                         name="citations",
#                         marker=dict(color='rgba(16, 112, 2, 0.8)'))
#
#     layout = dict(title=title,
#                   xaxis=dict(title='Time', ticklen=5, zeroline=False))
#     fig = dict(data=trace1, layout=layout)
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     configs = {'scrollZoom': True, 'displaylogo': False, 'displayModeBar': True}
#     plotly.offline.plot(fig, config=configs, filename=f'{save_folder}/{title}_original.html', auto_open=False)


# def original_render(title, data, save_folder, dx):
#     plt.suptitle(f'{title} (original)')
#     x = np.arange(len(data)) * dx
#     plt.plot(x, data)
#     plt.axhline(y=0, lw=1, ls='--', color='gray')
#     plt.grid(axis='x')
#
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     plt.savefig(f'{save_folder}/{title}_original.png', format='png')
#     plt.close()


def draw_slices(zip_file, frequency, dx, save_folder, title, show=False):
    # d = np.array(zip_file)[:, 0][1000:]
    d = np.array(zip_file)[:, 0]
    shift_max = max(d)
    shift_min = abs(min(d))
    shift = 0.01#max(shift_min, shift_max)

    for i in range(10):
        start = int(frequency * i / dx)
        end = int(frequency * (i + 1) / dx)
        plt.plot(d[start:end] + (i * shift))
    plt.savefig(f'{save_folder}/{title}_slices.png', format='png')
    if show:
        plt.show()
    plt.close()


def smoothed_render(title, data, save_folder, dx, show=False):
    plt.suptitle(f'{title} (smoothed)')
    data = butter_bandpass_filter(np.array(data), lowcut, highcut, fs)
    #moving_average(data, 150)
    x = np.arange(len(data)) * dx
    zip_data_x = list(zip(data, x))

    plt.plot(x, data, color='g')
    # peaks = []

    # for index, point in enumerate(data):
    # difference = []
    # min_diff = None
    # if point > 3:
    #     if not peaks:
    #         peaks.append(index * dx)
    #     if index * dx - peaks[-1] > 1:
    #         peaks.append(index * dx)
    # if peaks:
    #     for i, peak in enumerate(peaks[:-1]):
    #         dif = peaks[i + 1] - peaks[i]
    #         difference.append(dif)
    #     if difference:
    #         if len(difference) > 1:
    #             min_diff = min(difference)
    #         else:
    #             min_diff = difference[0]
    # print(f'peaks = {peaks} \n')
    # print(f'difference = {difference} \n')
    # print(f'min_diff = {min_diff} \n')

    # for i in range(10):
    #     line = i * 1.7
    #     plt.axvline(x=line, color='r')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/{title}_smoothed.png', format='png')
    if show:
        plt.show()
    plt.close()

    draw_slices(zip_file=zip_data_x, dx=dx, frequency=0.0255, save_folder=save_folder, title=title, show=True)


# def smoothed_render_html(title, data, save_folder, dx):
#     data = moving_average(data, 150)
#     trace1 = go.Scatter(x=np.arange(len(data)) * dx,
#                         y=data,
#                         mode="lines",
#                         name="citations",
#                         marker=dict(color='rgba(16, 112, 2, 0.8)'))
#
#     layout = dict(title=title,
#                   xaxis=dict(title='Time', ticklen=5, zeroline=False))
#     fig = dict(data=trace1, layout=layout)
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     configs = {'scrollZoom': True, 'displaylogo': False, 'displayModeBar': True}
#     plotly.offline.plot(fig, config=configs, filename=f'{save_folder}/{title}_smoothed.html', auto_open=False)


def main():
    datapath = '/home/b-rain/rhythmic'
    read_data(datapath)


if __name__ == '__main__':
    main()
