import inspect
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import plotly

def moving_average(data, weight):
    return np.convolve(data, np.ones(weight), 'valid') / weight


def read_data(datapath):
    filename = f'{datapath}/on the left side SS3'
    dict_data = sio.loadmat(filename)
    save_folder = f'{datapath}/render'

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


def original_render_html(title, data, save_folder, dx):
    trace1 = go.Scatter(x=np.arange(len(data)) * dx,
                        y=data,
                        mode="lines",
                        name="citations",
                        marker=dict(color='rgba(16, 112, 2, 0.8)'))

    layout = dict(title=title,
                  xaxis=dict(title='Time', ticklen=5, zeroline=False))
    fig = dict(data=trace1, layout=layout)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    configs = {'scrollZoom': True, 'displaylogo': False, 'displayModeBar': True}
    plotly.offline.plot(fig, config=configs, filename=f'{save_folder}/{title}_original.html', auto_open=False)


def original_render(title, data, save_folder, dx):
    plt.suptitle(f'{title} (original)')
    x = np.arange(len(data)) * dx
    plt.plot(x, data)
    plt.axhline(y=0, lw=1, ls='--', color='gray')
    plt.grid(axis='x')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/{title}_original.png', format='png')
    plt.close()


def smoothed_render(title, data, save_folder, dx):
    plt.suptitle(f'{title} (smoothed)')
    data = moving_average(data, 150)
    x = np.arange(len(data)) * dx
    plt.plot(x, data, color='g')
    peaks = []

    for index, point in enumerate(data):
        difference = []
        min_diff = None
        if point > 4:
            if not peaks:
                peaks.append(index * dx)
            if index * dx - peaks[-1] > 1:
                peaks.append(index * dx)
        # if peaks:
        #     for i, peak in enumerate(peaks[:-1]):
        #         dif = peaks[i + 1] - peaks[i]
        #         difference.append(dif)
        #     if difference:
        #         if len(difference) > 1:
        #             min_diff = min(difference)
        #         else:
        #             min_diff = difference[0]
    print(f'peaks = {peaks} \n')
    # print(f'difference = {difference} \n')
    # print(f'min_diff = {min_diff} \n')

    for i, p in enumerate(peaks):
        # line = peaks[0] + min_diff * i
        line = peaks[i] - 0.01
        plt.axvline(x=line, color='r')

    plt.xlabel('Seconds')
    plt.ylabel('Volts')
    plt.axhline(y=0, lw=1, ls='--', color='gray')
    plt.grid(axis='x')
    plt.show()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/{title}_smoothed.png', format='png')
    plt.close()


def smoothed_render_html(title, data, save_folder, dx):
    data = moving_average(data, 150)
    trace1 = go.Scatter(x=np.arange(len(data)) * dx,
                        y=data,
                        mode="lines",
                        name="citations",
                        marker=dict(color='rgba(16, 112, 2, 0.8)'))

    layout = dict(title=title,
                  xaxis=dict(title='Time', ticklen=5, zeroline=False))
    fig = dict(data=trace1, layout=layout)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    configs = {'scrollZoom': True, 'displaylogo': False, 'displayModeBar': True}
    plotly.offline.plot(fig, config=configs, filename=f'{save_folder}/{title}_smoothed.html', auto_open=False)

def draw_slices(start, end, titles, time, period, muscle):
    logger.info("slices")
    for s, e, t in zip(start, end, titles):
        # slices
        # if t == muscle:
        logger.info("muscle is here")
        d = data[int(s):int(e)] # + 2 *k
        d_f = butter_bandpass_filter(np.array(d), lowcut, highcut, fs)
        # d_f = d
        logger.info(len(d))
        f = 0
        plt.clf()
        for i in range(8):
            p = d_f[time*4+i*period*4:time*4+(i+1)*period*4] + slice_height *i
            plt.plot(np.arange(len(p)) * 0.25, p)
        plt.savefig(f'/Users/sulgod/Desktop/graphs/1312/{t}_time{time}_f.png')
        # plt.show()

def main():
    datapath = '/home/b-rain/rhythmic'
    filename = f'{datapath}/on the left side SS3'
    read_data(datapath)


if __name__ == '__main__':
    main()
