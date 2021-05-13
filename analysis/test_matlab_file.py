import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import plotly


def moving_average(data, weight):
    return np.convolve(data, np.ones(weight), 'valid') / weight


datapath = 'C:/Users/Ann/PycharmProjects/test_matlab/data'


def read_data(filename):
    filename = f'{datapath}/on the left side SS3'
    dict_data = sio.loadmat(filename)
    save_folder = f'{datapath}/render data2'

    raw_data = dict_data['data'][0]
    # by [:-2] we omit the last redundant art1/art2 data
    starts = [int(d[0]) for d in dict_data['datastart']][:-2]
    ends = [int(d[0]) for d in dict_data['dataend']][:-2]
    titles = dict_data['titles'][:-2]
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
        if peaks:
            for i, peak in enumerate(peaks[:-1]):
                dif = peaks[i+1] - peaks[i]
                difference.append(dif)
            if difference:
                if len(difference) > 1:
                    min_diff = min(difference)
                else:
                    min_diff = difference[0]
    print(f'peaks = {peaks}')
    print(f'difference = {difference}')
    print(f'min_diff = {min_diff}')

    for i, p in enumerate(peaks):
        wtf = peaks[0] + min_diff * i
        plt.axvline(x=wtf, color='r')

    plt.xlabel('Seconds')
    plt.ylabel('Volts')
    plt.axhline(y=0, lw=1, ls='--', color='gray')
    plt.grid(axis='x')
    if 'TA R' in title:
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


def main():
    datapath = 'C:/Users/Ann/PycharmProjects/test_matlab/data'
    filename = f'{datapath}/on the left side SS3'
    read_data(filename)


if __name__ == '__main__':
    main()
