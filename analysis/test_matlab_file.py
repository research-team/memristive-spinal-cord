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


def calc_frequency(data, samplerate, save_folder, show=False):
    for title, art_data in data.items():
        sampling_frequency = samplerate
        sampling_size = len(art_data)  # get size (length) of the data

        # frequency domain representation
        fourier_transform = np.fft.fft(art_data) / sampling_size  # normalize amplitude
        fourier_transform = abs(fourier_transform[range(int(sampling_size / 2))])  # exclude sampling frequency

        # remove the mirrored part of the FFT
        values = np.arange(int(sampling_size / 2))
        time_period = sampling_size / sampling_frequency
        frequencies = values / time_period

        # cuts frequency and calc max
        mask = (frequencies <= 40) & (frequencies >= 20)
        frequencies = frequencies[mask]
        fourier_transform = fourier_transform[mask]
        fourier_dict = dict(zip(frequencies, fourier_transform))
        max_frequency = list(fourier_dict.keys())[list(fourier_dict.values()).index(max(fourier_transform))]

        # plotting
        figure, axis = plt.subplots(2, 1)
        axis[1].set_title('Fourier transform depicting the frequency components')
        axis[1].plot(frequencies, fourier_transform)
        axis[1].set_xlabel('Frequency')
        axis[1].set_ylabel('Amplitude')

        # squeeze plot
        plt.tight_layout()
        if show:
            plt.show()
        plt.close()

    return 1 / max_frequency


def read_data(datapath):
    filenames = [name[:-4] for name in os.listdir(f"{datapath}") if name.endswith(".mat")]
    for filename in filenames:
        # filename = 'on the left side SS2'
        dict_data = sio.loadmat(f'{datapath}/{filename}')
        save_folder = f'{datapath}/render/{filename}'

        raw_data = dict_data['data'][0]
        samplerate = int(dict_data['samplerate'][0][0])

        starts = [int(d[0]) for d in dict_data['datastart']]
        ends = [int(d[0]) for d in dict_data['dataend']]
        titles = dict_data['titles'][:-2]
        arts_titles = dict_data['titles'][-2:]
        dx = 1 / samplerate

        arts = {}
        for t, s, e in zip(arts_titles, starts[-2:], ends[-2:]):
            arts[t] = raw_data[s:e]
        frequency = calc_frequency(data=arts, samplerate=samplerate,save_folder=save_folder, show=False)

        muscles = {}
        for t, s, e in zip(titles, starts, ends):
            muscles[t] = raw_data[s:e]

        for title, data in muscles.items():
            # smoothed_render(title=title, data=data, save_folder=save_folder, dx=dx, frequency=frequency, show=True)
            smoothed_render(title=title, data=data, save_folder=save_folder, dx=dx, frequency=frequency, show=False)


def draw_slices(zip_file, frequency, dx, save_folder, title, show=False):
    d = np.array(zip_file)[:, 0]
    shift_max = max(d)
    shift_min = abs(min(d))
    shift = 0.01  # max(shift_min, shift_max)

    for i in range(50):
        start = int(frequency * i / dx)
        end = int(frequency * (i + 1) / dx)
        plt.plot(d[start:end] + (i * shift))
    plt.savefig(f'{save_folder}/{title}_slices.png', format='png')
    if title == 'Art 1' or title == 'Art 2':
        plt.show()
    plt.close()


def smoothed_render(title, data, save_folder, dx, frequency, show=False):
    plt.suptitle(f'{title}')
    data = butter_bandpass_filter(np.array(data), lowcut, highcut, fs)
    x = np.arange(len(data)) * dx
    zip_data_x = list(zip(data, x))

    plt.plot(x, data, color='g')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/{title}_smoothed.png', format='png')
    if show:
        plt.show()
    plt.close()

    # if title == 'Art 1' or title == 'Art 2':
    #     frequency = calc_frequency(data=data, samplerate=4000, show=True)
    #     print(frequency)
    # draw_slices(zip_file=zip_data_x, dx=dx, frequency=frequency, save_folder=save_folder, title=title, show=True)
    draw_slices(zip_file=zip_data_x, dx=dx, frequency=frequency, save_folder=save_folder, title=title, show=False)
    # else:
    #     frequency = 0.0025


def main():
    datapath = '/home/b-rain/rhythmic'
    read_data(datapath)


if __name__ == '__main__':
    main()
