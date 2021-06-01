import os
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import butter, lfilter, argrelextrema

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
        plt.suptitle('Fourier transform depicting the frequency components')
        plt.plot(frequencies, fourier_transform)
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency ")

        # squeeze plot
        plt.tight_layout()
        if show:
            plt.show()
        plt.close()

    return max_frequency


def read_data(datapath):
    filenames = [name[:-4] for name in os.listdir(f"{datapath}") if name.endswith(".mat")]
    for filename in filenames:
        filename = 'on the right side SS2'
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
        frequency = calc_frequency(data=arts, samplerate=samplerate, save_folder=save_folder, show=True)

        for title, data in arts.items():
            smoothed_render(title=title, data=data, save_folder=save_folder, dx=dx, frequency=frequency,
                            show=True)

        muscles = {}
        for t, s, e in zip(titles, starts, ends):
            muscles[t] = raw_data[s:e]

        for title, data in muscles.items():
            smoothed_render(title=title, data=data, save_folder=save_folder, dx=dx, frequency=frequency,
                            show=True)


def draw_slices(zip_data_duration, frequency, duration, dx, save_folder, title, show=False):
    d = np.array(zip_data_duration)[:, 0]
    slice_duration = 1 / frequency
    number_of_slices = math.floor(max(duration) / slice_duration)
    number_of_picture = (number_of_slices // 100 + 1)

    shift = 0.01  # max(shift_min, shift_max)

    start = 0
    delta = abs(0 - int(slice_duration * (1 / dx)))
    crop = 0
    for k in range(number_of_picture)[1:]:
        for i in range(100):
            # end = math.floor(start + delta - crop)
            start = int(slice_duration * i * k / dx)
            end = int((slice_duration * (i * k + 1) / dx))
            # crop = end * 0.005

            plt.suptitle(f'{title} slices ({k} part)')
            plt.plot(d[start:end] + (i * shift))
            plt.ylabel("Voltage")
            plt.xlabel("Time ")
            plt.savefig(f'{save_folder}/{title}_slices_{k}_part.png', format='png')

            start = end
        if show:
            plt.show()
        plt.close()
    plt.close()


def smoothed_render(title, data, save_folder, dx, frequency, show=False):
    plt.suptitle(f'{title}')
    data = butter_bandpass_filter(np.array(data), lowcut, highcut, fs)
    duration = np.arange(len(data)) * dx
    zip_data_duration = list(zip(data, duration))

    plt.plot(duration, data, color='g')
    plt.ylabel("Voltage")
    plt.xlabel("Time (sec)")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/{title}_smoothed.png', format='png')
    if show:
        plt.show()
    plt.close()

    draw_slices(zip_data_duration=zip_data_duration, duration=duration, dx=dx, frequency=frequency,
                save_folder=save_folder,
                title=title, show=False)  # show=False)

def smoothed_render(title, data, save_folder, dx, frequency, show=False):
    fs = 4000.0
    lowcut = 20.0
    highcut = 1000.0

    plt.suptitle(f'{title}')
    data = butter_bandpass_filter(np.array(data), lowcut, highcut, fs)
    duration = np.arange(len(data)) * dx
    zip_data_duration = list(zip(data, duration))

    # here is not the best code, but the most short and stable
    # get the 1st derivative
    diff = np.diff(data, n=1)
    slices = int(len(data) / (1 / frequency / dx)) + 1
    # find the extrema
    extrema_index = argrelextrema(diff, np.less)[0]
    extrema_vals = diff[extrema_index]

    plt.plot(diff)
    plt.plot(extrema_index, extrema_vals, '.', color='r')
    top = np.array(sorted(zip(extrema_index, extrema_vals), key=lambda x: x[1])[:slices])

    plt.plot(top[:, 0], top[:, 1], '.', color='green')
    plt.show()

    extrema = np.stack((extrema_index, extrema_vals), axis=-1)

    extrema.view('i8,f8').sort(order=['f1', 'f0'], axis=0)
    print(extrema)
    # exit()

    # let numpy guess the type with dtype=None
    # access columns by name
    # sort column 1 and column 0
    indicies = np.lexsort((extrema_index, extrema_vals), axis=0)
    print(extrema_index[indicies])
    print(extrema_vals[indicies])
    # exit()

    top = np.argsort(diff)[:slices]
    plt.plot(diff)
    plt.plot(top, diff[top], '.', color='r')
    plt.show()
    # print(d)
    # exit()

    mask = diff[extrema_index] < -0.03
    extrema_index = extrema_index[mask]
    plt.plot(diff)
    plt.plot(extrema_index, diff[extrema_index], '.', color='r')
    for index, (start, end) in enumerate(zip(extrema_index, extrema_index[1:])):
        d = data[start - 10:end - 10] + (index * 0.001)
        t = np.arange(len(d)) * dx * 1000
        plt.plot(t, d)
    plt.show()

    plt.plot(duration, data, color='g')
    plt.ylabel("Voltage")
    plt.xlabel("Time (sec)")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f'{save_folder}/{title}_smoothed.png', format='png')
    if show:
        plt.show()
    plt.close()

    draw_slices(zip_data_duration=zip_data_duration, duration=duration, dx=dx, frequency=frequency,
                save_folder=save_folder,
                title=title, show=True)  # show=False)


def main():
    datapath = '/home/b-rain/rhythmic'
    # datapath = 'C:/rhythmic'
    read_data(datapath)
    print('done')


if __name__ == '__main__':
    main()
