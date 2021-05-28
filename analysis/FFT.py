import numpy as np
import matplotlib.pyplot as plt

sampling_frequency = 40_000 # frquency of the data [Hz]
sampling_interval = 1 / sampling_frequency * 1000 # convert [Hz] to [ms]

# get the data
###### remove me from here
path = '/home/alex/GitHub/memristive-spinal-cord/GRAS/gras_neuron/dat/0_muscle_E.dat'
myogram = np.array(open(path).readline().split()).astype(np.float)
myogram[0] = 0
###### to here

sampling_size = len(myogram)    # get size (length) of the data
time = np.arange(sampling_size) * sampling_interval # convert a ticks to time

# frequency domain representation
fourier_transform = np.fft.fft(myogram) / sampling_size  # normalize amplitude
fourier_transform = abs(fourier_transform[range(int(sampling_size / 2))])  # exclude sampling frequency
# remove the mirrored part of the FFT
values = np.arange(int(sampling_size / 2))
time_period = sampling_size / sampling_frequency
frequencies = values / time_period

# find the maximal frequence
max_value_index = np.argmax(fourier_transform)
max_freq = frequencies[max_value_index]
max_ampl = fourier_transform[max_value_index]

# plotting
figure, axis = plt.subplots(2, 1)
# plot myogram
axis[0].set_title('Myogram, 40KHz')
axis[0].plot(time, myogram)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')
# plot FFT
axis[1].set_title('Fourier transform depicting the frequency components')
axis[1].plot(frequencies, fourier_transform)
axis[1].plot([max_freq], [max_ampl], '.', color='r', label='maximum')
axis[1].set_xlabel('Frequency')
axis[1].set_ylabel('Amplitude')
axis[1].legend()
# squeeze plot
plt.tight_layout()
plt.show()
