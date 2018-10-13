import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import scipy.io as sio
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.real_data_slices import slice_myogram, read_data

# https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python#25735274

logging.basicConfig(level=logging.DEBUG)

ds = [0.0, 13.0,  13.0,  15.0,  17.0,  17.0,  21.0,  16.0,  0.0]#delays
fs = [0.0, 300.0, 500.0, 500.0, 250.0, 250.0, 800.0, 500.0, 0.0]#frequencies
ls = [0.0, 2.0,   4.0,   10.0,  8.0,   8.0,   4.0,   9.0,   0.0]#lengths(durations)
vs = [0.0, 0.5,   2.0,   1.5,   2.0,   2.0,   1.5,   1.0,   0.0]#volts(amplitudes)

logging.info('Setup complete')

logging.info('Loading myograms')

'''
RTA right  tibialis anterior (flexor)
RMG right adductor magnus (extensor)
'''
path = '../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat'
#path= '../bio-data/SCI_Rat-1_11-22-2016_RMG_20Hz_one_step.mat'
#path = '../bio-data/SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat'
#path'../bio-data/SCI_Rat-1_11-22-2016_RTA_20Hz_one_step.mat'

myogram_data = read_data(path)

logging.info('Loaded myograms')

voltages, start_times = slice_myogram(myogram_data)
slices_max_time, slices_max_value, slices_min_time, slices_min_value = calc_max_min(start_times, voltages)

#TODO calculate delay time to 3rd extremum
#TODO calculate duration time_to_last_etremum - time_to_3rd_extremum
#TODO calculete frequencies with FFT https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python#25735274

logging.info('Plotting')
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(ds, ls, vs, lw=0.5)
ax.set_xlabel("Delay ms")
ax.set_ylabel("Duration ms")
ax.set_zlabel("Amplitude mV")
ax.set_title("Delay - Frequency - Amplitude")

plt.show()

logging.info('Processing complete')
