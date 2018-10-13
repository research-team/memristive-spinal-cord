import numpy as np
import logging
import scipy.io as sio
import os
import random
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.peaks_of_real_data_without_EES import remove_ees_from_min_max, delays, calc_durations
from analysis.real_data_slices import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python#25735274
# logging.basicConfig(level=logging.DEBUG)
raw_real_data = read_data(os.getcwd() + '/../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
myogram_data = slice_myogram(raw_real_data)
volt_data = myogram_data[0]
slices_begin_time = myogram_data[1]
slices_begin_time = [int(t / real_data_step) for t in slices_begin_time]

data = calc_max_min(slices_begin_time, volt_data, data_step=0.25)
data_with_deleted_ees = remove_ees_from_min_max(data[0], data[1], data[2], data[3])
max_min_delays = delays(data_with_deleted_ees[0], data_with_deleted_ees[2])
max_min_durations = calc_durations(data_with_deleted_ees[0], data_with_deleted_ees[2])
ds = max_min_delays[0]
ls = max_min_durations[0]


ds =[7.5, 4.0, 4.5, 8.75, 4.0, 4.0, 7.75, 7.75, 4.0, 4.0, 8.5, 8.25, 8.0, 8.0, 8.0, 8.75, 8.0, 10.5, 8.0, 7.5, 19.75, 10.75, 11.0, 10.5, 8.75, 4.25]
ls= [16.5, 19.0, 14.25, 15.0, 15.75, 19.75, 12.25, 11.75, 19.0, 17.25, 15.25, 16.0, 12.5, 10.5, 15.75, 11.25, 16.0, 13.0, 9.0, 16.25, 0.0, 7.25, 9.5, 7.75, 10.25, 17.25]
# ds = [0.0, 13.0,  13.0,  15.0,  17.0,  17.0,  21.0,  16.0,  0.0]#delays
fs = [0.0, 300.0, 500.0, 500.0, 250.0, 250.0, 800.0, 500.0, 0.0]    # frequencies
# ls = [0.0, 2.0,   4.0,   10.0,  8.0,   8.0,   4.0,   9.0,   0.0]  # lengths(durations)
# for i in range (26):
vs = [0.0, 0.5,   1.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0, 4.5, 5.0, 5.5, 6.0, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]    # volts(amplitudes)
print(len(vs))
print(len(ds))
print(len(ls))
# logging.info('Setup complete')
#
# logging.info('Loading myograms')

'''
RTA right  tibialis anterior (flexor)
RMG right adductor magnus (extensor)
'''
# path = '../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat'
# #path= '../bio-data/SCI_Rat-1_11-22-2016_RMG_20Hz_one_step.mat'
# #path = '../bio-data/SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat'
# #path'../bio-data/SCI_Rat-1_11-22-2016_RTA_20Hz_one_step.mat'
#
# myogram_data = read_data(path)
#
# logging.info('Loaded myograms')
#
# voltages, start_times = slice_myogram(myogram_data)
# slices_max_time, slices_max_value, slices_min_time, slices_min_value = calc_max_min(start_times, voltages)

#TODO calculate delay time to 3rd extremum
#TODO calculate duration time_to_last_etremum - time_to_3rd_extremum
#TODO calculete frequencies with FFT https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python#25735274

# logging.info('Plotting')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(ds, ls, vs, lw=0.5)
ax.plot(ds, ls, vs, '.', lw=0.5, color='r', markersize=5)
ax.set_xlabel("Delay ms")
ax.set_ylabel("Duration ms")
ax.set_zlabel("Amplitude mV")
ax.set_title("Delay - Duration - Amplitude")

plt.show()
#
# logging.info('Processing complete')
