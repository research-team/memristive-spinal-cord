import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import logging

# https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python#25735274

logging.basicConfig(level=logging.DEBUG)

ds = [0.0, 13.0,  13.0,  15.0,  17.0,  17.0,  21.0,  16.0,  0.0]
fs = [0.0, 300.0, 500.0, 500.0, 250.0, 250.0, 800.0, 500.0, 0.0]
ls = [0.0, 2.0,   4.0,   10.0,  8.0,   8.0,   4.0,   9.0,   0.0]
vs = [0.0, 0.5,   2.0,   1.5,   2.0,   2.0,   1.5,   1.0,   0.0]

logging.info('Setup complete')

logging.info('Loading data')

mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
#mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RMG_20Hz_one_step.mat')
#mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat')
#mat_data = sio.loadmat('../bio-data/SCI_Rat-1_11-22-2016_RTA_20Hz_one_step.mat')

tick_rate = int(mat_data['tickrate'][0][0])

logging.info('Loaded data')



logging.info('Plotting')
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(ds, ls, fs, lw=0.5)
ax.set_xlabel("Delay ms")
ax.set_ylabel("Duration ms")
#ax.set_zlabel("Amplitude mV")
ax.set_zlabel("Frequency Hz")
ax.set_title("Delay - Duration - Frequency")

plt.show()

logging.info('Processing complete')
