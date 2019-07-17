from analysis.PCA import get_lat_amp, prepare_data, get_peaks
from analysis.functions import read_data
import numpy as np
from analysis.cut_several_steps_files import select_slices
import pandas as pd
from matplotlib import pylab as plt

herz = 40
step = 0.25
color_bio = '#7a1c15'
color_neuron = '#e0930d'
color_gras = '#287a71'

bio = read_data('../bio-data/hdf5/bio_control_E_21cms_40Hz_i100_2pedal_no5ht_T_2017-09-05.hdf5')
bio = prepare_data(bio)

neuron = np.array(select_slices('../../neuron-data/mn_E25tests_10.hdf5', 11000, 17000))
neuron = np.negative(neuron)
neuron_zoomed = []
for sl in neuron:
	neuron_zoomed.append(sl[::10])
neuron = prepare_data(neuron_zoomed)

gras = np.array(select_slices('../../GRAS/E_21cms_40Hz_100%_2pedal_no5ht.hdf5', 5000, 11000))
gras = np.negative(gras)
gras_zoomed = []
for sl in gras:
	gras_zoomed.append(sl[::10])
gras = prepare_data(gras_zoomed)

bio_amp = get_lat_amp(bio, herz, step)[1]
neuron_amp = get_lat_amp(neuron, herz, step)[1]
gras_amp = get_lat_amp(gras, herz, step)[1]

bio_peaks = get_peaks(bio, herz, step)[7]
neuron_peaks = get_peaks(neuron, herz, step)[7]
gras_peaks = get_peaks(gras, herz, step)[7]

delta_amp_bio_neuron = []
delta_amp_bio_gras = []
for i in range(len(bio_amp)):
	delta_amp_bio_neuron.append(abs(bio_amp[i] - neuron_amp[i]))
	delta_amp_bio_gras.append(abs(bio_amp[i] - gras_amp[i]))

delta_peaks_bio_neuron = []
delta_peaks_bio_gras = []
for i in range(len(bio_amp)):
	delta_peaks_bio_neuron.append(abs(bio_peaks[i] - neuron_peaks[i]))
	delta_peaks_bio_gras.append(abs(bio_peaks[i] - gras_peaks[i]))

slices_neuron = []
for i in range(len(delta_amp_bio_neuron)):
	slices_neuron.append(i + 1)

df = pd.DataFrame({'Bio': bio_amp#, 'Neuron': neuron_amp#, 'Gras': gras_amp
                   })

df_peaks = pd.DataFrame({'Bio': bio_peaks#, 'Neuron': neuron_peaks#, 'Gras': gras_peaks
                   })
df_delta_peaks = pd.DataFrame({'Bio - neuron': delta_peaks_bio_neuron#, 'Bio - Gras': delta_peaks_bio_gras
                   })
df_delta_amps = pd.DataFrame({'Bio - neuron': delta_amp_bio_neuron#, 'Bio - Gras': delta_peaks_bio_gras
                   })

colors = [color_neuron, color_gras]
df_delta_amps.plot(kind='bar', color=colors)

plt.xticks(range(len(bio_amp)), [i  + 1 if i % 1 == 0 else "" for i in range(len(bio_amp))],
           fontsize=56)
plt.xlabel('Slices', fontsize=56)
plt.yticks(fontsize=56)
plt.ylabel('Amplitudes, mV', fontsize=56)

plt.show()
colors = [color_neuron, color_gras]
df_delta_peaks.plot(kind='bar', color=colors)

plt.xticks(range(len(bio_amp)), [i  + 1 if i % 1 == 0 else "" for i in range(len(bio_amp))],
           fontsize=56)
plt.xlabel('Slices', fontsize=56)
plt.yticks(fontsize=56)
plt.ylabel('Peaks', fontsize=56)

plt.show()