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

bio_path= '../bio-data/hdf5/bio_sci_E_15cms_40Hz_i100_4pedal_no5ht_T_2016-06-12.hdf5'
bio = read_data(bio_path)
bio = prepare_data(bio)

neuron_path = '../../neuron-data/mn_E_4pedal_15speed_25tests_hdf.hdf5'
neuron = np.array(select_slices(neuron_path, 0, 12000))
neuron = np.negative(neuron)
neuron_zoomed = []
for sl in neuron:
	neuron_zoomed.append(sl[::10])
neuron = prepare_data(neuron_zoomed)

gras_path = '../../GRAS/MN_E_4pedal_15.hdf5'
gras = np.array(select_slices(gras_path, 10000, 22000))
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

print("bio_path = ", bio_path)
print("bio_amp = ", bio_amp)
print("bio_peaks = ", bio_peaks)

print("neuron_path= ", neuron_path)
print("neuron_amp = ", neuron_amp)
print("neuron_peaks = ", neuron_peaks)

print("gras_path = ", gras_path)
print("gras_amp = ", gras_amp)
print("gras_peaks = ", gras_peaks)

df = pd.DataFrame({'Gras': gras_amp # 'Neuron': neuron_amp # 'Bio': bio_amp
                   })

df_peaks = pd.DataFrame({'Gras': gras_peaks # 'Neuron': neuron_peaks # 'Bio': bio_peaks
                   })
df_delta_peaks = pd.DataFrame({'Bio - Neuron': delta_peaks_bio_gras  # 'Bio - neuron': delta_peaks_bio_neuron#,
                   })
df_delta_amps = pd.DataFrame({'Bio - Neuron': delta_amp_bio_gras    # 'Bio - neuron': delta_amp_bio_neuron#,
                   })

colors = ['#472650', color_gras, color_neuron, ]
df_delta_amps.plot(kind='bar', color=colors)

for i in range(len(delta_amp_bio_neuron)):
	print("i = ", i)
plt.xticks(range(len(delta_amp_bio_neuron)), [i  + 1 if i % 3 == 0 or i % 11 == 0
                                              else "" for i in range(len(delta_amp_bio_neuron))],
           fontsize=56, rotation=0)
# plt.xlabel('Slices', fontsize=56)
plt.gca().get_legend().remove()
plt.yticks(fontsize=56)
# plt.ylabel('Amplitudes, mV', fontsize=56)

plt.show()
colors = ['#472650', color_gras, color_neuron, ]
df_delta_peaks.plot(kind='bar', color=colors)

plt.xticks(range(len(delta_peaks_bio_neuron)), [i  + 1 if i % 3 == 0 or i % 11 == 0 # i % 5 == 0 or
                                              else "" for i in range(len(delta_peaks_bio_neuron))],
           fontsize=56, rotation=0)

# plt.xlabel('Slices', fontsize=56)
plt.gca().get_legend().remove()
plt.yticks(fontsize=56)
# plt.ylabel('Peaks', fontsize=56)

plt.show()