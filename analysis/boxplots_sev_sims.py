from analysis.PCA import get_lat_amp, prepare_data, get_peaks, plot_peaks
from analysis.functions import read_data, sim_process
import numpy as np
from analysis.cut_several_steps_files import select_slices
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from operator import sub
from analysis.functions import changing_peaks

herz = 40
step = 0.25
color_bio = '#7a1c15'
color_neuron = '#e0930d'
color_gras = '#287a71'
bio = read_data('../bio-data/hdf5/bio_control_E_21cms_40Hz_i100_2pedal_no5ht_T_2017-09-05.hdf5')
bio = prepare_data(bio)
latencies = get_lat_amp(bio, herz, step)[0]
print(len(bio[0]))

ees_end = 9 * 4
latencies_zero = []
# for i in range(int(len(bio[0]) / 100)):
	# latencies_zero.append(0)
amplitudes_bio = []
amplitudes_bio_all = []
for run in bio:
	amplitudes_bio.append(sim_process(latencies, run, step, ees_end, inhibition_zero=True, after_latencies=True)[1])
	# amplitudes_bio_all.append(sim_process(latencies_zero, run, step, inhibition_zero=True, after_latencies=True)[1])
print("amplitudes_bio = ", amplitudes_bio)

amplitudes_bio_mono = []
# for run in range(len(amplitudes_bio_all)):
	# amplitudes_bio_mono.append(list(map(sub, amplitudes_bio_all[run], amplitudes_bio[run])))

peaks_bio = get_peaks(bio, herz, step)[6]
peaks_mono_bio = changing_peaks(bio, herz, step, ees_end)[-1]
peaks_mono_bio = list(zip(*peaks_mono_bio))
for i in range(len(peaks_mono_bio)):
	peaks_mono_bio[i] = list(peaks_mono_bio[i])

print("peaks_bio = ", peaks_bio)
print("peaks_mono_bio = ", peaks_mono_bio)
amplitudes_bio_flat_list = []
for a in range(len(amplitudes_bio)):
	for el in range(len(amplitudes_bio[a])):
		amplitudes_bio_flat_list.append(amplitudes_bio[a][el])

peaks_bio_flat_list = []
for a in range(len(peaks_bio)):
	for el in range(len(peaks_bio[a])):
		peaks_bio_flat_list.append(peaks_bio[a][el])

neuron = np.array(select_slices('../../neuron-data/mn_E25tests_nr.hdf5', 11000, 17000))
neuron = np.negative(neuron)
neuron_zoomed = []
for sl in neuron:
	neuron_zoomed.append(sl[::10])
neuron = prepare_data(neuron_zoomed)
neuron = np.array(neuron)
print("type(neuron) = ", type(neuron))
latencies = get_lat_amp(neuron, herz, step)[0]

data_for_peaks_plotting = get_peaks(neuron, herz, step)
# plot_peaks(neuron, latencies, data_for_peaks_plotting[1], data_for_peaks_plotting[2], data_for_peaks_plotting[3],
#            data_for_peaks_plotting[4], data_for_peaks_plotting[5], data_for_peaks_plotting[6],
#            data_for_peaks_plotting[7], data_for_peaks_plotting[8], data_for_peaks_plotting[9])
print("latencies = ", latencies)
latencies_zero = []
# for i in range(int(len(neuron[0]) / 100)):
	# latencies_zero.append(0)

ees_end = 9 * 4
amplitudes_neuron = []
amplitudes_neuron_all = []
for run in neuron:
	amplitudes_neuron.append(sim_process(latencies, run, step, ees_end, inhibition_zero=True, after_latencies=True)[1])
	# amplitudes_neuron_all.append(sim_process(latencies_zero, run, step, ees_end, inhibition_zero=True,
	#                                          after_latencies=True)[1])

print("amplitudes_neuron = ", len(amplitudes_neuron), len(amplitudes_neuron[0]), amplitudes_neuron)
# print("amplitudes_neuron_all = ", len(amplitudes_neuron_all), len(amplitudes_neuron_all[0]), amplitudes_neuron_all)

amplitudes_neuron_mono = []
# for run in range(len(amplitudes_neuron_all)):
	# amplitudes_neuron_mono.append(list(map(sub, amplitudes_neuron_all[run], amplitudes_neuron[run])))

# print("amplitudes_neuron mono = ", len(amplitudes_neuron_mono), len(amplitudes_neuron_mono[0]), amplitudes_neuron_mono)
peaks_neuron = get_peaks(neuron, herz, step)[6]

peaks_mono_neuron = changing_peaks(neuron, herz, step, ees_end)[-1]
peaks_mono_neuron = list(zip(*peaks_mono_neuron))
for i in range(len(peaks_mono_neuron)):
	peaks_mono_neuron[i] = list(peaks_mono_neuron[i])

amplitudes_neuron_flat_list = []
for a in range(len(amplitudes_neuron)):
	for el in range(len(amplitudes_neuron[a])):
		amplitudes_neuron_flat_list.append(amplitudes_neuron[a][el])

peaks_neuron_flat_list = []
for a in range(len(peaks_neuron)):
	for el in range(len(peaks_neuron[a])):
		peaks_neuron_flat_list.append(peaks_neuron[a][el])

gras = np.array(select_slices('../../GRAS/E_15cms_40Hz_100%_2pedal_no5ht.hdf5', 10000, 22000))
gras = np.negative(gras)
gras_zoomed = []
for sl in gras:
	gras_zoomed.append(sl[::10])
gras = prepare_data(gras_zoomed)
latencies = get_lat_amp(gras, herz, step)[0]
amplitudes_gras = []
for run in gras:
	amplitudes_gras.append(sim_process(latencies, run, step, ees_end, inhibition_zero=True, after_latencies=True)[1])

# print("amplitudes_gras = ", amplitudes_gras)

peaks_gras = get_peaks(gras, herz, step)[6]

peaks_mono_gras = changing_peaks(gras, herz, step, ees_end)[-1]
peaks_mono_gras = list(zip(*peaks_mono_gras))
for i in range(len(peaks_mono_gras)):
	peaks_mono_gras[i] = list(peaks_mono_gras[i])

amplitudes_gras_flat_list = []
for a in range(len(amplitudes_gras)):
	for el in range(len(amplitudes_gras[a])):
		amplitudes_gras_flat_list.append(amplitudes_gras[a][el])

amplitudes = amplitudes_bio_flat_list + amplitudes_neuron_flat_list
peaks_gras_flat_list = []
for a in range(len(peaks_mono_gras)):
	for el in range(len(peaks_mono_gras[a])):
		peaks_gras_flat_list.append(peaks_mono_gras[a][el])

peaks = peaks_bio_flat_list + peaks_neuron_flat_list ##+ # +peaks_gras_flat_list

simulators = []
for i in range(len(peaks_bio_flat_list)):
	simulators.append('bio')

# for i in range(len(peaks_bio_flat_list), len(peaks_bio_flat_list) + len(peaks_gras_flat_list)):
# 	simulators.append('gras')

for i in range(len(peaks_bio_flat_list), len(peaks_bio_flat_list) + len(peaks_neuron_flat_list)):
	simulators.append('neuron')
# for i in range(len(peaks_bio_flat_list) + len(peaks_neuron_flat_list),
#                len(peaks_bio_flat_list) + len(peaks_neuron_flat_list) + len(peaks_gras_flat_list)):
# 	simulators.append('gras')

slices_bio = []
for i in range(len(peaks_bio)):
	for j in range(len(peaks_bio[i])):
		slices_bio.append(j + 1)

slices_neuron = []
for i in range(len(peaks_neuron)):
	for j in range(len(peaks_neuron[i])):
		slices_neuron.append(j + 1)

slices_gras = []
for i in range(len(peaks_gras)):
	for j in range(len(peaks_gras[i])):
		slices_gras.append(j + 1)

slices = slices_bio +slices_neuron #+ #+ #slices_gras

print("peaks = ", len(peaks), peaks)
print("simulators = ", len(simulators), simulators)
print("slices = ", len(slices), slices)
# print("amplitudes = ", len(amplitudes), amplitudes)
df = pd.DataFrame({'Amplitudes': amplitudes, 'Simulators': simulators, 'Slices': slices},
                  columns=['Amplitudes', 'Simulators', 'Slices'])

df_peaks = pd.DataFrame({'Peaks': peaks, 'Simulators': simulators, 'Slices': slices},
                  columns=['Peaks', 'Simulators', 'Slices'])

pal = {simulators: color_bio if simulators == 'bio' else color_neuron if simulators == 'neuron' else color_gras
       for simulators in df_peaks['Simulators']}
bp = sns.boxplot(x='Slices', y='Amplitudes', hue='Simulators', data=df, palette=pal)
m1 = df.groupby(['Slices', 'Simulators'])['Amplitudes'].median().values
mL1 = [str(np.round(s, 2)) for s in m1]
plt.xticks(fontsize=56)
plt.xlabel('Slices', fontsize=56)
plt.yticks(fontsize=56)
plt.ylabel('Amplitudes, mV', fontsize=56)
plt.show()

bp = sns.boxplot(x='Slices', y='Peaks', hue='Simulators', data=df_peaks, palette=pal)
m1 = df_peaks.groupby(['Slices', 'Simulators'])['Peaks'].median().values
mL1 = [str(np.round(s, 2)) for s in m1]
plt.xticks(fontsize=56)
plt.xlabel('Slices', fontsize=56)
plt.yticks(fontsize=56)
plt.ylabel('Peaks', fontsize=56)
plt.show()