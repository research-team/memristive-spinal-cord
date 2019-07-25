from analysis.PCA import get_lat_amp, prepare_data, get_peaks
from analysis.functions import read_data
import numpy as np
from analysis.cut_several_steps_files import select_slices
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from analysis.functions import changing_peaks

herz = 40
step = 0.25
ees_end = 9 * 4

color_bio = '#7a1c15'
color_neuron = '#e0930d'
color_gras = '#287a71'

bio = read_data('../bio-data/hdf5/bio_sci_E_15cms_40Hz_i100_2pedal_5ht_T_2016-05-12.hdf5')
bio = prepare_data(bio)

all_bio_slices = []
for k in range(len(bio)):
	bio_slices = []
	offset= 0
	for i in range(int(len(bio[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio[k][j])
		bio_slices.append(bio_slices_tmp)
		offset += 100
	all_bio_slices.append(bio_slices)
all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]

print(len(all_bio_slices), len(all_bio_slices[0]))

for sl in range(len(all_bio_slices)):
	all_bio_slices[sl] = list(all_bio_slices[sl])
# for sl in all_bio_slices:
# 	print("all_bio_slices = ", len(sl), len(sl[0]), sl)

mono_bio = []
for sl in range(len(all_bio_slices)):
	mono_bio_tmp = []
	for run in range(len(all_bio_slices[sl])):
		mono_bio_tmp.append([abs(a) for a in all_bio_slices[sl][run][:ees_end]])
	mono_bio.append(mono_bio_tmp)

# print(len(mono_bio))
# for sl in mono_bio:
# 	print("mono_bio = ", len(sl), len(sl[0]), sl)

# print("len(sum_data_by_run_bio_flat_list) = ", len(sum_data_by_run_bio_flat_list))
# for sl in sum_data_by_run_bio:
# 	print("sum_data_by_run_bio = ", sl)

latencies = get_lat_amp(bio, herz, step)[0]
print("latencies = ", latencies)

poly_bio = []
for sl in range(len(all_bio_slices)):
	poly_bio_tmp = []
	for run in range(len(all_bio_slices[sl])):
		poly_bio_tmp.append([abs(a) for a in all_bio_slices[sl][run][int(latencies[sl] * 4):]])
	poly_bio.append(poly_bio_tmp)

data_by_run_bio = []
for index, sl in enumerate(mono_bio):
	data_by_run_bio.append([list(a) for a in zip(*sl)])

print("data_by_run_bio = ", data_by_run_bio)
for sl in data_by_run_bio:
	print("data_by_run_bio = ", len(sl), sl)

sum_data_by_run_bio = []
for sl in data_by_run_bio:
	sum_data_by_run_bio.append([sum(i) for i in zip(*sl)])

print(len(sum_data_by_run_bio), len(sum_data_by_run_bio[0]))

# sum_data_by_run_bio = np.array(sum_data_by_run_bio)
# sum_data_by_run_bio = sum_data_by_run_bio.T
# sum_data_by_run_bio = list(sum_data_by_run_bio)
print(len(sum_data_by_run_bio), len(sum_data_by_run_bio[0]))

sum_data_by_run_bio_flat_list = []
for sl in sum_data_by_run_bio:
	for i in sl:
		sum_data_by_run_bio_flat_list.append(i)

# print(len(bio[0]))

# amplitudes_bio = []
# amplitudes_bio_all = []
# for run in bio:
# 	amplitudes_bio.append(sim_process(latencies, run, step, ees_end, inhibition_zero=True, after_latencies=True)[1])
# print("amplitudes_bio = ", amplitudes_bio)

# amplitudes_bio_mono = []

peaks_bio = get_peaks(bio, herz, step)[6]
peaks_mono_bio = changing_peaks(bio, herz, step, ees_end)[-1]
peaks_mono_bio = list(zip(*peaks_mono_bio))
for i in range(len(peaks_mono_bio)):
	peaks_mono_bio[i] = list(peaks_mono_bio[i])

# print("peaks_bio = ", peaks_bio)
# print("peaks_mono_bio = ", peaks_mono_bio)
# amplitudes_bio_flat_list = []
# for a in range(len(amplitudes_bio)):
# 	for el in range(len(amplitudes_bio[a])):
# 		amplitudes_bio_flat_list.append(amplitudes_bio[a][el])

peaks_bio_flat_list = []
for a in range(len(peaks_bio)):
	for el in range(len(peaks_bio[a])):
		peaks_bio_flat_list.append(peaks_bio[a][el])

neuron = np.array(select_slices('../../neuron-data/mn_E_2pedal_15speed_5th_25tests_hdf.hdf5', 0, 12000))
neuron = np.negative(neuron)
neuron_zoomed = []
for sl in neuron:
	neuron_zoomed.append(sl[::10])
neuron = prepare_data(neuron_zoomed)

all_neuron_slices = []
for k in range(len(neuron)):
	neuron_slices= []
	offset= 0
	for i in range(int(len(neuron[k]) / 100)):
		neuron_slices_tmp = []
		for j in range(offset, offset + 100):
			neuron_slices_tmp.append(neuron[k][j])
		neuron_slices.append(neuron_slices_tmp)
		offset += 100
	all_neuron_slices.append(neuron_slices)
all_neuron_slices = list(zip(*all_neuron_slices)) # list [16][4][100]

for sl in range(len(all_neuron_slices)):
	all_neuron_slices[sl] = list(all_neuron_slices[sl])

mono_neuron = []
for sl in range(len(all_neuron_slices)):
	mono_neuron_tmp = []
	for run in range(len(all_neuron_slices[sl])):
		mono_neuron_tmp.append([abs(a) for a in all_neuron_slices[sl][run][:ees_end]])
	mono_neuron.append(mono_neuron_tmp)

neuron = np.array(neuron)
# print("type(neuron) = ", type(neuron))
latencies = get_lat_amp(neuron, herz, step)[0]

poly_neuron = []
for sl in range(len(all_neuron_slices)):
	poly_neuron_tmp = []
	for run in range(len(all_neuron_slices[sl])):
		poly_neuron_tmp.append([abs(a) for a in all_neuron_slices[sl][run][int(latencies[sl] * 4):]])
	poly_neuron.append(poly_neuron_tmp)
data_for_peaks_plotting = get_peaks(neuron, herz, step)

data_by_run_neuron = []
for index, sl in enumerate(mono_neuron):
	data_by_run_neuron.append([list(a) for a in zip(*sl)])

sum_data_by_run_neuron = []
for sl in data_by_run_neuron:
	sum_data_by_run_neuron.append([sum(i) for i in zip(*sl)])

# sum_data_by_run_neuron = np.array(sum_data_by_run_neuron)
# sum_data_by_run_neuron = sum_data_by_run_neuron.T
# sum_data_by_run_neuron = list(sum_data_by_run_neuron)

sum_data_by_run_neuron_flat_list = []
for sl in sum_data_by_run_neuron:
	for i in sl:
		sum_data_by_run_neuron_flat_list.append(i)

# print("latencies = ", latencies)

# ees_end = 9 * 4
# amplitudes_neuron = []
# for run in neuron:
# 	amplitudes_neuron.append(sim_process(latencies, run, step, ees_end, inhibition_zero=True, after_latencies=True)[1])

# print("amplitudes_neuron = ", len(amplitudes_neuron), len(amplitudes_neuron[0]), amplitudes_neuron)

peaks_neuron = get_peaks(neuron, herz, step)[6]

peaks_mono_neuron = changing_peaks(neuron, herz, step, ees_end)[-1]
peaks_mono_neuron = list(zip(*peaks_mono_neuron))
for i in range(len(peaks_mono_neuron)):
	peaks_mono_neuron[i] = list(peaks_mono_neuron[i])

# amplitudes_neuron_flat_list = []
# for a in range(len(amplitudes_neuron)):
# 	for el in range(len(amplitudes_neuron[a])):
# 		amplitudes_neuron_flat_list.append(amplitudes_neuron[a][el])

peaks_neuron_flat_list = []
for a in range(len(peaks_neuron)):
	for el in range(len(peaks_neuron[a])):
		peaks_neuron_flat_list.append(peaks_neuron[a][el])

gras = np.array(select_slices('../../GRAS/MN_E _4pedal_21.hdf5', 5000, 11000))
# gras = np.negative(gras)
gras_zoomed = []
for sl in gras:
	gras_zoomed.append(sl[::10])
gras = prepare_data(gras_zoomed)

print("gras = ", len(gras), gras)
all_gras_slices = []
for k in range(len(gras)):
	gras_slices = []
	offset= 0
	for i in range(int(len(gras[k]) / 100)):
		gras_slices_tmp = []
		for j in range(offset, offset + 100):
			gras_slices_tmp.append(gras[k][j])
		gras_slices.append(gras_slices_tmp)
		offset += 100
	all_gras_slices.append(gras_slices)
all_gras_slices = list(zip(*all_gras_slices)) # list [16][4][100]

for sl in range(len(all_gras_slices)):
	all_gras_slices[sl] = list(all_gras_slices[sl])

mono_gras = []
for sl in range(len(all_gras_slices)):
	mono_gras_tmp = []
	for run in range(len(all_gras_slices[sl])):
		mono_gras_tmp.append([abs(a) for a in all_gras_slices[sl][run][:ees_end]])
	mono_gras.append(mono_gras_tmp)

latencies = get_lat_amp(gras, herz, step, debugging=False)[0]

poly_gras = []
for sl in range(len(all_gras_slices)):
	poly_gras_tmp = []
	for run in range(len(all_gras_slices[sl])):
		poly_gras_tmp.append([abs(a) for a in all_gras_slices[sl][run][int(latencies[sl] * 4):]])
	poly_gras.append(poly_gras_tmp)

data_by_run_gras = []
for index, sl in enumerate(poly_gras):
	data_by_run_gras.append([list(a) for a in zip(*sl)])

print(data_by_run_gras)

sum_data_by_run_gras = []
for sl in data_by_run_gras:
	sum_data_by_run_gras.append([sum(i) for i in zip(*sl)])

sum_data_by_run_gras = np.array(sum_data_by_run_gras)
sum_data_by_run_gras = sum_data_by_run_gras.T
sum_data_by_run_gras = list(sum_data_by_run_gras)

sum_data_by_run_gras_flat_list = []
for sl in sum_data_by_run_gras:
	for i in sl:
		sum_data_by_run_gras_flat_list.append(i)

print("sum_data_by_run_gras_flat_list = ", sum_data_by_run_gras_flat_list)

# amplitudes_gras = []
# for run in gras:
# 	amplitudes_gras.append(sim_process(latencies, run, step, ees_end, inhibition_zero=True, after_latencies=True)[1])

# print("amplitudes_gras f= ", amplitudes_gras)
# peaks_gras = get_peaks(gras, herz, step)[6]

# peaks_mono_gras = changing_peaks(gras, herz, step, ees_end)[-1]
# peaks_mono_gras = list(zip(*peaks_mono_gras))
# for i in range(len(peaks_mono_gras)):
# 	peaks_mono_gras[i] = list(peaks_mono_gras[i])

# print("amplitudes_gras = ", amplitudes_gras)
# amplitudes_gras_flat_list = []
# for a in range(len(amplitudes_gras)):
# 	for el in range(len(amplitudes_gras[a])):
# 		amplitudes_gras_flat_list.append(amplitudes_gras[a][el])
# print("amplitudes_gras_flat_list = ", amplitudes_gras_flat_list)

print(len(sum_data_by_run_bio_flat_list ))
print(len(sum_data_by_run_gras_flat_list))

amplitudes = sum_data_by_run_bio_flat_list + sum_data_by_run_neuron_flat_list   # sum_data_by_run_gras_flat_list #
peaks_gras_flat_list = []
# for a in range(len(peaks_mono_gras)):
# 	for el in range(len(peaks_mono_gras[a])):
# 		peaks_gras_flat_list.append(peaks_mono_gras[a][el])

peaks = peaks_bio_flat_list + peaks_gras_flat_list  # peaks_neuron_flat_list##+ # +
# print("peaks_bio_flat_list = ", len(peaks_bio_flat_list))

print("len(sum_data_by_run_bio_flat_list) = ", len(sum_data_by_run_bio_flat_list))
simulators = []
for i in range(len(sum_data_by_run_bio_flat_list)):
	simulators.append('bio')

# for i in range(len(sum_data_by_run_bio_flat_list),
#                len(sum_data_by_run_bio_flat_list) + len(sum_data_by_run_gras_flat_list)):
# 	simulators.append('gras')

for i in range(len(sum_data_by_run_bio_flat_list),
               len(sum_data_by_run_bio_flat_list) + len(sum_data_by_run_neuron_flat_list)):
	simulators.append('neuron')
# for i in range(len(peaks_bio_flat_list) + len(peaks_neuron_flat_list),
#                len(peaks_bio_flat_list) + len(peaks_neuron_flat_list) + len(peaks_gras_flat_list)):
# 	simulators.append('gras')

print("len(sum_data_by_run_bio) = ", len(sum_data_by_run_bio), len(sum_data_by_run_bio[0]))
slices_bio = []
for i in range(len(sum_data_by_run_bio)):
	for j in range(len(sum_data_by_run_bio[i])):
		slices_bio.append(i + 1)

# print("len(sum_data_by_run_neuron) = ", len(sum_data_by_run_neuron), len(sum_data_by_run_neuron[0]))
slices_neuron = []
for i in range(len(sum_data_by_run_neuron)):
	for j in range(len(sum_data_by_run_neuron[i])):
		slices_neuron.append(i + 1)

slices_gras = []
for i in range(len(sum_data_by_run_gras)):
	for j in range(len(sum_data_by_run_gras[i])):
		slices_gras.append(j + 1)

slices = slices_bio + slices_neuron # slices_gras

print("peaks = ", len(peaks), peaks)
print("simulators = ", len(simulators), simulators)
print("slices = ", len(slices), slices)
print("amplitudes = ", len(amplitudes), amplitudes)
df = pd.DataFrame({'Amplitudes': amplitudes, 'Simulators': simulators, 'Slices': slices},
                  columns=['Amplitudes', 'Simulators', 'Slices'])

# df_peaks = pd.DataFrame({'Peaks': peaks, 'Simulators': simulators, 'Slices': slices},
#                   columns=['Peaks', 'Simulators', 'Slices'])

pal = {simulators: color_bio if simulators == 'bio' else color_neuron if simulators == 'neuron' else color_gras
       for simulators in df['Simulators']}
bp = sns.boxplot(x='Slices', y='Amplitudes', hue='Simulators', data=df, palette=pal)
m1 = df.groupby(['Slices', 'Simulators'])['Amplitudes'].median().values
mL1 = [str(np.round(s, 2)) for s in m1]
plt.xticks(fontsize=35)
plt.xlabel('Slices', fontsize=56)
plt.yticks(fontsize=35)
# plt.ylabel('Amplitudes, mV', fontsize=56)
plt.show()

bp = sns.boxplot(x='Slices', y='Peaks', hue='Simulators', data=df_peaks, palette=pal)
m1 = df_peaks.groupby(['Slices', 'Simulators'])['Peaks'].median().values
mL1 = [str(np.round(s, 2)) for s in m1]
plt.xticks(fontsize=28)
plt.xlabel('Slices', fontsize=56)
plt.yticks(fontsize=28)
plt.ylabel('Peaks', fontsize=56)
plt.show()