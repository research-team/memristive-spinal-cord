from analysis.PCA import get_lat_amp, prepare_data
from analysis.functions import read_data, sim_process
import numpy as np
from analysis.cut_several_steps_files import select_slices
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

herz = 40
step = 0.25
color_bio = '#7a1c15'
color_neuron = '#e0930d'
color_gras = '#287a71'
bio = read_data('../bio-data/hdf5/bio_control_E_21cms_40Hz_i100_4pedal_no5ht_T_2017-09-05.hdf5')
bio = prepare_data(bio)
latencies = get_lat_amp(bio, herz, step)[0]
amplitudes_bio = []
for run in bio:
	amplitudes_bio.append(sim_process(latencies, run, step, inhibition_zero=True, after_latencies=True)[1])

print("amplitudes_bio = ", amplitudes_bio)
amplitudes_bio_flat_list = []
for a in range(len(amplitudes_bio)):
	for el in range(len(amplitudes_bio[a])):
		amplitudes_bio_flat_list.append(amplitudes_bio[a][el])

neuron = np.array(select_slices('../../neuron-data/mn_E_quad_21.hdf5', 0, 6000))
neuron = np.negative(neuron)
neuron_zoomed = []
for sl in neuron:
	neuron_zoomed.append(sl[::10])
neuron = prepare_data(neuron_zoomed)
latencies = get_lat_amp(neuron, herz, step)[0]
print("latencies = ", latencies)
amplitudes_neuron = []
for run in neuron:
	amplitudes_neuron.append(sim_process(latencies, run, step, inhibition_zero=True, after_latencies=True)[1])

print("amplitudes_neuron = ", amplitudes_neuron)
amplitudes_neuron_flat_list = []
for a in range(len(amplitudes_neuron)):
	for el in range(len(amplitudes_neuron[a])):
		amplitudes_neuron_flat_list.append(amplitudes_neuron[a][el])

gras = np.array(select_slices('../../GRAS/E_21cms_40Hz_100%_2pedal_no5ht.hdf5', 5000, 11000))
gras = np.negative(gras)
gras_zoomed = []
for sl in gras:
	gras_zoomed.append(sl[::10])
gras = prepare_data(gras_zoomed)
latencies = get_lat_amp(gras, herz, step)[0]
amplitudes_gras = []
for run in gras:
	amplitudes_gras.append(sim_process(latencies, run, step, inhibition_zero=True, after_latencies=True)[1])

print("amplitudes_gras = ", amplitudes_gras)
amplitudes_gras_flat_list = []
for a in range(len(amplitudes_gras)):
	for el in range(len(amplitudes_gras[a])):
		amplitudes_gras_flat_list.append(amplitudes_gras[a][el])

amplitudes = amplitudes_bio_flat_list + amplitudes_neuron_flat_list# + amplitudes_gras_flat_list

simulators = []
for i in range(len(amplitudes_bio_flat_list)):
	simulators.append('bio')

for i in range(len(amplitudes_bio_flat_list), len(amplitudes_bio_flat_list) + len(amplitudes_neuron_flat_list)):
	simulators.append('neuron')
# for i in range(len(amplitudes_bio_flat_list) + len(amplitudes_neuron_flat_list),
#                len(amplitudes_bio_flat_list) + len(amplitudes_neuron_flat_list) + len(amplitudes_gras_flat_list)):
# 	simulators.append('gras')

slices_bio = []
for i in range(len(amplitudes_bio)):
	for j in range(len(amplitudes_bio[i])):
		slices_bio.append(j + 1)

slices_neuron = []
for i in range(len(amplitudes_neuron)):
	for j in range(len(amplitudes_neuron[i])):
		slices_neuron.append(j + 1)

slices_gras = []
for i in range(len(amplitudes_gras)):
	for j in range(len(amplitudes_gras[i])):
		slices_gras.append(j + 1)

slices = slices_bio + slices_neuron# + slices_gras

df = pd.DataFrame({'Amplitudes': amplitudes, 'Simulators': simulators, 'Slices': slices},
                  columns=['Amplitudes', 'Simulators', 'Slices'])

pal = {simulators: color_bio if simulators == 'bio' else color_neuron# if simulators == 'neuron' else color_gras
       for simulators in df['Simulators']}
bp = sns.boxplot(x='Slices', y='Amplitudes', hue='Simulators', data=df, palette=pal)
m1 = df.groupby(['Slices', 'Simulators'])['Amplitudes'].median().values
mL1 = [str(np.round(s, 2)) for s in m1]
plt.xticks(fontsize=56)
plt.xlabel('Slices', fontsize=56)
plt.yticks(fontsize=56)
plt.ylabel('Amplitudes, mV', fontsize=56)
plt.show()