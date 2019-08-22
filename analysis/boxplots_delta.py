from analysis.functions import read_data
from analysis.PCA import prepare_data, get_lat_amp, calc_boxplots
import numpy as np
from analysis.cut_several_steps_files import select_slices
import pandas as pd
import pylab as plt

herz = 40
step = 0.25
ees_end = 9 * 4

bio_path= '../bio-data/hdf5/bio_sci_E_15cms_40Hz_i100_2pedal_5ht_T_2016-05-12.hdf5'
neuron_path = '../../neuron-data/mn_E_4pedal_15speed_25tests_hdf.hdf5'
gras_path = '../../GRAS/MN_E_5HT.hdf5'
bio = read_data(bio_path)
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

print(len(all_bio_slices), len(all_bio_slices[0]), len(all_bio_slices[0][0]))

for sl in range(len(all_bio_slices)):
	all_bio_slices[sl] = list(all_bio_slices[sl])
	all_bio_slices[sl] = sum(all_bio_slices[sl], [])

print("all_bio_slices = ", len(all_bio_slices), len(all_bio_slices[0]))
mono_bio = []
for sl in range(len(all_bio_slices)):
	mono_bio.append([abs(a) for a in all_bio_slices[sl][:ees_end]])

print("mono_bio = ", mono_bio)
latencies = get_lat_amp(bio, herz, step)[0]
print("latencies = ", latencies)

poly_bio = []
for sl in range(len(all_bio_slices)):
	poly_bio.append([abs(a) for a in all_bio_slices[sl][int(latencies[sl] * 4):]])

high_whisker_bio_mono = []
low_whisker_bio_mono = []
for sl in mono_bio:
	high_whisker_bio_mono.append(calc_boxplots(sl)[3])
	low_whisker_bio_mono.append(calc_boxplots(sl)[4])

whiskers_diff_mono = []
for sl in range(len(high_whisker_bio_mono)):
	whiskers_diff_mono.append(high_whisker_bio_mono[sl] - low_whisker_bio_mono[sl])

print("high_whisker_bio_mono = ", high_whisker_bio_mono)
print("low_whisker_bio_mono = ", low_whisker_bio_mono)

high_whisker_bio_poly = []
low_whisker_bio_poly = []
for sl in poly_bio:
	high_whisker_bio_poly.append(calc_boxplots(sl)[3])
	low_whisker_bio_poly.append(calc_boxplots(sl)[4])

whiskers_diff_poly = []
for sl in range(len(high_whisker_bio_poly)):
	whiskers_diff_poly.append(high_whisker_bio_poly[sl] - low_whisker_bio_poly[sl])

print("high_whisker_bio_poly = ", high_whisker_bio_poly)
print("low_whisker_bio_poly = ", low_whisker_bio_poly)

neuron = np.array(select_slices(neuron_path, 0, 12000))
neuron = np.negative(neuron)
neuron_zoomed = []
for sl in neuron:
	neuron_zoomed.append(sl[::10])
neuron = prepare_data(neuron_zoomed)

print(len(neuron), len(neuron[0]), len(neuron[0]))

all_neuron_slices = []
for k in range(len(neuron)):
	neuron_slices = []
	offset = 0
	for i in range(int(len(neuron[k]) / 100)):
		neuron_slices_tmp = []
		for j in range(offset, offset + 100):
			neuron_slices_tmp.append(neuron[k][j])
		neuron_slices.append(neuron_slices_tmp)
		offset += 100
	all_neuron_slices.append(neuron_slices)
all_neuron_slices = list(zip(*all_neuron_slices)) # list [16][4][100]

print(len(all_neuron_slices), len(all_neuron_slices[0]), len(all_neuron_slices[0][0]))

for sl in range(len(all_neuron_slices)):
	all_neuron_slices[sl] = list(all_neuron_slices[sl])
	all_neuron_slices[sl] = sum(all_neuron_slices[sl], [])

print("all_neuron_slices = ", len(all_neuron_slices), len(all_neuron_slices[0]))
mono_neuron = []
for sl in range(len(all_neuron_slices)):
	mono_neuron.append([abs(a) for a in all_neuron_slices[sl][:ees_end]])

print("mono_neuron = ", mono_neuron)
latencies = get_lat_amp(neuron, herz, step)[0]
print("latencies = ", latencies)

poly_neuron = []
for sl in range(len(all_neuron_slices)):
	poly_neuron.append([abs(a) for a in all_neuron_slices[sl][int(latencies[sl] * 4):]])

high_whisker_neuron_mono = []
low_whisker_neuron_mono = []

for sl in mono_neuron:
	high_whisker_neuron_mono.append(calc_boxplots(sl)[3])
	low_whisker_neuron_mono.append(calc_boxplots(sl)[4])

whiskers_diff_mono_neuron = []
for sl in range(len(high_whisker_neuron_mono)):
	whiskers_diff_mono_neuron.append(high_whisker_neuron_mono[sl] - low_whisker_neuron_mono[sl])

print("high_whisker_neuron_mono = ", high_whisker_neuron_mono)
print("low_whisker_neuron_mono = ", low_whisker_neuron_mono)

high_whisker_neuron_poly = []
low_whisker_neuron_poly = []

for sl in poly_neuron:
	high_whisker_neuron_poly.append(calc_boxplots(sl)[3])
	low_whisker_neuron_poly.append(calc_boxplots(sl)[4])

whiskers_diff_poly_neuron = []
for sl in range(len(high_whisker_neuron_poly)):
	whiskers_diff_poly_neuron.append(high_whisker_neuron_poly[sl] - low_whisker_neuron_poly[sl])

gras = np.array(select_slices(gras_path, 10000, 22000))
gras = np.negative(gras)
gras_zoomed = []
for sl in gras:
	gras_zoomed.append(sl[::10])
gras = prepare_data(gras_zoomed)

print(len(gras))

all_gras_slices = []
for k in range(len(gras)):
	gras_slices = []
	offset = 0
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
	all_gras_slices[sl] = sum(all_gras_slices[sl], [])

mono_gras = []
for sl in range(len(all_gras_slices)):
	mono_gras.append([abs(a) for a in all_gras_slices[sl][:ees_end]])

print("mono_gras = ", mono_gras)
latencies = get_lat_amp(gras, herz, step)[0]
print("latencies = ", latencies)

poly_gras = []
for sl in range(len(all_gras_slices)):
	poly_gras.append([abs(a) for a in all_gras_slices[sl][int(latencies[sl] * 4):]])

high_whisker_gras_mono = []
low_whisker_gras_mono = []

for sl in mono_gras:
	high_whisker_gras_mono.append(calc_boxplots(sl)[3])
	low_whisker_gras_mono.append(calc_boxplots(sl)[4])

whiskers_diff_mono_gras = []
for sl in range(len(high_whisker_gras_mono)):
	whiskers_diff_mono_gras.append(high_whisker_gras_mono[sl] - low_whisker_gras_mono[sl])

print("high_whisker_gras_mono = ", high_whisker_gras_mono)
print("low_whisker_gras_mono = ", low_whisker_gras_mono)

high_whisker_gras_poly = []
low_whisker_gras_poly = []

for sl in poly_gras:
	high_whisker_gras_poly.append(calc_boxplots(sl)[3])
	low_whisker_gras_poly.append(calc_boxplots(sl)[4])

whiskers_diff_poly_gras = []
for sl in range(len(high_whisker_gras_poly)):
	whiskers_diff_poly_gras.append(high_whisker_gras_poly[sl] - low_whisker_gras_poly[sl])

print("high_whisker_gras_poly = ", high_whisker_gras_poly)
print("low_whisker_gras_poly = ", low_whisker_gras_poly)

diff_bio_neuron_mono = []
diff_bio_neuron_poly = []

for sl in range(len(whiskers_diff_mono)):
	diff_bio_neuron_mono.append(abs(whiskers_diff_mono[sl] - whiskers_diff_mono_neuron[sl]))
	diff_bio_neuron_poly.append(abs(whiskers_diff_poly[sl] - whiskers_diff_poly_neuron[sl]))

diff_bio_gras_mono = []
diff_bio_gras_poly = []

for sl in range(len(whiskers_diff_mono)):
	diff_bio_gras_mono.append(abs(whiskers_diff_mono[sl] - whiskers_diff_mono_gras[sl]))
	diff_bio_gras_poly.append(abs(whiskers_diff_poly[sl] - whiskers_diff_poly_gras[sl]))

print("diff_bio_gras_mono= ", diff_bio_gras_mono)
print("diff_bio_gras_poly = ", diff_bio_gras_poly)

df_mono = pd.DataFrame({'Bio - Neuron mono': diff_bio_neuron_mono})
df_poly = pd.DataFrame({'Bio - Neuron poly': diff_bio_neuron_poly})

color = ['#472650']
df_mono.plot(kind='bar', color=color)

plt.xticks(range(len(diff_bio_neuron_mono)), [i + 1 if i % 3 == 0 or i % 11 == 0    # or i % 5 == 0
                                              else "" for i in range(len(diff_bio_neuron_mono))],
           fontsize=56, rotation=0)
plt.gca().get_legend().remove()
plt.yticks(fontsize=56)

plt.show()

df_poly.plot(kind='bar', color=color)

plt.xticks(range(len(diff_bio_neuron_poly)), [i + 1 if i % 3 == 0 or i % 11 == 0    # or i % 5 == 0
                                              else "" for i in range(len(diff_bio_neuron_poly))],
           fontsize=56, rotation=0)
plt.gca().get_legend().remove()
plt.yticks(fontsize=56)

plt.show()
df_mono_gras = pd.DataFrame({'Bio - GRAS mono': diff_bio_gras_mono})
df_poly_gras = pd.DataFrame({'Bio - GRAS poly': diff_bio_gras_poly})

color = ['#472650']
df_mono_gras.plot(kind='bar', color=color)

plt.xticks(range(len(diff_bio_gras_mono)), [i + 1 if i % 3 == 0 or i % 11 == 0
                                              else "" for i in range(len(diff_bio_gras_mono))],
           fontsize=56, rotation=0)
plt.gca().get_legend().remove()
plt.yticks(fontsize=56)

plt.show()

df_poly_gras.plot(kind='bar', color=color)

plt.xticks(range(len(diff_bio_gras_poly)), [i + 1 if i % 3 == 0 or i % 11 == 0
                                              else "" for i in range(len(diff_bio_gras_poly))],
           fontsize=56, rotation=0)
plt.gca().get_legend().remove()
plt.yticks(fontsize=56)

plt.show()