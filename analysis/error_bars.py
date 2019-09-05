from analysis.functions import auto_prepare_data, calc_boxplots
from analysis.PCA import get_lat_per_exp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt

folder = '/home/anna/Desktop/data/bio/foot/'
filename = 'bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5'
step_size_to = 0.1
bio= auto_prepare_data(folder, filename, step_size_to)
print(len(bio), len(bio[0]), len(bio[0][0]))

lat= get_lat_per_exp(bio, 0.1)
lat = np.reshape(lat, (8, -1))
lat = lat.T
print(len(lat), len(lat[0]), lat)

bp_median = []
bp_high = []
bp_low = []
for sl in lat:
	bp_median.append(round(calc_boxplots(sl)[0]))
	bp_high.append(round(calc_boxplots(sl)[1]))
	bp_low.append(round(calc_boxplots(sl)[2]))

print("bp_low = ", bp_low)
print("bp_median =", bp_median)
print("bp_high = ", bp_high)
print('---')

bp_median = [int(bp /step_size_to) for bp in bp_median]
bp_high = [int(bp /step_size_to) for bp in bp_high]
bp_low = [int(bp /step_size_to) for bp in bp_low]

print("bp_low = ", bp_low)
print("bp_median =", bp_median)
print("bp_high = ", bp_high)
print('---')

amplitudes_from_low = []
amplitudes_from_median = []
amplitudes_from_high = []

for run in bio:
	amplitudes_from_low_tmp = []
	for sl in range(len(run)):
		amplitude_sum = np.sum(np.abs(run[sl][bp_low[sl]:]))
		amplitudes_from_low_tmp.append(amplitude_sum)
	amplitudes_from_low.append(amplitudes_from_low_tmp)
# print("amplitudes_from_low = ", len(amplitudes_from_low), len(amplitudes_from_low[0]), amplitudes_from_low)

amplitudes_from_low_sum = amplitudes_from_low[0]
for run in range(1, len(amplitudes_from_low)):
	# print("run = ", run)
	for i in range(len(amplitudes_from_low[run])):
		# print("i = ", i)
		amplitudes_from_low_sum[i] += amplitudes_from_low[run][i]
print("amplitudes_from_low_sum = ", len(amplitudes_from_low_sum), amplitudes_from_low_sum)

amplitudes_from_low = np.array(amplitudes_from_low)
# amplitudes_from_low = amplitudes_from_low.T
# print("amplitudes_from_low = ", len(amplitudes_from_low), len(amplitudes_from_low[0]))

for run in bio:
	amplitudes_from_median_tmp = []
	for sl in range(len(run)):
		amplitude_sum = np.sum(np.abs(run[sl][bp_median[sl]:]))
		amplitudes_from_median_tmp.append(amplitude_sum)
	amplitudes_from_median.append(amplitudes_from_median_tmp)
# print("amplitudes_from_median = ", len(amplitudes_from_median), len(amplitudes_from_median[0]), amplitudes_from_median)

amplitudes_from_median_sum = amplitudes_from_median[0]
for run in range(1, len(amplitudes_from_median)):
	# print("run = ", run)
	for i in range(len(amplitudes_from_median[run])):
		# print("i = ", i)
		amplitudes_from_median_sum[i] += amplitudes_from_median[run][i]
print("amplitudes_from_median_sum = ", len(amplitudes_from_median_sum), amplitudes_from_median_sum)

amplitudes_from_median = np.array(amplitudes_from_median)
# amplitudes_from_median = amplitudes_from_median.T
# print("amplitudes_from_median = ", len(amplitudes_from_median), len(amplitudes_from_median[0]))

for run in bio:
	amplitudes_from_high_tmp = []
	for sl in range(len(run)):
		amplitude_sum = np.sum(np.abs(run[sl][bp_high[sl]:]))
		amplitudes_from_high_tmp.append(amplitude_sum)
	amplitudes_from_high.append(amplitudes_from_high_tmp)

# print("amplitudes_from_high = ", len(amplitudes_from_high), len(amplitudes_from_high[0]), amplitudes_from_high)

amplitudes_from_high_sum = amplitudes_from_high[0]
for run in range(1, len(amplitudes_from_high)):
	# print("run = ", run)
	for i in range(len(amplitudes_from_high[run])):
		# print("i = ", i)
		amplitudes_from_high_sum[i] += amplitudes_from_high[run][i]
print("amplitudes_from_high_sum = ", len(amplitudes_from_high_sum), amplitudes_from_high_sum)
amplitudes_from_high = np.array(amplitudes_from_high)
# amplitudes_from_high = amplitudes_from_high.T
# print("amplitudes_from_high = ", len(amplitudes_from_high), len(amplitudes_from_high[0]))

xs = []
for i in range(len(bp_low)):
	xs.append(int(bp_low[i]/ 10))
	xs.append(int(bp_median[i] / 10))
	xs.append(int(bp_high[i] / 10))
print("xs = ", xs)

ys= []
for i in range(len(amplitudes_from_low_sum)):
	ys.append(amplitudes_from_low_sum[i])
	ys.append(amplitudes_from_median_sum[i])
	ys.append(amplitudes_from_high_sum[i])
print("ys = ", ys)

# df = pd.DataFrame({'xs': xs, 'ys':ys})
# print("df = ", df)
colors = ['black', 'dimgrey', 'darkgray', 'rosybrown', 'lightcoral', 'brown',
          'red', 'salmon', 'orangered', 'lightsalmon', 'sienna', 'chocolate']
plt.figure(figsize=(15, 5))
offset = 0
for i in range(len(bp_low)):
	xs_in = []
	ys_in= []
	for j in range(3):
		xs_in.append(xs[offset + j])
		ys_in.append(ys[offset + j])
		df = pd.DataFrame({'xs_in': xs_in, 'ys_in': ys_in})
		print("xs_in = ", xs_in)
		print("ys_in = ", ys_in)
		df.plot(x='xs_in', y='ys_in')
		# plt.scatter(xs[offset + j], ys[offset + j], color=colors[i])
		# plt.plot(xs[offset + j], ys[offset + j], 'xb-')
	offset += 3
# df.plot(x='xs', y='ys')
plt.show()
amplitudes = []
amplitudes.append(amplitudes_from_low.flatten())
amplitudes.append(amplitudes_from_median.flatten())
amplitudes.append(amplitudes_from_high.flatten())
amplitudes = np.array(amplitudes)
amplitudes = amplitudes.flatten()
print("amplitudes = ", len(amplitudes))

slices = []
for run in amplitudes_from_low:
	for sl in range(len(run)):
		slices.append(sl + 1)
slices = slices * 3
print("slices = ", len(slices), slices)

times = []
for run in range(len(amplitudes_from_low)):
	times.append([int(b / 10) for b in bp_low])

for run in range(len(amplitudes_from_median)):
	times.append([int(b / 10) for b in bp_median])

for run in range(len(amplitudes_from_high)):
	times.append([int(b / 10) for b in bp_high])

times = np.array(times).flatten()
# print("times = ", len(times), times)

df = pd.DataFrame({'Amplitudes': amplitudes, 'Slices': slices, 'Times': times})
boxplot = sns.boxplot(x='Times', y = 'Amplitudes', hue='Slices', data=df)
plt.xticks(range(len(amplitudes_from_median)), [i if i % 5 == 0 else "" for i in range(len(amplitudes_from_median))])
plt.show()