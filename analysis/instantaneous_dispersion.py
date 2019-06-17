from analysis.patterns_in_bio_data import bio_data_runs
from matplotlib import pylab as plt
from cycler import cycler
from analysis.functions import sim_process, normalization
from analysis.cut_several_steps_files import select_slices

# importing the list of all runs of the bio  data from the function 'bio_data_runs'
bio_runs = bio_data_runs()
sim_runs = select_slices('../../neuron-data/mn_E25tests.hdf5', 0, 6000)
print("len(sim_runs) = ", len(sim_runs))
print("len(sim_runs) = ", len(sim_runs[0]))
for run in sim_runs:
	print("sim_runs = ", len(run), run)
# for i in range(len(bio_runs)):
# 	bio_runs[i] = normalization(bio_runs[i], a, b)
offset = 0
all_bio_slices = []
step = 0.25
sim_step = 0.025
interslice_coef = 16

# forming list for the plot
for k in range(len(bio_runs)):
	bio_slices = []
	offset = 0
	for i in range(int(len(bio_runs[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio_runs[k][j])
		bio_slices.append(normalization(bio_slices_tmp, -1, 1))
		offset += 100
	all_bio_slices.append(bio_slices)   # list [4][16][100]
# print("all_bio_slices = ", all_bio_slices)
all_bio_slices = list(zip(*all_bio_slices)) # list [16][4][100]

all_sim_slices = []
for k in range(len(sim_runs)):
	if sim_runs[k]:
		sim_slices = []
		offset = 0
		for i in range(int(len(sim_runs[k]) / 1000)):
			sim_slices_tmp = []
			for j in range(offset, offset + 1000):
				sim_slices_tmp.append(sim_runs[k][j])
			sim_slices.append(normalization(sim_slices_tmp, -1, 1))
			offset += 1000
		all_sim_slices.append(sim_slices)

all_sim_slices = list(zip(*all_sim_slices))

print("all_sim_slices =", len(all_sim_slices))
print("all_sim_slices =", len(all_sim_slices[0]))
print("all_sim_slices =", len(all_sim_slices[0][0]))

# calculating the instant dispersion
colors = ['black', 'rosybrown', 'firebrick', 'sandybrown', 'gold', 'olivedrab', # 6
          'darksalmon', 'green', 'sienna', 'darkblue', 'coral', 'orange',   # 12
          'darkkhaki', 'red', 'tan', 'steelblue', 'darkgreen', 'darkblue', 'palegreen', 'k', 'forestgreen',   # 21
          'slategray', 'limegreen', 'dimgrey', 'darkorange', 'darkgreen', 'cornflowerblue', 'dimgray', 'burlywood', # 29
          'royalblue', 'grey', 'g', 'gray', 'lime', 'midnightblue', 'seagreen', 'navy']   # 37
plt.rc('axes', prop_cycle=cycler(color=colors))

instant_mean = []
for slice in range(len(all_bio_slices)):
	instant_mean_sum = []
	for dot in range(len(all_bio_slices[slice][0])):
		instant_mean_tmp = []
		for run in range(len(all_bio_slices[slice])):
			instant_mean_tmp.append(abs(all_bio_slices[slice][run][dot]))
		instant_mean_sum.append(sum(instant_mean_tmp))
	instant_mean.append(instant_mean_sum)

print("len(instant_mean) = ", len(instant_mean), len(instant_mean[0]))
print("instant_mean = ", instant_mean)

instant_mean_sim = []
for slice in range(len(all_sim_slices)):
	instant_mean_sim_sum = []
	for dot in range(len(all_sim_slices[slice][0])):
		instant_mean_sim_tmp = []
		for run in range(len(all_sim_slices[slice])):
			instant_mean_sim_tmp.append(abs(all_sim_slices[slice][run][dot]))
		instant_mean_sim_sum.append(sum(instant_mean_sim_tmp))
	instant_mean_sim.append(instant_mean_sim_sum)

print("len(instant_mean_sim) = ", len(instant_mean_sim), len(instant_mean_sim[0]))
print("instant_mean_sim = ", instant_mean_sim)

# maxes = []
# for sli in instant_mean:
# 	maxes.append(max(sli))

# shifts = []
# for m in maxes:
# 	shifts.append(0.105 * m)

# creating the list of dots of stimulations

# stimulations = []
# for stim in range(0, 1201, 100):
# 	stimulations.append(stim)

# creating the lists of voltages
volts = []
for i in instant_mean:
	for j in i:
		volts.append(j)

volts_sim = []
for i in instant_mean_sim:
	for j in i:
		volts_sim.append(j)

# derivatives
derivatives = []
for i in range(1, len(volts)):
	derivatives.append((volts[i] - volts[i - 1]) / step)
print("derivatives = ", len(derivatives), derivatives)

der_slices = []
der_first_slice = []
der_last_slice = []
for i in range(99):
	der_first_slice.append(derivatives[i])
print("der_first_slice = ", der_first_slice)

for i in range(500, len(derivatives)):
	der_last_slice.append(derivatives[i])
print("der_last_slice = ", der_last_slice)

offset = 99
for i in range(4):
	der_slice_tmp = []
	for j in range(offset, offset + 100):
		der_slice_tmp.append(derivatives[j])
	print("der_slice_tmp = ", der_slice_tmp)
	offset += 100
	der_slices.append(der_slice_tmp)

der_slices = [der_first_slice] + der_slices + [der_last_slice]
for slice in der_slices:
	print("der_slices = ", slice)

# derivatives for sim data
sim_derivatives = []
for i in range(1, len(volts_sim)):
	sim_derivatives.append((volts_sim[i] - volts_sim[i - 1]) / sim_step)
print("sim derivatives = ", len(sim_derivatives), sim_derivatives)

der_sim_slices = []
der_sim_first_slice = []
der_sim_last_slice = []
for i in range(1, 999):
	der_sim_first_slice.append(sim_derivatives[i])
print("der_sim first_slice = ", len(der_sim_first_slice), der_sim_first_slice)

for i in range(5001, len(sim_derivatives)):
	der_sim_last_slice.append(sim_derivatives[i])
print("der sim_last_slice = ", len(der_sim_last_slice), der_sim_last_slice)

offset = 1000
for i in range(4):
	der_slice_tmp = []
	for j in range(offset, offset + 1000):
		der_slice_tmp.append(sim_derivatives[j])
	print("der_slice_tmp = ", len(der_slice_tmp), der_slice_tmp)
	offset += 1000
	der_sim_slices.append(der_slice_tmp)

der_sim_slices = [der_sim_first_slice] + der_sim_slices + [der_sim_last_slice]
for slice in der_sim_slices:
	print("der sim_slices = ", len(slice), slice)
# list for latencies' finding
# volts_and_stims = [volts, stimulations]

# latencies finding

for index, sl in enumerate(der_sim_slices):
	offset = index * 32
	# plt.plot([s + offset for s in sl])
# plt.show()

latencies_sim = sim_process(volts_sim, sim_step, inhibition_zero=True)[0]
print("latencies_sim = ", latencies_sim)
# raise Exception
latencies = sim_process(volts, step, inhibition_zero=True)[0]
print("latencies = ", latencies)
# print("latencies = ", latencies)
yticks = []
# color_number = 0
# for index, sl in enumerate(all_bio_slices):
# 	offset = index * 16
	# print("sl[{}][0]".format(run), sl[run][0])
	# times = [time * step for time in range(len(all_bio_slices[0][0]))]
	# for run in range(len(sl)):
		# plt.plot(times, [s + offset for s in sl[run]], color=colors[color_number], linewidth=1)
	#  this think draws a lot of slices
	# color_number += 1
	# yticks.append(sl[run][0] + offset)

# color_number = 12
# yticks = []

sim_latency_x = []
for slice in range(len(instant_mean_sim)):
	for dot in range(int(latencies_sim[slice] * interslice_coef), 36, -1):
	# 	print("dot = ", dot / 4)
		# print("instant_mean[{}][{}] = ".format(slice, dot), instant_mean[slice][dot])
		# if instant_mean[slice][dot] < necessary_points[count]:
		# 	necessary_latencies.append(instant_mean[slice][dot])
			sim_latency_x.append(dot / interslice_coef)
print("sim_latency_x = ", len(sim_latency_x), sim_latency_x)
x_coor = []
y_coor = []

for index, sl in enumerate(instant_mean_sim):
	print("len(sl) = ", len(sl))
	offset = index * interslice_coef
	times = [time * sim_step for time in range(len(sl))]
	# plt.plot(times, [s + offset for s in sl], linewidth=2, color='green')
	yticks.append(sl[0] + offset)
	# plt.plot(latencies_sim[index], sl[int(latencies_sim[index] / sim_step)] + offset, marker='.', markersize=12,
	#          color='green')
	plt.plot([s + offset for s in der_sim_slices[index]], color='black')
	# pltting of the lines
	x_coor.append(latencies_sim[index])
	y_coor.append(sl[int(latencies_sim[index] / sim_step)] + offset)
	x_2_coors = []
	y_2_coors = []
	if len(x_coor) > 1:
		x_2_coors.append(x_coor[-2])
		x_2_coors.append(x_coor[-1])
		y_2_coors.append(y_coor[-2])
		y_2_coors.append(y_coor[-1])
		# plt.plot(x_2_coors, y_2_coors, linestyle='--', color='green')
print("yticks sim= ", yticks)

# times = []
# for index, sl in enumerate(slices):
# 	offset = index * 16 + 8
	# yticks.append(sl[0] + offset)
	# times = [time * step for time in range(len(sl))]
	# plt.plot(times, [s + offset for s in sl])
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)])
# plt.yticks(yticks, range(1, len(slices) + 1))
# plt.xlim(0, 25)
# plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
# plt.show()

# plotting the dispersion
# yticks = []

# creating the list of x & y coordinates of the lines
# x_coor = []
# y_coor = []
# plotting of the dots that show the latencies

# latencies_values = []
# for index, sl in enumerate(instant_mean):
	# latencies_values.append(sl[int(latencies[index] / step)])

# necessary_points = []
# for dot in range(len(latencies_values)):
# 	necessary_points.append(latencies_values[dot]- shifts[dot]) #

# necessary_latencies = []
# count = 0
latency_x = []
for slice in range(len(instant_mean)):
	for dot in range(int(latencies[slice] * interslice_coef), 36, -1):
	# 	print("dot = ", dot / 4)
		# print("instant_mean[{}][{}] = ".format(slice, dot), instant_mean[slice][dot])
		# if instant_mean[slice][dot] < necessary_points[count]:
		# 	necessary_latencies.append(instant_mean[slice][dot])
			latency_x.append(dot / interslice_coef)
		# 	print("latency_x = ", latency_x)
		# 	break
	# count += 1
	# if len(latency_x) != count:
	# 	latency_x.append(24.75)
yticks = []
for index, sl in enumerate(instant_mean):
	offset = index * interslice_coef
	times = [time * step for time in range(len(sl))]
	plt.plot(times, [s + offset for s in sl], linewidth=2, color='red')
	yticks.append(sl[0] + offset)
	plt.plot(latencies[index], sl[int(latency_x[index] / step)] + offset, marker='.', markersize=12, color='red')
	plt.plot([s + offset for s in der_slices[index]], color='black')
	# pltting of the lines
	x_coor.append(latencies[index])
	y_coor.append(sl[int(latencies[index] / step)] + offset)
	x_2_coors = []
	y_2_coors = []
	if len(x_coor) > 1:
		x_2_coors.append(x_coor[-2])
		x_2_coors.append(x_coor[-1])
		y_2_coors.append(y_coor[-2])
		y_2_coors.append(y_coor[-1])
		plt.plot(x_2_coors, y_2_coors, linestyle='--', color='red')
# print("yticks bio = ", yticks)
plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)])
plt.yticks(yticks, range(1, len(instant_mean_sim) + 1))
plt.xlim(0, 25)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.show()