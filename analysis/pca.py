import numpy as np
from matplotlib import pylab as plt
from analysis.functions import changing_peaks
from analysis.cut_several_steps_files import select_slices
from GRAS.PCA import prepare_data
import h5py as hdf5

sim_step = 0.025
bio_step = 0.25

# bio_data = bio_data_runs()
filepath = '../bio-data/hdf5/bio_sci_E_15cms_40Hz_i100_2pedal_no5ht_T_2016-06-12.hdf5'
def read_data(filepath, sign=1):
	with hdf5.File(filepath) as file:
		data_by_test = [sign * test_values[:] for test_values in file.values()]
	return data_by_test

bio_data = read_data(filepath, sign=1)
bio_data = prepare_data(bio_data)

bio_slices = []
offset = 0
for i in range(int(len(bio_data[0]) / 100)):
	bio_slices_tmp = []
	for j in range(offset, offset + 100):
		bio_slices_tmp.append(bio_data[0][j])
	offset += 100
	bio_slices.append(bio_slices_tmp)

neuron_list = np.array(select_slices('../../neuron-data/mn_E15_speed25tests.hdf5', 0, 12000))
neuron_list = np.negative(neuron_list)
neuron_list = prepare_data(neuron_list)

neuron_list_zoomed = []
for sl in neuron_list:
	neuron_list_zoomed_tmp = []
	for i in range(0, len(sl), 10):
		neuron_list_zoomed_tmp.append(sl[i])
	neuron_list_zoomed.append(neuron_list_zoomed_tmp)
neuron_run_zoomed = []
for i in range(0, len(neuron_list[0]), 10):
	neuron_run_zoomed.append(neuron_list[0][i])

neuron_slices = []
offset = 0
for sl in range(int(len(neuron_run_zoomed) / 100)):
	neuron_slices_tmp = []
	for i in range(offset, offset + 100):
		neuron_slices_tmp.append(neuron_run_zoomed[i])
	offset += 100
	neuron_slices.append(neuron_slices_tmp)

gras_list = np.array(select_slices('/home/anna/PycharmProjects/LAB/memristive-spinal-cord/GRAS/matrix_solution/bio_data/gras_15.hdf5', 10000, 22000))
gras_list = prepare_data(gras_list)

gras_list_zoomed = []
for sl in gras_list:
	gras_list_zoomed_tmp = []
	for i in range(0, len(sl), 10):
		gras_list_zoomed_tmp.append(sl[i])
	gras_list_zoomed.append(gras_list_zoomed_tmp)

gras_run_zoomed = []
for i in range(0, len(gras_list[0]), 10):
	gras_run_zoomed.append(gras_list[0][i])

gras_slices = []
offset = 0
for sl in range(int(len(gras_run_zoomed) / 100)):
	gras_slices_tmp = []
	for i in range(offset, offset + 100):
		gras_slices_tmp.append(gras_run_zoomed[i])
	offset += 100
	gras_slices.append(gras_slices_tmp)

print("len(gras_slices) = ", len(gras_slices))
print("len(gras_slices[0]) = ", len(gras_slices[0]))

for index, sl in enumerate(neuron_slices):
	offset = index
	plt.plot([s + offset for s in sl])
plt.show()

for index, sl in enumerate(gras_slices):
	offset = index
	plt.plot([s + offset for s in sl])
plt.show()

latencies, indexes_max, indexes_min, corr_ampls_max, corr_ampls_min = changing_peaks(neuron_list_zoomed, 40, bio_step)
yticks = []

max_peaks = []
for ind in indexes_max:
	max_peaks.append(len(ind))

min_peaks = []
for ind in indexes_min:
	min_peaks.append(len(ind))

sum_peaks = []
for i in range(len(min_peaks)):
	sum_peaks.append(max_peaks[i] + min_peaks[i])

all_peaks_sum = sum(sum_peaks)
latencies = [int(l / bio_step) for l in latencies]
# print("latencies = ", latencies)
for index, sl in enumerate(neuron_slices):
	offset = index * 0.5
	plt.plot([s + offset for s in sl])
	yticks.append(sl[0] + offset)
	plt.plot(latencies[index], neuron_list[0][latencies[index]] + offset, '.', color='k', markersize=24)
	plt.text(latencies[index], neuron_list[0][latencies[index]] + offset, round(latencies[index] * bio_step, 1),
	         color='green', fontsize=16)
	plt.plot([m for m in indexes_max[index]], [m + offset for m in corr_ampls_max[index]], 's', color='red',
	         markersize=9)
	plt.plot([m for m in indexes_min[index]], [m + offset for m in corr_ampls_min[index]], 's', color='blue',
	         markersize=9)
	plt.text(sl[10], sl[0] + offset, sum_peaks[index], fontsize=16)

ticks = []
labels = []
for i in range(0, len(neuron_slices[0]) + 1, 4):
	ticks.append(i)
	labels.append(i / 4)
plt.yticks(yticks, range(1, len(neuron_slices) + 1), fontsize=14)
# print("yticks = ", yticks)
plt.xticks(ticks, [int(i) for i in labels], fontsize=14)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.xlim(0, 100)
latencies = [round(l * sim_step , 1)for l in latencies]
plt.title("Neuron Peaks sum = {}".format(all_peaks_sum))
plt.show()

latencies_bio, bio_indexes_max, bio_indexes_min, bio_corr_ampls_max, bio_corr_ampls_min = \
	changing_peaks(bio_data, 40, bio_step, max_amp_coef=0, min_amp_coef=-3)
max_peaks = []
for ind in bio_indexes_max:
	max_peaks.append(len(ind))
print("max_peaks = ", max_peaks)

min_peaks = []
for ind in bio_indexes_min:
	min_peaks.append(len(ind))
print("min_peaks = ", min_peaks)

sum_peaks = []
for i in range(len(min_peaks)):
	sum_peaks.append(max_peaks[i] + min_peaks[i])
print("sum_peaks = ", sum_peaks)

all_peaks_sum = sum(sum_peaks)

yticks = []
latencies_bio = [int(l / bio_step) for l in latencies_bio]
print("latencies_bio = ", latencies_bio)
for index, sl in enumerate(bio_slices):
	offset = index * 0.5
	plt.plot([s + offset for s in sl])
	yticks.append(sl[0] + offset)
	plt.plot(latencies_bio[index], sl[latencies_bio[index]] + offset, '.', color='k', markersize=24)
	plt.text(latencies_bio[index], sl[latencies_bio[index]] + offset, round(latencies_bio[index] * bio_step,1),
	         color='green', fontsize=16)
	plt.plot([m for m in bio_indexes_max[index]], [m + offset for m in bio_corr_ampls_max[index]], 's', color='red',
	         markersize=9)
	plt.plot([m for m in bio_indexes_min[index]], [m + offset for m in bio_corr_ampls_min[index]], 's', color='blue',
	         markersize=9)
	plt.text(sl[0] + 2, sl[0] + offset, sum_peaks[index], fontsize=16)

ticks = []
labels = []
for i in range(0, len(bio_slices[0]) + 1, 4):
	ticks.append(i)
	labels.append(i / 4)
plt.yticks(yticks, range(1, len(bio_slices) + 1), fontsize=14)
plt.xticks(ticks, [int(i) for i in labels], fontsize=14)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.xlim(0, 100)
latencies = [round(l * bio_step , 1)for l in latencies_bio]
plt.title("Bio Peaks sum = {}".format(all_peaks_sum))
plt.show()

latencies, indexes_max, indexes_min, corr_ampls_max, corr_ampls_min = \
	changing_peaks(gras_list_zoomed, 40, bio_step)
yticks = []

max_peaks = []
for ind in indexes_max:
	max_peaks.append(len(ind))

min_peaks = []
for ind in indexes_min:
	min_peaks.append(len(ind))

sum_peaks = []
for i in range(len(min_peaks)):
	sum_peaks.append(max_peaks[i] + min_peaks[i])

all_peaks_sum = sum(sum_peaks)
latencies = [int(l / bio_step) for l in latencies]
# print("latencies = ", latencies)
for index, sl in enumerate(gras_slices):
	offset = index * 0.5
	plt.plot([s + offset for s in sl])
	yticks.append(sl[0] + offset)
	plt.plot(latencies[index], neuron_list[0][latencies[index]] + offset, '.', color='k', markersize=24)
	plt.text(latencies[index], neuron_list[0][latencies[index]] + offset, round(latencies[index] * bio_step, 1),
	         color='green', fontsize=16)
	plt.plot([m for m in indexes_max[index]], [m + offset for m in corr_ampls_max[index]], 's', color='red',
	         markersize=9)
	plt.plot([m for m in indexes_min[index]], [m + offset for m in corr_ampls_min[index]], 's', color='blue',
	         markersize=9)
	plt.text(sl[10], sl[0] + offset, sum_peaks[index], fontsize=16)

ticks = []
labels = []
for i in range(0, len(neuron_slices[0]) + 1, 4):
	ticks.append(i)
	labels.append(i / 4)
plt.yticks(yticks, range(1, len(neuron_slices) + 1), fontsize=14)
# print("yticks = ", yticks)
plt.xticks(ticks, [int(i) for i in labels], fontsize=14)
plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
plt.xlim(0, 100)
latencies = [round(l * sim_step , 1)for l in latencies]
plt.title("GRAS Peaks sum = {}".format(all_peaks_sum))
plt.show()

raise Exception
neuron_slices = []
for run in neuron_list:
	neuron_slices_list = []
	for i in range(int(len(run) / 1000)):
		offset = 0
		neuron_slices_tmp = []
		for j in range(offset, offset + 1000):
			neuron_slices_tmp.append(run[j])
		offset += 1000
		neuron_slices_list.append(neuron_slices_tmp)
	neuron_slices.append(neuron_slices_list)
neuron_means = np.sum(np.array([np.absolute(normalization(data, -1, 1)) for data in neuron_list]), axis=0)

# calculating latencies and amplitudes of mean values
neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=True)[0]
neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=True, after_latencies=True)[1]

neuron_peaks_number = []
max_times_slices = []
min_times_slices = []
max_values_slices = []
min_values_slices = []

count = 0
for index, run in enumerate(neuron_list):
	# print("index = ", index)
	try:
		neuron_peaks_number_run = sim_process(run, sim_step, inhibition_zero=True, after_latencies=True)[2]
		max_times_run = sim_process(run, sim_step, inhibition_zero=True, after_latencies=True)[3]
		min_times_run = sim_process(run, sim_step, inhibition_zero=True, after_latencies=True)[4]
		max_values_run = sim_process(run, sim_step, inhibition_zero=True, after_latencies=True)[5]
		min_values_run = sim_process(run, sim_step, inhibition_zero=True, after_latencies=True)[6]

		count += 1
	except IndexError:
		continue

	neuron_peaks_number.append(neuron_peaks_number_run)
	max_times_slices.append(max_times_run)
	min_times_slices.append(min_times_run)
	max_values_slices.append(max_values_run)
	min_values_slices.append(min_values_run)

# print("count = ", count)
# print("len(neuron_peaks_number) = ", len(neuron_peaks_number))
# print("len(max_times_slices) = ", len(max_times_slices))
# print("len(min_times_slices) = ", len(min_times_slices))
# print("len(max_values_slices) = ", len(max_values_slices))
# print("len(min_values_slices) = ", len(min_values_slices))

# for r in max_times_slices:
# 	print("max_times_slices = ", len(r))
# for r in min_times_slices:
# 	print("min_times_slices = ", len(r))
# for r in max_values_slices:
# 	print("max_values_slices = ", len(r))
# for r in min_values_slices:
# 	print("min_values_slices = ", len(r))
# for run in neuron_peaks_number:
	# print("run = ", len(run), run)

max_times = []
for j in max_times_slices:
	max_times_tmp = []
	for i in j:
		max_times_tmp += j
	max_times.append(max_times_tmp)
next_slice = []
for i in range(len(max_times)):
	next_slice_tmp = []
	for j in range(1, len(max_times[i]) - 1):
		next_slice_tmp.append([sum(pair) for pair in zip(max_times[i][j - 1], max_times[i][j])])
	next_slice.append(next_slice_tmp)
print(len(next_slice))
for n in next_slice:
	print("next slice = ", len(n))
min_times = []
for i in min_times_slices:
	min_times += i

max_values = []
for i in max_values_slices:
	max_values += i

min_values = []
for i in min_values_slices:
	min_values += i

# print("len(max_times) = ", len(max_times), max_times)
sum_neuron_peaks_number = [sum(sl) / count for sl in zip(*neuron_peaks_number)]

int_sum_neuron_peaks_number = []
for s in sum_neuron_peaks_number:
	int_sum_neuron_peaks_number.append(round(s))

# print("sum_neuron_peaks_number = ", sum_neuron_peaks_number)
# print("int_sum_neuron_peaks_number = ", int_sum_neuron_peaks_number)

for index in range(len(neuron_list)):
	offset = index
	for sl in range(len(neuron_list)):
		plt.plot(neuron_slices[index][sl], color='k')
		plt.plot([m for m in max_times_slices[index][sl]], [m + offset for m in max_values_slices[index][sl]],
	             marker='.', markersize=6, color='red')
		plt.plot([m for m in min_times_slices[index][sl]], [m + offset for m in min_values_slices[index][sl]],
	             marker='.', markersize=6, color='blue')
		plt.show()

bio_means_lat = sim_process(bio_data, bio_step, inhibition_zero=True)[0]
bio_means_amp = sim_process(bio_data, bio_step, inhibition_zero=True, after_latencies=True)[1]
# print("bio_means_lat = ", len(bio_means_lat))
# print("neuron_means_lat = ", neuron_means_lat)
# print("neuron_means_amp = ", neuron_means_amp)

slices = []
for i in range(len(neuron_means_lat)):
	slices.append(i + 1)

neuron_sum_of_peaks = []

slices_bio = []
for i in range(len(bio_means_lat)):
	slices_bio.append(i + 1)

bio_sum_of_peaks = []

slices_nparray = np.array([np.array(x) for x in slices])
neuron_lat_nparray = np.array([np.array(x) for x in neuron_means_lat])
neuron_amp_nparray = np.array([np.array(x) for x in neuron_means_amp])
neuron_sum_peaks_nparray = np.array([np.array(x) for x in int_sum_neuron_peaks_number])

bio_slices_nparray = np.array([np.array(x) for x in slices_bio])
bio_lat_nparray = np.array([np.array(x) for x in bio_means_lat])
bio_amp_nparray = np.array([np.array(x) for x in bio_means_amp])
print("len(bio_slices_nparray) = ", len(bio_slices_nparray), bio_slices_nparray)

# neuron_data = read_neuron_data('../../neuron-data/15cm.hdf5')
# neuron_data = neuron_data[:1]
# cutted_neuron = []
# for run in neuron_data:
# 	cutted_neuron_run = []
# 	for i in run[::10]:
# 		cutted_neuron_run.append(i)
# 	cutted_neuron.append(cutted_neuron_run)
# print("cutted_neuron = ", len(cutted_neuron), len(cutted_neuron[0]))
slices_nparray = slices_nparray.T
neuron_amp_nparray = neuron_amp_nparray.T
neuron_lat_nparray = neuron_lat_nparray.T
neuron_sum_peaks_nparray = neuron_sum_peaks_nparray.T

# bio_slices_nparray = bio_slices_nparray.T
bio_amp_nparray = bio_amp_nparray.T
bio_lat_nparray = bio_lat_nparray.T
slices_nparray = np.reshape(slices_nparray, (len(slices_nparray), 1))
neuron_amp_nparray = np.reshape(neuron_amp_nparray, (len(neuron_means_lat), 1))
neuron_lat_nparray = np.reshape(neuron_lat_nparray, (len(neuron_means_lat), 1))
neuron_sum_peaks_nparray = np.reshape(neuron_sum_peaks_nparray, (len(neuron_sum_peaks_nparray), 1))

print("len(bio_slices_nparray) = ", bio_slices_nparray)
bio_slices_nparray = np.reshape(bio_slices_nparray, (12, 1))
bio_amp_nparray = np.reshape(bio_amp_nparray, (len(bio_means_amp), 1))
bio_lat_nparray = np.reshape(bio_lat_nparray, (len(bio_means_amp), 1))

# neuron_np_array = np.array([np.array(x) for x in cutted_neuron])
# neuron_np_array = neuron_np_array.T
# bio_np_array = np.reshape(bio_np_array, (1200, 1))
# neuron_np_array = np.reshape(neuron_np_array, (1200, 1))
neuron_data = np.hstack((neuron_sum_peaks_nparray, neuron_amp_nparray, neuron_lat_nparray))
bio_data = np.hstack((bio_slices_nparray, bio_amp_nparray, bio_lat_nparray))

yticks = []
# for index, run in enumerate(bio_slices):
# 	offset = index * 5
# 	times = [time * bio_step for time in range(len(run))]
	# plt.plot(times, [r + offset for r in run ])
	# yticks.append(run[0] + offset)
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
# plt.yticks(yticks, range(1, len(bio_means) + 1), fontsize=14)
# plt.xlim(0, 25)
# plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
# plt.show()
processed_data = neuron_data

mu = processed_data.mean(axis=0)
data = processed_data - mu
eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
projected_data = np.dot(data, eigenvectors)
sigma = projected_data.std(axis=0).mean()
print("eigenvectors = ", len(eigenvectors), len(eigenvectors[0]), eigenvectors)

points = go.Scatter3d(x =int_sum_neuron_peaks_number, y=neuron_means_amp, z=neuron_means_lat, mode='markers',
                      marker=dict(size=2, color="rgb(227, 26, 28)"), name='points')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig, ax = plt.subplots()
ax.scatter(int_sum_neuron_peaks_number, neuron_amp_nparray, neuron_lat_nparray)
starts = []
ends = []
for axis in eigenvectors:
	start, end = mu, mu + sigma * axis
	print("start = ", len(start), start)
	print("end = ", len(end), end)
	starts.append(list(start))
	ends.append(list(end))
print("starts = ", starts)
print("ends = ", ends)

x_vector1 = [starts[0][0], ends[0][0]]
y_vector1 = [starts[0][1], ends[0][1]]
z_vector1 = [starts[0][2], ends[0][2]]
x_vector2 = [starts[1][0], ends[1][0]]
y_vector2 = [starts[1][1], ends[1][1]]
z_vector2 = [starts[1][2], ends[1][2]]
x_vector3 = [starts[2][0], ends[2][0]]
y_vector3 = [starts[2][1], ends[2][1]]
z_vector3 = [starts[2][2], ends[2][2]]
vector1 = go.Scatter3d(x=x_vector1, y=y_vector1, z=z_vector1, marker=dict(size=1, color="rgb(84, 48, 5)"),
                      line=dict(color="rgb(242, 227, 19)", width=6), name='Sum of peaks')
vector2 = go.Scatter3d(x=x_vector2, y=y_vector2, z=z_vector2, marker=dict(size=1, color="rgb(84, 48, 5)"),
                      line=dict(color="rgb(84, 88, 111)", width=6), name='Amplitudes')
vector3 = go.Scatter3d(x=x_vector3, y=y_vector3, z=z_vector3, marker=dict(size=1, color="rgb(84, 48, 5)"),
                      line=dict(color="rgb(71, 242,19)", width=6), name='Latencies')
data = [points, vector1, vector2, vector3]

layout = go.Layout(xaxis=dict(title='Sum of peaks', titlefont=dict(family='Arial, sans-serif', size=18, color='black')),
                   yaxis=dict(title='Amplitudes', titlefont=dict(family='Arial, sans-serif', size=18, color='black')))
fig = go.Figure(data=data, layout=layout)
plot(fig, filename="pca3d.html", auto_open=True, image='png', image_height=800, image_width=3000)
ax.annotate(
		'', xy=end, xycoords='data',
		xytext=start, textcoords='data',
		arrowprops=dict(facecolor='red', width=2.0)
		)
ax.set_aspect('equal')
plt.show()