import numpy as np
from matplotlib.mlab import PCA
from analysis.patterns_in_bio_data import bio_data_runs
from matplotlib import pylab as plt
from analysis.functions import read_neuron_data
from analysis.histogram_lat_amp import sim_process
from mpl_toolkits.mplot3d import axes3d, Axes3D
import plotly.graph_objs as go
from plotly.offline import plot
from analysis.cut_several_steps_files import select_slices

bio_data = bio_data_runs()
bio_np_array = np.array([np.array(x) for x in bio_data])

bio_np_array = bio_np_array.T
bio_data = bio_data_runs()
print("bio_data = ", len(bio_data), len(bio_data[0]))
bio_means = list(map(lambda voltages: np.mean(voltages), zip(*bio_data)))
sim_step = 0.025
bio_step = 0.25

bio_slices = []
offset = 0
for j in range(int(len(bio_means) / 100)):
	bio_slices_tmp = []
	for i in range(offset, offset + 100):
		bio_slices_tmp.append(bio_means[i])
	offset += 100
	bio_slices.append(bio_slices_tmp)
print("bio_slices = ", len(bio_slices), len(bio_slices[0]))
print("bio_means = ", len(bio_means))

neuron_list = select_slices('../../neuron-data/3steps_speed15_EX.hdf5', 17000, 29000)
neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))
# calculating latencies and amplitudes of mean values
neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=True)[0]
neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=True)[1]

bio_means_lat = sim_process(bio_means, bio_step, inhibition_zero=True)[0]
bio_means_amp = sim_process(bio_means, bio_step, inhibition_zero=True)[1]
print("bio_means_lat = ", len(bio_means_lat))
print("neuron_means_lat = ", neuron_means_lat)
print("neuron_means_amp = ", neuron_means_amp)

slices = []
for i in range(len(neuron_means_lat)):
	slices.append(i + 1)

slices_bio = []
for i in range(len(bio_means_lat)):
	slices_bio.append(i + 1)

slices_nparray = np.array([np.array(x) for x in slices])
neuron_lat_nparray = np.array([np.array(x) for x in neuron_means_lat])
neuron_amp_nparray = np.array([np.array(x) for x in neuron_means_amp])

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

# bio_slices_nparray = bio_slices_nparray.T
bio_amp_nparray = bio_amp_nparray.T
bio_lat_nparray = bio_lat_nparray.T
slices_nparray = np.reshape(slices_nparray, (len(slices_nparray), 1))
neuron_amp_nparray = np.reshape(neuron_amp_nparray, (len(neuron_means_lat), 1))
neuron_lat_nparray = np.reshape(neuron_lat_nparray, (len(neuron_means_lat), 1))

print("len(bio_slices_nparray) = ", bio_slices_nparray)
bio_slices_nparray = np.reshape(bio_slices_nparray, (12, 1))
bio_amp_nparray = np.reshape(bio_amp_nparray, (len(bio_means_amp), 1))
bio_lat_nparray = np.reshape(bio_lat_nparray, (len(bio_means_amp), 1))

# neuron_np_array = np.array([np.array(x) for x in cutted_neuron])
# neuron_np_array = neuron_np_array.T
# bio_np_array = np.reshape(bio_np_array, (1200, 1))
# neuron_np_array = np.reshape(neuron_np_array, (1200, 1))
neuron_data = np.hstack((slices_nparray, neuron_amp_nparray, neuron_lat_nparray))
bio_data = np.hstack((bio_slices_nparray, bio_amp_nparray, bio_lat_nparray))

yticks = []
for index, run in enumerate(bio_slices):
	offset = index * 5
	times = [time * bio_step for time in range(len(run))]
	# plt.plot(times, [r + offset for r in run ])
	# yticks.append(run[0] + offset)
# plt.xticks(range(26), [i if i % 1 == 0 else "" for i in range(26)], fontsize=14)
# plt.yticks(yticks, range(1, len(bio_means) + 1), fontsize=14)
# plt.xlim(0, 25)
# plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
# plt.show()
processed_data = bio_data

mu = processed_data.mean(axis=0)
data = processed_data - mu
eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
projected_data = np.dot(data, eigenvectors)
sigma = projected_data.std(axis=0).mean()
print("eigenvectors = ", len(eigenvectors), len(eigenvectors[0]), eigenvectors)

points = go.Scatter3d(x =slices, y=bio_means_amp, z=bio_means_lat, mode='markers',
                      marker=dict(size=2, color="rgb(227, 26, 28)"), name='points')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig, ax = plt.subplots()
ax.scatter(slices_nparray, neuron_amp_nparray, neuron_lat_nparray)
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
                      line=dict(color="rgb(242, 227, 19)", width=6), name='Slices')
vector2 = go.Scatter3d(x=x_vector2, y=y_vector2, z=z_vector2, marker=dict(size=1, color="rgb(84, 48, 5)"),
                      line=dict(color="rgb(84, 88, 111)", width=6), name='Amplitudes')
vector3 = go.Scatter3d(x=x_vector3, y=y_vector3, z=z_vector3, marker=dict(size=1, color="rgb(84, 48, 5)"),
                      line=dict(color="rgb(71, 242,19)", width=6), name='Latencies')
data = [points, vector1, vector2, vector3]

layout = go.Layout(xaxis=dict(title='Slices', titlefont=dict(family='Arial, sans-serif', size=18, color='black')),
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