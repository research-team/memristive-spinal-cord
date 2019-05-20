import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from analysis.functions import read_neuron_data
from matplotlib import pyplot as plt
from analysis.histogram_lat_amp import sim_process
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib

sim_step = 0.025
neuron_list = read_neuron_data('../../neuron-data/15EX_serotonin.hdf5')
neuron_means = list(map(lambda voltages: np.mean(voltages), zip(*neuron_list)))[:12000]

# calculating latencies and amplitudes of mean values
neuron_means_lat = sim_process(neuron_means, sim_step, inhibition_zero=False)[0]
neuron_means_amp = sim_process(neuron_means, sim_step, inhibition_zero=False)[1]
slices = []
for i in range(len(neuron_means_lat)):
	slices.append(i + 1)
print(slices)

neuron_dict = {}
neuron_dict["slices"] = slices
neuron_dict["latencies"] = neuron_means_lat
neuron_dict["amplitudes"] = neuron_means_amp
print(neuron_dict)

df = pd.DataFrame(neuron_dict)
print(df)

my_dpi = 96
plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)

pca = PCA(n_components=3)
# a = np.array(neuron_means_amp)
# print("a = ", a)
# a.reshape(1, -1)
# print("a = ", a)

pca.fit(df)

# p = pca.components_
# centroid = np.mean(a, 0)
# segments = np.arange(-40, 40)[:, np.newaxis] * p

# matplotlib.use('TkAgg')
# plt.ion()

result = pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(3)], index=df.index)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# scatterplot = ax.scatter(*(a.T))
# lineplot = ax.plot(*(centroid + segments).T, color="red")
# plt.xlabel('x')
# plt.ylabel('y')
ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap="Set2_r", s=60)

xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0, 0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0, 0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0, 0), (min(result['PCA2']), max(result['PCA2'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

ax.set_xlabel("Slices")
ax.set_ylabel("Latencies")
ax.set_zlabel("Amplitudes")
ax.set_title("PCA on slices - latencies - amplitudes (NEURON)")
plt.show()