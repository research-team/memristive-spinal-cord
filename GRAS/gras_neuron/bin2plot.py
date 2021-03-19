import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

dx = 0.025
neurons = 5000
path = "gras_neuron/file.bin"


intracellular = np.fromfile(path, dtype=np.double).reshape(-1, neurons)
extracellular = np.diff(intracellular, axis=0) / dx
extracellular *= 1 / (4 * np.pi * 10000)
# extracellular = gaussian_filter(extracellular, sigma=1.25)

plt.subplot(221)
plt.title("intracellular")
average = np.mean(intracellular, axis=1)
time = np.arange(len(average)) * dx
plt.plot(time, average)

plt.subplot(223)
plt.title("extracellular")
average = np.mean(extracellular, axis=1)
time = np.arange(len(average)) * dx
plt.plot(time, average)

plt.subplot(122)
plt.title("slices")
average = np.mean(extracellular, axis=1)

slices = len(average) // 1000
slice_len = 1000

yticks = []
for s_index in range(slices):
	sdata = -average[s_index * slice_len:s_index * slice_len + 1000]
	time = np.arange(len(sdata)) * dx
	y = sdata + s_index / 1000
	yticks.append(y[0])
	plt.plot(time, y)

plt.yticks(yticks, np.arange(slices) + 1)
plt.show()

