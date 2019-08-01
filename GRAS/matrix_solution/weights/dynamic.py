import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

with open('slice_and_weights.dat') as file:
	weights = np.array(list(map(float, file.read().split())))

slice_number = 33
synapses_number = 663666
medium_weights = []
n = 0

for j in range(slice_number):
    medium_weight = 0
    for i in range(synapses_number):
        medium_weight += weights[i + n]
    medium_weights.append(medium_weight / synapses_number)
    n += synapses_number

x = []

for i in range(slice_number):
    x.append(i)

print(medium_weights)

fig, ax = plt.subplots(figsize=(16, 9))
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

plt.scatter(x, medium_weights)
plt.savefig("dynamic.png")