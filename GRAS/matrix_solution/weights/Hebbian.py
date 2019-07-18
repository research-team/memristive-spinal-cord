import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

with open('dT.dat') as file:
	dT = np.array(list(map(float, file.read().split())))

with open('dW.dat') as file:
	dW = np.array(list(map(float, file.read().split())))

fig = plt.figure()

plt.title('Hebbian')
plt.xlabel("Δt (ms)")
plt.ylabel("ΔW (%)")
plt.xlim(-5, 5)
plt.ylim(-10, 10)
plt.axvline(0, linestyle="-", color="black")
plt.axhline(0, linestyle="-", color="black")

plt.scatter(dT[dW != 0], dW[dW != 0])

plt.savefig('Hebbian.png')


