import pylab as plt

step = 0.25
filepath = "/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU/test.dat"
neurons_volt = []
neurons_curr = []

with open(filepath) as file:
	for line in file:
		gid, t, v, i = line.split()
		neurons_volt.append(float(v))
		neurons_curr.append(float(i))

ax1 = plt.subplot(211)
plt.plot([x * step for x in range(len(neurons_volt))], neurons_volt, color='b', label='volt')
for i in range(6):
	plt.axvline(x=i * 20 * step, color='grey', linestyle='--')
plt.legend()



ax2 = plt.subplot(212, sharex=ax1)
plt.plot([x * step for x in range(len(neurons_curr))], neurons_curr, color='r', label='curr')
for i in range(6):
	plt.axvline(x=i * 20 * step, color='grey', linestyle='--')
plt.legend()

plt.show()
