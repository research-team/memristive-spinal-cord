import numpy as np
import pylab


path = '/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/ADC.dat'
upper_left = 'upper left'

with open(path) as file:
	adc = list(map(int, file.readline().split()))
	dac = list(map(int, file.readline().split()))
	norm = list(map(float, file.readline().split()))
	volt = list(map(float, file.readline().split()))

low = -80
peak = 35

a = volt
b = list(map(lambda x: x - low, a))
c = list(map(lambda x: x / 65 * 5, b))
volt = b
for d in zip(a, b, c):
	for o in d:
		print(f"{o:8.2f}", end='')
	print()

print(f"max V {max(a):.2f}, mean {np.mean(a):.2f}, min {min(a):.2f}")
print(f"max {max(c):.2f}, mean {np.mean(c):.2f}, min {min(c):.2f}")

k = 127 / 2047

for a in adc:
	print(a, k * a)

adc = list(map(lambda x: x * k, adc))

# Yi = [Xi - min(X)]/[max(X) - min(X)]

volt = list(map(lambda x: (x + 80), volt))

sim_time = int(len(volt) * 0.25)
shared_x = [v * 0.25 for v in range(len(volt))]

a = pylab.subplot(311)
pylab.plot(shared_x, volt, label="neuron")
pylab.plot(shared_x, adc, label="ADC")
pylab.plot(shared_x, norm, label="normalized")
pylab.xlim(0, sim_time)
pylab.ylim(0, 128)
pylab.grid(axis='x')
pylab.xticks(range(0, sim_time+1, 25))
pylab.legend(loc=upper_left)

pylab.subplot(312, sharex=a)
pylab.plot(shared_x, volt, label="neuron")
pylab.plot(shared_x, dac, label="DAC")
pylab.xlim(0, sim_time)
pylab.grid(axis='x')
pylab.xticks(range(0, sim_time+1, 25))
pylab.legend(loc=upper_left)

pylab.subplot(313, sharex=a)
pylab.plot(shared_x, adc, label="ADC")
pylab.plot(shared_x, dac, label="DAC")
pylab.xlim(0, sim_time)
pylab.xticks(range(0, sim_time+1, 25))
pylab.grid(axis='x')
pylab.legend(loc=upper_left)


pylab.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.97, hspace=0.1)
pylab.xlabel("time, ms")
pylab.xlim(0, sim_time)
pylab.show()
