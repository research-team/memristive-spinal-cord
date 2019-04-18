import os
import pylab as plt
from copy import deepcopy

step = 0.025

global_volt = []
global_h = []
global_m = []
global_n = []
global_g_exc = []
global_g_inh = []
global_spikes = []

abs_path = "/home/alex/GitHub/memristive-spinal-cord/GPU/matrix_solution/dat/"

for filename in [f for f in os.listdir(abs_path) if f.endswith(".dat")]:
	with open(abs_path + filename) as file:
		plt.figure()
		plt.suptitle(filename)

		voltages = list(map(float, file.readline().split()))
		g_exc = list(map(float, file.readline().split()))
		g_inh = list(map(float, file.readline().split()))
		spikes = list(map(float, file.readline().split()))

		shared_x = [x * step for x in range(len(voltages))]

		# volts and spikes
		volts_subplot = plt.subplot(311)
		volts_subplot.plot(shared_x, voltages, color='k')
		volts_subplot.plot(spikes, [0] * len(spikes), '.', color='r')
		volts_subplot.set_ylabel("Membrane potential, mV")
		volts_subplot.legend()

		# activ_subplot = plt.subplot(312, sharex=volts_subplot)
		# activ_subplot.plot(shared_x, h, label='h [Na+] act', color='g')
		# activ_subplot.plot(shared_x, m, label='m [Na+] inact', color='b')
		# activ_subplot.plot(shared_x, n, label='n [K+] act', color='k')
		# activ_subplot.set_ylabel("Activation variables, x(t)")
		# activ_subplot.legend()

		conduct_subplot = plt.subplot(313, sharex=volts_subplot)
		conduct_subplot.plot(shared_x, g_exc, label='g_exc', color='r')
		conduct_subplot.plot(shared_x, g_inh, label='g_inh', color='b')
		conduct_subplot.set_ylabel("synaptic conductance, nS")
		conduct_subplot.set_xlabel("time (ms)")
		conduct_subplot.legend()

		plt.xlim(0, shared_x[-1])
		plt.show()



