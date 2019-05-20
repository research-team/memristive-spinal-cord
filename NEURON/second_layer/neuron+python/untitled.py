from neuron import h, gui
h.load_file('nrngui.hoc')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import math
import numpy as np

from motoneuron import motoneuron
from bioaff import bioaff
import random


ncs = []
stims = []
afferents = []
motos = []

for i in range(12):
	cell = bioaff(random.randint(2, 10))
	afferents.append(cell)

for i in range(10):
	stim = h.SpikeGenerator(0.5)
	stim.start = 10
	stim.number = 1000000
	stim.speed = 300
	stim.k = 3/stim.speed
	stims.append(stim)
#stim.interval = 10
#for j in range(1):
#	for i in range(1):
for cell in afferents:
	for i in range(10):
		ncstim = h.NetCon(stims[0], cell.synlistex[i])
		ncstim.delay = random.gauss(1, 0.1)
		ncstim.weight[0] = random.gauss(1, 0.1)
		ncs.append(ncstim)

def set_recording_vectors(pool):
	v_vec = []
	for cell in pool:
		vec = h.Vector()
		vec.record(cell.soma(0.5)._ref_v)
		v_vec.append(vec)
	t_vec = h.Vector() # Time stamp vector
	return v_vec, t_vec

def simulate(tstop=170, vinit=-65):
    h.tstop = tstop
    h.v_init = vinit
    h.run()

def avgarr(z):
	summa = 0
	for item in z:
		summa += np.array(item)
	return summa

def show_output(v_vec, t_vec):
	outavg = []
	for j in v_vec:
		outavg.append(j)
	outavg = avgarr(outavg)
	dend_plot = pyplot.plot(outavg)
	pyplot.xlabel('time (ms)')
	pyplot.ylabel('mV')

if __name__ == '__main__':
    branch_vec, t_vec = set_recording_vectors(afferents)
    simulate()
    show_output(branch_vec, t_vec)
    pyplot.show()