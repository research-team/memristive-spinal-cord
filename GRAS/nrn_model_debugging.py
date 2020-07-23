import numpy as np
import physipy as p
import matplotlib.pyplot as plt

'''
Formulas and value units were taken from:

Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011). 
Principles of Computational Modelling in Neuroscience. Cambridge: Cambridge University Press. 
DOI:10.1017/CBO9780511975899
'''
Cm = 1              # uF / cm2 membrane capacity
E_Na = 50           # mV
E_K = -90           # mV
E_L = -72           # mV
g_Na = 120          # mS / cm2
g_K = 36            # mS / cm2
g_L = 0.3           # mS / cm2
Ra = 100            # Ohm cm
d = 6               # um - diameter
x = 2               # um - length of one compartment
dt = 0.025          # ms - sim step
V_adj = -63         # mV -
Vm = [-72] * 3      # mV - array for three compartments volatge
old_Vm = -72        #
n = [0] * 3         # compartments channel
m = [0] * 3         # compartments channel
h = [1] * 3         # compartments channel
g_inh, g_exc = 0, 0 # conductivity level
E_ex = 50           # reverse exc
E_in = -80          # reverse inh
tau_syn_exc = 0.2   #
tau_syn_inh = 2.0   #
ref_time_timer = 0  #
ref_time = int(3 / dt)  #

# cylindric neuron's area
S = np.pi * d * x

# lists for records
v_dat = []
g_dat = []
s_dat = []
N = []
H = []
M = []
INP, MID, OUT = 0, 1, 2

sim_time_steps = int(20 / dt)   # simulation time

# constants for saving performance
const1 = dt / Cm
const2 = d / (4 * Ra * x * x)

# SIMULATION
for t in range(sim_time_steps):
	# collect data
	v_dat.append(Vm[:])
	g_dat.append(g_exc)
	N.append(n[:])
	M.append(m[:])
	H.append(h[:])
	# add a stimulus
	if t == 100:
		g_exc = 5
	# save old value of Vm
	old_Vm = Vm[INP]
	# calculate synaptic currents
	I_syn_exc = g_exc * (Vm[INP] - E_ex)
	I_syn_inh = 0 # g_inh * (Vm[INP] - E_in)
	# update compartments
	for comp in range(3):
		# calc K/NA/L currents
		I_K = g_K * n[comp] ** 4 * (Vm[comp] - E_K)
		I_Na = g_Na * m[comp] ** 3 * h[comp] * (Vm[comp] - E_Na)
		I_L = g_L * (Vm[comp] - E_L)
		# input Vm comp
		if comp == 0:
			Vm[comp] += const1 * (const2 * (2 * Vm[MID] - 2 * Vm[INP]) - I_Na - I_K - I_L - I_syn_exc - I_syn_inh)
		# middle Vm comp
		elif comp == 1:
			Vm[comp] += const1 * (const2 * (Vm[OUT] - 2 * Vm[MID] + Vm[INP]) - I_Na - I_K - I_L)
		# output Vm comp
		else:
			Vm[comp] += const1 * (const2 * (2 * Vm[MID] - 2 * Vm[OUT]) - I_Na - I_K - I_L)
		# some staff for easier a/b calculations
		dV = Vm[comp] - V_adj
		# K act
		a = 0.032 * (15 - dV) / (np.e ** ((15 - dV) / 5) - 1)
		b = 0.5 * np.e ** ((10 - dV) / 40)
		n[comp] += (a - (a + b) * n[comp]) * dt
		# Na act
		a = 0.32 * (13 - dV) / (np.e ** ((13 - dV) / 4) - 1)
		b = 0.28 * (dV - 40) / (np.e ** ((dV - 40) / 5) - 1)
		m[comp] += (a - (a + b) * m[comp]) * dt
		# Na inact
		a = 0.128 * np.e ** ((17 - dV) / 18)
		b = 4 / (1 + np.e ** ((40 - dV) / 5))
		h[comp] += (a - (a + b) * h[comp]) * dt
	# recalc conductivity
	g_exc -= g_exc / tau_syn_exc * dt
	g_inh -= g_inh / tau_syn_inh * dt
	# check on spike
	if ref_time_timer == 0 and V_adj + 30 <= Vm[INP] < old_Vm:
		ref_time_timer = ref_time
		s_dat.append(t)
	# update refractory period timer
	if ref_time_timer > 0:
		ref_time_timer -= 1

# plot results
a = plt.subplot(311)
# plot Voltages
for v in np.array(v_dat).T:
	plt.plot(np.arange(sim_time_steps) * dt, v)
# plot spikes
plt.plot(np.array(s_dat) * dt, [0] * len(s_dat), '.')
plt.ylabel("Voltage, mV")
# plot channels
plt.subplot(312, sharex=a)
for c in np.array(N).T:
	plt.plot(np.arange(sim_time_steps) * dt, c)
for c in np.array(H).T:
	plt.plot(np.arange(sim_time_steps) * dt, c)
for c in np.array(M).T:
	plt.plot(np.arange(sim_time_steps) * dt, c)
plt.ylabel("Channels")
# plot g exc and g inh
plt.subplot(313, sharex=a)
plt.plot(np.arange(sim_time_steps) * dt, g_dat)
plt.xlabel("Time, ms")
plt.ylabel("Conductivity, mS ?")
plt.show()


'''
UNITS CHECKING
'''
Ohm = p.units["V"] / p.units["A"]
mV = p.units["V"] / 1000
mA = p.units["mA"]
ms = p.units["ms"]
uF = p.units["F"] / 10 ** 6
s = p.units["s"]
Cm = p.units["cm"]
um = p.units["cm"] / 1000
cm2 = Cm ** 2
um2 = um ** 2
uA = mA / 1000

print(ms * um * mV / (uF / cm2 * Ohm * Cm * um2))
print(ms * mA / (uF / cm2 * cm2))
