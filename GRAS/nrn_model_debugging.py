import numpy as np
import matplotlib.pyplot as plt

'''
Formulas and value units were taken from:

Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011). 
Principles of Computational Modelling in Neuroscience. Cambridge: Cambridge University Press. 
DOI:10.1017/CBO9780511975899
'''
muscle = True

MICRO = 10 ** -6
CENTI = 10 ** -1
uF_m2 = 10 ** 4         # 1 microfarad per square centimeter = 10 000 microfarad per square meter
mS_m2 = 10 ** 4         # 1 millisiemens per square centimeter = 10 000 millisiemens per square meter
Cm = 1 * uF_m2          # convert [uF / cm2] to [uF / m2] membrane capacity
E_Na = 50               # [mV] - reversal potential
E_K = -90               # [mV] - reversal potential
E_L = -72               # [mV] - reversal potential
x = 2 * MICRO

if muscle:
	g_Na = 10 * mS_m2   # convert [mS / cm2] to [mS / m2]
	g_K = 1 * mS_m2     # convert [mS / cm2] to [mS / m2]
	g_L = 0.3 * mS_m2   # convert [mS / cm2] to [mS / m2]
	Ra = 1000 * CENTI   # convert [Ohm cm] to [Ohm m]
	d = 5 * MICRO       # convert [um] to [m] - diameter
	E_ex = 0            # [mV] reverse exc
else:
	g_Na = 120 * mS_m2
	g_K = 36 * mS_m2
	g_L = 0.3 * mS_m2
	Ra = 100 * CENTI
	d = 6 * MICRO
	E_ex = 50

dt = 0.025              # [ms] - sim step
V_adj = -63             # [mV] - adjust voltage for -55 threshold
Vm = [-72] * 3          # [mV] - array for three compartments volatge
old_Vm = -72            # [mV] = old value of Vm
n = [0] * 3             # [0..1] compartments channel
m = [0] * 3             # [0..1] compartments channel
h = [1] * 3             # [0..1] compartments channel
I_K = [0] * 3           # [nA] ionic currents
I_Na = [0] * 3          # [nA] ionic currents
I_L = [0] * 3           # [nA] ionic currents
g_inh, g_exc = 0, 0     # [mS] conductivity level
E_in = -80              # [mV] reverse inh
tau_syn_exc = 0.3       # [ms]
tau_syn_inh = 2.0       # [ms]
ref_time_timer = 0      # [steps]
ref_time = int(3 / dt)  # [steps]
V_extra = 0             # [mV] extracellular potential
Re = 333 * 10 ** -1     # convert [Ohm cm] to [Ohm um] Resistance of extracellular space

sim_time_steps = int(25 / dt)   # simulation time

# lists for records
ve_dat = []
g_dat = []
s_dat = []
v_dat = np.zeros(shape=(sim_time_steps, 3))
N = np.zeros(shape=(sim_time_steps, 3))
M = np.zeros(shape=(sim_time_steps, 3))
H = np.zeros(shape=(sim_time_steps, 3))

INP, MID, OUT = 0, 1, 2

# constants for saving performance
const1 = dt / Cm
const2 = d / (4 * Ra * x * x)
# back to um for extracelullar calculation
x /= MICRO
d /= MICRO
const3 = (np.log(np.sqrt(x ** 2 + d ** 2) + x) - np.log(np.sqrt(x ** 2 + d ** 2) - x)) / (4 * np.pi * x * Re)
# SIMULATION
for t in range(sim_time_steps):
	# collect data
	ve_dat.append(V_extra)
	g_dat.append((g_K * n[0] ** 4,
	              g_Na * m[0] ** 3 * h[0]))
	v_dat[t] = Vm[:]
	N[t] = n[:]
	M[t] = m[:]
	H[t] = h[:]
	# add a stimulus
	if t == 100:
		g_exc = 60000
	# save old value of Vm
	old_Vm = Vm[INP]
	# calculate synaptic currents
	I_syn_exc = g_exc * (Vm[INP] - E_ex)
	I_syn_inh = g_inh * (Vm[INP] - E_in)
	# update compartments
	for comp in range(3):
		# calc K/NA/L currents
		I_K[comp] = g_K * n[comp] ** 4 * (Vm[comp] - E_K)
		I_Na[comp] = g_Na * m[comp] ** 3 * h[comp] * (Vm[comp] - E_Na)
		I_L[comp] = g_L * (Vm[comp] - E_L)
		# input Vm comp
		if comp == 0:
			Vm[comp] += const1 * (const2 * (2 * Vm[MID] - 2 * Vm[INP]) - I_Na[comp] - I_K[comp] - I_L[comp] - I_syn_exc - I_syn_inh)
		# middle Vm comp
		elif comp == 1:
			dv = const1 * (const2 * (Vm[OUT] - 2 * Vm[MID] + Vm[INP]) - I_Na[comp] - I_K[comp] - I_L[comp])
			Vm[comp] += dv
			ik = g_K / mS_m2 * n[comp] ** 4 * (Vm[comp] - E_K)
			ina = g_Na / mS_m2 * m[comp] ** 3 * h[comp] * (Vm[comp] - E_Na)
			il = g_L / mS_m2 * (Vm[comp] - E_L)
			V_extra = -const3 * (ik + ina + il + (Cm / uF_m2) / dt * dv)
		# output Vm comp
		else:
			Vm[comp] += const1 * (const2 * (2 * Vm[MID] - 2 * Vm[OUT]) - I_Na[comp] - I_K[comp] - I_L[comp])
		# some staff for easier a/b calculations
		dV = Vm[comp] - V_adj
		# K act
		a = (10 - dV) / (100 * (np.exp((10 - dV) / 10) - 1))
		b = 0.125 * np.exp(-dV / 80)
		n[comp] += (a - (a + b) * n[comp]) * dt
		# Na act
		a = (25 - dV) / (10 * (np.exp((25 - dV) / 10) - 1))
		b = 4 * np.exp(-dV / 18)
		m[comp] += (a - (a + b) * m[comp]) * dt
		# Na inact
		a = 0.07 * np.exp(-dV / 20)
		b = 1 / (np.exp((30 - dV) / 10) + 1)
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
a = plt.subplot(411)
# plot Voltages
for v, label in zip(np.array(v_dat).T, ['input', 'middle', 'output']):
	plt.plot(np.arange(sim_time_steps) * dt, v, label=label)
plt.legend()
# plot spikes
plt.plot(np.array(s_dat) * dt, [0] * len(s_dat), '.')
plt.ylabel("Voltage, mV")
# plot channels
plt.subplot(412, sharex=a)
for c in N.T:
	plt.plot(np.arange(sim_time_steps) * dt, c)
for c in H.T:
	plt.plot(np.arange(sim_time_steps) * dt, c)
for c in M.T:
	plt.plot(np.arange(sim_time_steps) * dt, c)
plt.ylabel("Channels")
# plot g exc and g inh
plt.subplot(413, sharex=a)
for g, label in zip(np.array(g_dat).T, ['K', 'Na']):
	plt.plot(np.arange(sim_time_steps) * dt, g, label=label)
plt.ylabel("Conductivity, mS")
plt.legend()

plt.subplot(414, sharex=a)
plt.plot(np.arange(sim_time_steps) * dt, ve_dat, label='extracellular')
plt.legend()
plt.xlabel("Time, ms")
plt.show()
