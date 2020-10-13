import numpy as np
import matplotlib.pyplot as plt

'''
Formulas and value units were taken from:

Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011). 
Principles of Computational Modelling in Neuroscience. Cambridge: Cambridge University Press. 
DOI:10.1017/CBO9780511975899
'''
dt = 0.025  # [ms] - sim step
E_Na = 55   # [mV] - reversal potential
E_K = -90   # [mV] - reversal potential
E_L = -72   # [mV] - reversal potential
E_ex = 50   # [mV] - reversal potential
E_in = -100  # [mV] reverse inh

test_inter = False

if test_inter:
	Cm = 1  # [uF/cm2] membrane capacity

	# inter
	g_Na = 120  # [S / cm2]
	g_K = 36   # [S / cm2]
	g_L = 0.3   # [S / cm2]
	# inter
	Ra = 100    # [Ohm cm]
	d = 6
	x = 2
else:
	Cm = 2  # [uF/cm2] membrane capacity
	gnabar = 0.05  # [S / cm2]
	gkrect = 0.3   # [S / cm2]
	g_L = 0.002   # [S / cm2]
	# inter
	Ra = 200    # [Ohm cm]
	d = 50
	x = 16
	gcaN = 0.05
	gcaL = 0.0001
	gcak = 0.3
	ca0 = 2
	amA = 0.4
	amB = 66
	amC = 5
	bmA = 0.4
	bmB = 32
	bmC = 5
	R = 8.314472
	F = 96485.34

V_adj = -63  # [mV] - adjust voltage for -55 threshold
tau_syn_exc = 0.3  # [ms]
tau_syn_inh = 2.0  # [ms]
ref_time = int(3 / dt)  # [steps]
Re = 333  # * CENTI     # convert [Ohm cm] to [Ohm um] Resistance of extracellular space

nrns = 50
segs = 3
sim_time_steps = int(25 / dt)  # simulation time

Vm = np.full(fill_value=-72, shape=(nrns, segs), dtype=float)  # [mV] - array for three compartments volatge
n = np.zeros(shape=(nrns, segs), dtype=float)  # [0..1] compartments channel
m = np.zeros(shape=(nrns, segs), dtype=float)  # [0..1] compartments channel
h = np.ones(shape=(nrns, segs), dtype=float)  # [0..1] compartments channel
cai = np.ones(shape=(nrns, segs), dtype=float) * 0.0001  # [0..1] compartments channel
hc = np.ones(shape=(nrns, segs), dtype=float)  # [0..1] compartments channel
mc = np.ones(shape=(nrns, segs), dtype=float)  # [0..1] compartments channel
p = np.ones(shape=(nrns, segs), dtype=float)  # [0..1] compartments channel

I_K = np.zeros(shape=(nrns, segs), dtype=float)  # [nA] ionic currents
I_Na = np.zeros(shape=(nrns, segs), dtype=float)  # [nA] ionic currents
I_L = np.zeros(shape=(nrns, segs), dtype=float)  # [nA] ionic currents
Eca = np.zeros(shape=(nrns, segs), dtype=float)  # [nA] ionic currents
I_ica = np.zeros(shape=(nrns, segs), dtype=float)  # [nA] ionic currents
g_inh = np.zeros(shape=nrns, dtype=float)
g_exc = np.zeros(shape=nrns, dtype=float)  # [mS] conductivity level

ref_time_timer = np.zeros(shape=nrns, dtype=float)  # [steps]
V_extra = np.zeros(shape=(sim_time_steps, nrns), dtype=float)  # [mV] extracellular potential
old_Vm = np.full(fill_value=-72, shape=nrns, dtype=float)  # [mV] = old value of Vm

INP, MID, OUT = 0, 1, 2

save_Vm = []
save_n = []
save_m = []
save_h = []
save_I_K = []
save_I_Na = []
save_I_L = []
save_g_inh = []
save_g_exc = []
spikes = []
const3 = (np.log(np.sqrt(x ** 2 + d ** 2) + x) - np.log(np.sqrt(x ** 2 + d ** 2) - x)) / (4 * np.pi * x * Re)

rand_spike_time = np.random.randint(0, sim_time_steps-1, 5)

def Exp(_lx):
	# _lExp
	if _lx < - 100.0:
		return 0
	return np.exp(_lx)

def alpham(volt):
	if abs((volt + amB) / amC) < 1e-6:
		return amA * amC
	return (amA * (x + amB)) / (1 - Exp(-(volt + amB) / amC))

def betam(volt):
	if abs((volt + bmB) / bmC) < 1e-6:
		return -bmA * bmC
	return (bmA * (-(x + bmB))) / (1 - Exp((volt + bmB) / bmC))

for t in range(sim_time_steps):
	save_Vm.append(Vm[:, 0].copy())
	print(Eca)

	for nrn_id in range(nrns):
		# add a stimulus
		# if t in rand_spike_time:
		# 	rands_spike = np.random.randint(0, nrns-1, 25)
		# 	# g_exc[rands_spike] = np.random.uniform(0.5, 1, len(rands_spike))
		# 	g_exc[rands_spike] = np.random.uniform(5, 10, len(rands_spike))
		# if t * dt in [5, 15]:
		# 	rands_spike = np.random.randint(0, nrns - 1, 25)
		# 	g_exc[rands_spike] = 0.0001
		# save old value of Vm
		old_Vm[nrn_id] = Vm[nrn_id, INP]
		# calculate synaptic currents
		I_syn_exc = g_exc[nrn_id] * (E_ex - Vm[nrn_id, INP])
		I_syn_inh = g_inh[nrn_id] * (E_in - Vm[nrn_id, INP])
		# update compartments
		for segment in range(segs):
			if test_inter:
				# Vm[nrn_id, segment] += np.random.uniform(-0.5, 0.5)
				# calc K/NA/L currents
				I_Na[nrn_id, segment] = g_Na * m[nrn_id, segment] ** 3 * h[nrn_id, segment] * (E_Na - Vm[nrn_id, segment])
				I_K[nrn_id, segment] = g_K * n[nrn_id, segment] ** 4 * (E_K - Vm[nrn_id, segment])
				I_L[nrn_id, segment] = g_L * (E_L - Vm[nrn_id, segment])
				# first segment
				if segment == 0:
					vv1 = 0
					vv2 = Vm[nrn_id, segment + 1] - Vm[nrn_id, segment]
				# the last segment
				elif segment == segs - 1:
					vv1 = Vm[nrn_id, segment - 1] - Vm[nrn_id, segment]
					vv2 = 0
				else:
					vv1 = Vm[nrn_id, segment - 1] - Vm[nrn_id, segment]
					vv2 = Vm[nrn_id, segment + 1] - Vm[nrn_id, segment]
				#
				I_leak = g_L * (E_L - Vm[nrn_id, segment])
				I_ionic = (I_K[nrn_id, segment] + I_Na[nrn_id, segment]) / (np.pi * x * d)
				I_axonal = d / (4 * x * x * Ra) * (vv1 + vv2) * 10000
				I_inj = (I_syn_exc + I_syn_inh) / (np.pi * x * d) * 10000
				#
				if segment == INP:
					if ref_time_timer[nrn_id] > 0:
						I_inj = 0
					Vm[nrn_id, segment] += (dt / Cm) * (I_leak + I_ionic + I_axonal + I_inj)
				else:
					dV = (dt / Cm) * (I_leak + I_ionic + I_axonal)
					Vm[nrn_id, segment] += dV
					# if segment == MID:
					# 	V_extra[nrn_id] = -const3 * (I_leak + I_ionic * 100 + Cm / dt * dV)
				dV = Vm[nrn_id, segment] - V_adj
				# K act
				a = 0.032 * (15 - dV) / (np.exp((15 - dV) / 5) - 1)
				b = 0.5 * np.exp((10 - dV) / 40)
				n[nrn_id, segment] += (1 - np.exp(-dt * (a + b))) * (a / (a + b) - n[nrn_id, segment])
				# Na act
				a = 0.32 * (13 - dV) / (np.exp((13 - dV) / 4) - 1)
				b = 0.28 * (dV - 40) / (np.exp((dV - 40) / 5) - 1)
				m[nrn_id, segment] += (1 - np.exp(-dt * (a + b))) * (a / (a + b) - m[nrn_id, segment])
				# Na inact
				a = 0.128 * np.exp((17 - dV) / 18)
				b = 4 / (1 + np.exp((40 - dV) / 5))
				h[nrn_id, segment] += (1 - np.exp(-dt * (a + b))) * (a / (a + b) - h[nrn_id, segment])
			else:
				# calc K/NA/L currents
				I_Na[nrn_id, segment] = gnabar * m[nrn_id, segment] ** 3 * h[nrn_id, segment] * (E_Na - Vm[nrn_id, segment])
				I_K[nrn_id, segment] = gkrect * n[nrn_id, segment] ** 4 * (E_K - Vm[nrn_id, segment]) + \
				                       gcak * cai[nrn_id, segment] ** 2 / (cai[nrn_id, segment] ** 2 + 0.014 ** 2) * (E_K - Vm[nrn_id, segment])
				I_L[nrn_id, segment] = g_L * (E_L - Vm[nrn_id, segment])
				# Ca channel
				Eca[nrn_id, segment] = (1000 * R * 309.15 / (2 * F)) * np.log(ca0 / cai[nrn_id, segment])
				I_ica[nrn_id, segment] = gcaN * mc[nrn_id, segment] ** 2 * hc[nrn_id, segment] * (Eca[nrn_id, segment] - Vm[nrn_id, segment]) + \
				                         gcaL * p[nrn_id, segment] * (Eca[nrn_id, segment] - Vm[nrn_id, segment])
				# print(f"Vm {Vm[nrn_id, segment]}")
				# print(f"Na {I_Na[nrn_id, segment]} | K {I_K[nrn_id, segment]} L {I_L[nrn_id, segment]} | Ica {I_ica[nrn_id, segment]}")
				# print(f"m {m[nrn_id, segment]} | n {n[nrn_id, segment]} | h {h[nrn_id, segment]}")
				# print(f"hc {hc[nrn_id, segment]} | mc {mc[nrn_id, segment]} | p {p[nrn_id, segment]} cai {cai[nrn_id, segment]}")
				# print("- " * 10)

				# first segment
				if segment == 0:
					vv1 = 0
					vv2 = Vm[nrn_id, segment + 1] - Vm[nrn_id, segment]
				# the last segment
				elif segment == segs - 1:
					vv1 = Vm[nrn_id, segment - 1] - Vm[nrn_id, segment]
					vv2 = 0
				else:
					vv1 = Vm[nrn_id, segment - 1] - Vm[nrn_id, segment]
					vv2 = Vm[nrn_id, segment + 1] - Vm[nrn_id, segment]
				#
				I_leak = g_L * (E_L - Vm[nrn_id, segment])
				I_ionic = (I_K[nrn_id, segment] + I_Na[nrn_id, segment] + I_ica[nrn_id, segment]) / (np.pi * x * d)
				I_axonal = d / (4 * x * x * Ra) * (vv1 + vv2) * 10000
				I_inj = (I_syn_exc + I_syn_inh) / (np.pi * x * d) * 10000

				if segment == INP:
					if ref_time_timer[nrn_id] > 0:
						I_inj = 0
					Vm[nrn_id, segment] += (dt / Cm) * (I_leak + I_ionic + I_axonal + I_inj)
				else:
					dV = (dt / Cm) * (I_leak + I_ionic + I_axonal)
					Vm[nrn_id, segment] += dV
					# if segment == MID:
					# 	V_extra[nrn_id] = -const3 * (I_leak + I_ionic * 100 + Cm / dt * dV)

				# FAST SODIUM
				V = Vm[nrn_id, segment]
				a = alpham(V)
				b = betam(V)
				# m
				tau_m = 1 / (a + b)
				m_inf = a / (a + b)
				# h
				tau_h = 30 / (Exp((V + 60) / 15) + Exp(-(V + 60) / 16))
				h_inf = 1 / (1 + Exp((V + 65) / 7))

				# DELAYED RECTIFIER POTASSIUM
				tau_n = 5 / (Exp((V + 50) / 40) + Exp(-(V + 50) / 50))
				n_inf = 1 / (1 + Exp(-(V + 38) / 15))

				# CALCIUM DYNAMICS N-type
				tau_mc = 15
				mc_inf = 1 / (1 + Exp(-(V + 32) / 5))
				tau_hc = 50
				hc_inf = 1 / (1 + Exp((V + 50) / 5))

				# CALCIUM DYNAMICS L-type
				tau_p = 400
				p_inf = 1 / (1 + Exp(-(V + 55.8) / 3.7))

				m[nrn_id, segment] += (1. - np.exp(dt * ((((- 1.0))) / tau_m))) * (- (((m_inf)) / tau_m) / ((((- 1.0))) / tau_m) - m[nrn_id, segment])
				h[nrn_id, segment] += (1. - np.exp(dt * ((((- 1.0))) / tau_h))) * (- (((h_inf)) / tau_h) / ((((- 1.0))) / tau_h) - h[nrn_id, segment])
				p[nrn_id, segment] += (1. - np.exp(dt * ((((- 1.0))) / tau_p))) * (- (((p_inf)) / tau_p) / ((((- 1.0))) / tau_p) - p[nrn_id, segment])
				n[nrn_id, segment] += (1. - np.exp(dt * ((((- 1.0))) / tau_n))) * (- (((n_inf)) / tau_n) / ((((- 1.0))) / tau_n) - n[nrn_id, segment])
				mc[nrn_id, segment] += (1. - np.exp(dt * ((((- 1.0))) / tau_mc))) * (- (((mc_inf)) / tau_mc) / ((((- 1.0))) / tau_mc) - mc[nrn_id, segment])
				hc[nrn_id, segment] += (1. - np.exp(dt * ((((- 1.0))) / tau_hc))) * (- (((hc_inf)) / tau_hc) / ((((- 1.0))) / tau_hc) - hc[nrn_id, segment])
				cai[nrn_id, segment] += (1. - np.exp(dt * ((0.01) * (((- (4.0) * (1.0))))))) * (- ((0.01) * ((- (I_ica[nrn_id, segment])))) / ((0.01) * (((- (4.0) * (1.0))))) - I_ica[nrn_id, segment])
		# recalc conductivity
		g_exc[nrn_id] -= g_exc[nrn_id] / tau_syn_exc * dt
		g_inh[nrn_id] -= g_inh[nrn_id] / tau_syn_inh * dt
		# check on spike
		if ref_time_timer[nrn_id] == 0 and V_adj + 30 <= Vm[nrn_id, INP] < old_Vm[nrn_id]:
			ref_time_timer[nrn_id] = ref_time
			spikes.append(t * dt)
		# update refractory period timer
		if ref_time_timer[nrn_id] > 0:
			ref_time_timer[nrn_id] -= 1

# plot results
# a = plt.subplot(111)
# plot Voltages

# for seg in range(segs):
# voltage = Vm[:, 10]
# plt.plot(np.arange(len(voltage)) * dt, voltage, label='volt')
# plt.show()
# plt.close()
plt.close()
voltage = np.array(save_Vm).T
print(voltage.shape)
print(sim_time_steps)
xticks = np.arange(sim_time_steps) * dt

for neuron_volt in voltage:
	plt.plot(xticks, neuron_volt, alpha=0.1)
mean_volt = np.mean(voltage, axis=0)
plt.plot(xticks, mean_volt, color='k', ls='--')

plt.plot(spikes, [0] * len(spikes), '.', color='r')

#
# for v, label in zip(np.array(Vm).T, ['input', 'middle', 'output']):
# 	plt.plot(np.arange(sim_time_steps) * dt, v, label=label)
# plt.legend()
# # plot spikes
# plt.plot(np.array(s_dat) * dt, [0] * len(s_dat), '.')
# plt.ylabel("Voltage, mV")
# # plot channels
# plt.subplot(512, sharex=a)
# xticks = np.arange(sim_time_steps) * dt
# for c, name, line in zip(N.T, ['in (K)', 'mid (K)', 'out (K)'], ['-', '--', '-.']):
# 	plt.plot(xticks, c, label=name, ls=line)
# for c, name, line in zip(H.T, ['in (Na)', 'mid (Na)', 'out (Na)'], ['-', '--', '-.']):
# 	plt.plot(xticks, c, label=name, ls=line)
# for c, name, line in zip(M.T, ['in (Na inact)', 'mid (Na inact)', 'out (Na inact)'], ['-', '--', '-.']):
# 	plt.plot(xticks, c, label=name, ls=line)
# # plt.legend()
# plt.ylabel("Channels")
# # plot g exc and g inh
# plt.subplot(513, sharex=a)
# for g, label in zip(np.array(curr_dat).T, ['K', 'Na']):
# 	plt.plot(np.arange(sim_time_steps) * dt, g, label=label)
# plt.ylabel("Currents, mA")
# plt.legend()
#
# plt.subplot(514, sharex=a)
# for g, label, color in zip(np.array(g_dat).T, ['g_exc', 'g_inh'], ['r', 'b']):
# 	plt.plot(np.arange(sim_time_steps) * dt, g, color=color, label=label)
# plt.ylabel("Conductivity (synaptic), mS")
# plt.legend()
#
# plt.subplot(515, sharex=a)
# plt.plot(np.arange(sim_time_steps) * dt, ve_dat, label='extracellular')
# plt.legend()
plt.xlabel("Time, ms")
plt.tight_layout()
plt.show()
