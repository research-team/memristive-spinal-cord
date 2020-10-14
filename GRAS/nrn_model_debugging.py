import numpy as np
import matplotlib.pyplot as plt
import logging as log
log.basicConfig(level=log.INFO)
'''
Formulas and value units were taken from:

Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011). 
Principles of Computational Modelling in Neuroscience. Cambridge: Cambridge University Press. 
DOI:10.1017/CBO9780511975899
'''
test_inter = False
debug = False
dt = 0.025              # [ms] - sim step
nrns = 1
segs = 3
sim_time = 150
stimulus = list(range(10, sim_time, 25))

E_Na = 55               # [mV] - reversal potential
E_K = -90               # [mV] - reversal potential
E_L = -72               # [mV] - reversal potential
E_ex = 50               # [mV] - reversal potential
E_in = -100             # [mV] reverse inh
V_adj = -63  # [mV] - adjust voltage for -55 threshold
tau_syn_exc = 0.3       # [ms]
tau_syn_inh = 2.0       # [ms]
ref_time = int(3 / dt)  # [steps]
Re = 333                # [Ohm cm] Resistance of extracellular space

if test_inter:
	Cm = 1              # [uF/cm2] membrane capacity
	g_Na = 120          # [S / cm2]
	g_K = 36            # [S / cm2]
	g_L = 0.3           # [S / cm2]
	Ra = 100            # [Ohm cm]
	d = 6               # [um]
	x = 2               # [um]
else:
	Cm = 2              # [uF/cm2] membrane capacity
	gnabar = 0.05       # [S / cm2]
	gkrect = 0.3        # [S / cm2]
	g_L = 0.002         # [S / cm2]
	Ra = 200            # [Ohm cm]
	d = 50              # [um] diameter
	x = 16              # [um] compartment length
	gcaN = 0.05         # const ???
	gcaL = 0.0001       # const ???
	gcak = 0.3          # const ???
	ca0 = 2             # const ???
	amA = 0.4           # const ???
	amB = 66            # const ???
	amC = 5             # const ???
	bmA = 0.4           # const ???
	bmB = 32            # const ???
	bmC = 5             # const ???
	R = 8.314472        # (k-mole) (joule/degC) const
	F = 96485.34        # (faraday) (kilocoulombs) const

Vm = np.full(fill_value=-72, shape=(nrns, segs), dtype=float)   # [mV] - array for three compartments volatge
n = np.ones(shape=(nrns, segs), dtype=float) * 0.105            # [0..1] compartments channel
m = np.ones(shape=(nrns, segs), dtype=float) * 0.079            # [0..1] compartments channel
h = np.ones(shape=(nrns, segs), dtype=float) * 0.67             # [0..1] compartments channel
cai = np.ones(shape=(nrns, segs), dtype=float) * 0.0001         # [0..1] compartments channel
hc = np.ones(shape=(nrns, segs), dtype=float) * 0.982           # [0..1] compartments channel
mc = np.ones(shape=(nrns, segs), dtype=float) * 0.0005          # [0..1] compartments channel
p = np.ones(shape=(nrns, segs), dtype=float) * 0.021            # [0..1] compartments channel

I_K = np.zeros(shape=(nrns, segs), dtype=float)                 # [nA] ionic currents
I_Na = np.zeros(shape=(nrns, segs), dtype=float)                # [nA] ionic currents
I_L = np.zeros(shape=(nrns, segs), dtype=float)                 # [nA] ionic currents
I_Ca = np.ones(shape=(nrns, segs), dtype=float) * -0.0004       # [nA] ionic currents
g_exc = np.zeros(shape=nrns, dtype=float)                       # [S] conductivity level
g_inh = np.zeros(shape=nrns, dtype=float)                       # [S] conductivity level
ref_time_timer = np.zeros(shape=nrns, dtype=float)              # [steps] refractory period timer

E_Ca = np.ones(shape=(nrns, segs), dtype=float) * 131           # [mV]
old_Vm = np.full(fill_value=-72, shape=nrns, dtype=float)       # [mV] old value of Vm

INP, MID, OUT = 0, 1, 2
spikes = []
# todo recheck
const3 = (np.log(np.sqrt(x ** 2 + d ** 2) + x) - np.log(np.sqrt(x ** 2 + d ** 2) - x)) / (4 * np.pi * x * Re)

axes_names = 'time I_Na I_K I_Ca E_Ca Volt m h n p mc hc'
debug_headers = "iter Vm vv1 vv2 INa IK1 IK2 IL ECa ICa1 ICa2 m h p n mc hc cai".split()
strformat = "{:<15.6f}" * len(debug_headers)
headformat = "{:<15}" * len(debug_headers)

GRAS_data = []
sim_time_steps = int(sim_time / dt)  # simulation time

def Exp(volt):
	if volt < -100:
		return 0
	return np.exp(volt)

def alpham(volt):
	if abs((volt + amB) / amC) < 1e-6:
		return amA * amC
	return (amA * (volt + amB)) / (1 - Exp(-(volt + amB) / amC))

def betam(volt):
	if abs((volt + bmB) / bmC) < 1e-6:
		return -bmA * bmC
	return (bmA * (-(volt + bmB))) / (1 - Exp((volt + bmB) / bmC))

def get_neuron_data():
	neuron_data = []
	with open('current_NEURON') as file1, open('mnh_NEURON') as file2:
		for lines in zip(file1, file2):
			# split by tabs and remove text of variables
			data = "\t".join(map(str.strip, lines)).split('\t')[1::2]
			neuron_data.append(list(map(float, data)))
	# remove duplicated dt rows
	neuron_data = np.array(neuron_data[::2])
	# remove duplicated time column of the second file
	neuron_data = np.delete(neuron_data, 6, axis=1)
	return neuron_data

# SIMULATION LOOP
def simulation():
	# for t in range(sim_time_steps):
	for t, vm_neuron in zip(range(sim_time_steps), get_neuron_data()[:, 5]):
		if t % 10 == 0 and debug:
			log.info(headformat.format(*debug_headers))
		GRAS_data.append([t * dt, I_Na[0, 0], I_K[0, 0], I_Ca[0, 0], E_Ca[0, 0], Vm[0, 0], m[0, 0], h[0, 0], n[0, 0], p[0, 0], mc[0, 0], hc[0, 0]])
		for nrn_id in range(nrns):
			if t * dt in stimulus:
				g_exc[0] = 0.5
			# save old value of Vm
			old_Vm[nrn_id] = vm_neuron # Vm[nrn_id, INP]
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
					nanew = gnabar * m[nrn_id, segment] ** 3 * h[nrn_id, segment] * (Vm[nrn_id, segment] - E_Na)
					iknew1 = gkrect * n[nrn_id, segment] ** 4 * (Vm[nrn_id, segment] - E_K)
					iknew2 = gcak * cai[nrn_id, segment] ** 2 / (cai[nrn_id, segment] ** 2 + 0.014 ** 2) * (Vm[nrn_id, segment] - E_K)
					ilnew = g_L * (E_L - Vm[nrn_id, segment])
					icanew1 = gcaN * mc[nrn_id, segment] ** 2 * hc[nrn_id, segment] * (Vm[nrn_id, segment] - E_Ca[nrn_id, segment])
					icanew2 = gcaL * p[nrn_id, segment] * (Vm[nrn_id, segment] - E_Ca[nrn_id, segment])
					ecanew = (1000 * R * 309.15 / (2 * F)) * np.log(ca0 / cai[nrn_id, segment])

					# calc K/NA/L currents
					I_Na[nrn_id, segment] = nanew
					I_K[nrn_id, segment] = iknew1 + iknew2
					I_L[nrn_id, segment] = ilnew
					# Ca channel
					I_Ca[nrn_id, segment] = icanew1 + icanew2
					E_Ca[nrn_id, segment] = ecanew

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

					if segment == 0 and nrn_id == 0 and debug:
						log.info(strformat.format(t, Vm[nrn_id, segment], vv1, vv2,
						                          nanew, iknew1, iknew2, ilnew, ecanew, icanew1, icanew2,
						                          m[nrn_id, segment], h[nrn_id, segment], p[nrn_id, segment],
						                          n[nrn_id, segment], mc[nrn_id, segment], hc[nrn_id, segment],
						                          cai[nrn_id, segment]))
					template = f"ERROR, step {t} | {nrn_id} neuron [{segment}] seg"
					assert 0 <= cai[nrn_id, segment] <= 0.05, f"cai " + template
					assert -200 <= Vm[nrn_id, segment] <= 200, f"Vm " + template
					assert 0 <= m[nrn_id, segment] <= 1, f"m " + template
					assert 0 <= n[nrn_id, segment] <= 1, f"n " + template
					assert 0 <= h[nrn_id, segment] <= 1, f"h " + template
					assert 0 <= p[nrn_id, segment] <= 1, f"p " + template
					assert 0 <= mc[nrn_id, segment] <= 1, f"mc " + template
					assert 0 <= hc[nrn_id, segment] <= 1, f"hc " + template
					#
					# I_leak = I_L[nrn_id, segment]
					# I_ionic = (I_K[nrn_id, segment] + I_Na[nrn_id, segment] + I_Ca[nrn_id, segment]) / (np.pi * x * d)
					# I_axonal = 0 #d / (4 * x * x * Ra) * (vv1 + vv2)
					I_inj = (I_syn_exc + I_syn_inh) / (np.pi * x * d) * 1000

					if segment == INP:
						Vm[nrn_id, segment] = vm_neuron #+= (dt / Cm) * (I_leak + I_ionic + I_axonal + I_inj)
					else:
						# dV = (dt / Cm) * (I_leak + I_ionic + I_axonal)
						Vm[nrn_id, segment] = vm_neuron #+= dV
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

					m[nrn_id, segment] += (1 - np.exp(-dt * (1 / tau_m))) * (m_inf / tau_m / (1 / tau_m) - m[nrn_id, segment])
					h[nrn_id, segment] += (1 - np.exp(-dt * (1 / tau_h))) * (h_inf / tau_h / (1 / tau_h) - h[nrn_id, segment])
					p[nrn_id, segment] += (1 - np.exp(-dt * (1 / tau_p))) * (p_inf / tau_p / (1 / tau_p) - p[nrn_id, segment])
					n[nrn_id, segment] += (1 - np.exp(-dt * (1 / tau_n))) * (n_inf / tau_n / (1 / tau_n) - n[nrn_id, segment])
					mc[nrn_id, segment] += (1 - np.exp(-dt * (1 / tau_mc))) * (mc_inf / tau_mc / (1 / tau_mc) - mc[nrn_id, segment])
					hc[nrn_id, segment] += (1 - np.exp(-dt * (1 / tau_hc))) * (hc_inf / tau_hc / (1 / tau_hc) - hc[nrn_id, segment])
					cai[nrn_id, segment] += (1 - np.exp(-dt * 0.04)) * (-0.01 * I_Ca[nrn_id, segment] / 0.04 - cai[nrn_id, segment])

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


def plot(gras_data, neuron_data):
	rows = 4
	cols = 3
	plt.close()
	fig, ax = plt.subplots(rows, cols)
	xticks = np.arange(neuron_data.shape[0]) * dt

	# plot NEURON
	row, col = 0, 0
	for i, (column_data, name) in enumerate(zip(neuron_data.T, axes_names.split())):
		if i % (rows - 1) == 0:
			row, col = i // (rows - 1), 0
		else:
			col += 1
		ax[row, col].plot(xticks, column_data, label='NEURON')
		ax[row, col].set_title(name)

	# plot GRAS
	row, col = 0, 0
	for i, column_data in enumerate(gras_data.T):
		if i % (rows - 1) == 0:
			row, col = i // (rows - 1), 0
		else:
			col += 1
		ax[row, col].plot(xticks, column_data, label='GRAS')
	plt.legend()
	plt.show()


if __name__ == "__main__":
	try:
		simulation()
	except AssertionError as mycheck:
		log.info(f'Assert error {mycheck}')
	except Exception as err:
		log.info(f'Common Exception: {err}')

	GRAS_data = np.array(GRAS_data)
	xlength = GRAS_data.shape[0]
	NEURON_data = get_neuron_data()[:xlength, :]
	log.info(f"GRAS shape {GRAS_data.shape}")
	log.info(f"NEURON shape {NEURON_data.shape}")

	plot(GRAS_data, NEURON_data)

