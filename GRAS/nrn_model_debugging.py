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
dt = 0.025  # [ms] - sim step
nrns = 1
segs = 3
sim_time = 500
stimulus = list(range(11, sim_time, 25))

E_Na = 50  # [mV] - reversal potential
E_K = -80  # [mV] - reversal potential
E_L = -70  # [mV] - reversal potential
E_ex = 50  # [mV] - reversal potential
E_in = -100  # [mV] reverse inh
V_adj = -63  # [mV] - adjust voltage for -55 threshold
tau_syn_exc = 0.3  # [ms]
tau_syn_inh = 2.0  # [ms]
ref_time = int(3 / dt)  # [steps]
Re = 333  # [Ohm cm] Resistance of extracellular space

if test_inter:
	Cm = 1          # [uF/cm2] membrane capacity
	g_Na = 120      # [S / cm2]
	g_K = 36        # [S / cm2]
	g_L = 0.3       # [S / cm2]
	Ra = 100        # [Ohm cm]
	d = 6           # [um]
	x = 2           # [um]
else:
	Cm = 2          # [uF/cm2] membrane capacity
	gnabar = 0.05   # [S/cm2]
	gkrect = 0.3    # [S/cm2]
	g_L = 0.002     # [S/cm2]
	Ra = 200        # [Ohm cm]
	d = 50          # [um] diameter
	x = 50          # [um] compartment length
	gcaN = 0.05     # const ???
	gcaL = 0.0001   # const ???
	gcak = 0.3      # const ???
	ca0 = 2         # const ???
	amA = 0.4       # const ???
	amB = 66        # const ???
	amC = 5         # const ???
	bmA = 0.4       # const ???
	bmB = 32        # const ???
	bmC = 5         # const ???
	R = 8.314472    # (k-mole) (joule/degC) const
	F = 96485.34    # (faraday) (kilocoulombs) const

nrn_shape = (nrns, segs)

Vm = np.full(nrn_shape, -70, dtype=np.float)  # [mV] - array for three compartments volatge
n = np.full(nrn_shape, 0.105, dtype=np.float)  # [0..1] compartments channel
m = np.full(nrn_shape, 0.079, dtype=np.float)  # [0..1] compartments channel
h = np.full(nrn_shape, 0.67, dtype=np.float)  # [0..1] compartments channel
cai = np.full(nrn_shape, 0.0001, dtype=np.float)  # [0..1] compartments channel
hc = np.full(nrn_shape, 0.982, dtype=np.float)  # [0..1] compartments channel
mc = np.full(nrn_shape, 0.0005, dtype=np.float)  # [0..1] compartments channel
p = np.full(nrn_shape, 0.021, dtype=np.float)  # [0..1] compartments channel

I_K = np.full(nrn_shape, 0, dtype=np.float)  # [nA] ionic currents
I_Na = np.full(nrn_shape, 0, dtype=np.float)  # [nA] ionic currents
I_L = np.full(nrn_shape, 0, dtype=np.float)  # [nA] ionic currents
I_Ca = np.full(nrn_shape, -0.0004, dtype=np.float)  # [nA] ionic currents
g_exc = np.full(nrns, 0, dtype=np.float)  # [S] conductivity level
g_inh = np.full(nrns, 0, dtype=np.float)  # [S] conductivity level
ref_time_timer = np.full(nrns, 0, dtype=np.float)  # [steps] refractory period timer

E_Ca = np.full(nrn_shape, 131, dtype=np.float)  # [mV]
old_Vm = np.full(nrns, -70, dtype=np.float)  # [mV] old value of Vm

INP, MID, OUT = 0, 1, 2
spikes = []
# todo recheck
const3 = (np.log(np.sqrt(x ** 2 + d ** 2) + x) - np.log(np.sqrt(x ** 2 + d ** 2) - x)) / (4 * np.pi * x * Re)

axes_names = 'time cai I_L I_Na I_K I_Ca E_Ca Voltage m h n p mc hc gsyn isyn _g _rhs'
debug_headers = "iter Vm vv1 vv2 INa IK1 IK2 IL ECa ICa1 ICa2 m h p n mc hc cai".split()
strformat = "{:<15.6f}" * len(debug_headers)
headformat = "{:<15}" * len(debug_headers)
rows = 5
cols = 4
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
	with open('nrn_500') as file:
		# remove headers
		file.readline()
		neuron_data = np.array([line.split() for line in file]).astype(float)
	return neuron_data

def simulation():
	# for t in range(sim_time_steps):
	# get_neuron_data()[:, 5] is hard inserted voltage from NEURON data
	for t, NRN_DATA in zip(range(sim_time_steps), get_neuron_data()):
		time_NRN, cai_NRN, I_L_NRN, I_Na_NRN, I_K_NRN, \
		I_Ca_NRN, E_Ca_NRN, Voltage_NRN, m_NRN, h_NRN, \
		n_NRN, p_NRN, mc_NRN, hc_NRN, g_NRN, i_NRN, _g_NRN, _rhs_NRN = list(NRN_DATA)

		if t % 10 == 0 and debug:
			log.info(headformat.format(*debug_headers))

		for nrn_id in range(nrns):
			if t * dt in stimulus:
				# [uS]
				g_exc[0] += 5.5
			# save old value of Vm
			old_Vm[nrn_id] = Vm[nrn_id, INP]
			# calculate synaptic currents
			# [nA] = [uS] * [mV]
			I_syn_exc = g_exc[nrn_id] * (Vm[nrn_id, INP] - E_ex)
			# [nA] = [uS] * [mV]
			I_syn_inh = g_inh[nrn_id] * (Vm[nrn_id, INP] - E_in)
			if nrn_id == 0:
				area = np.pi * x * d
				_g = g_NRN * (Voltage_NRN + .001 - E_ex) # [nA] = [uS] * [mV]
				_rhs = g_NRN * (Voltage_NRN - E_ex) # [nA]
				isyn = _rhs  # [nA]

				_g = (_g - _rhs) / .001
				_g *= 100 / area # [mA / cm2]
				_rhs *= 100 / area # [mA / cm2]

				GRAS_data.append([t * dt, cai[0, 0], I_L[0, 0], I_Na[0, 0], I_K[0, 0],
				                  I_Ca[0, 0], E_Ca[0, 0], Vm[0, 0], m[0, 0], h[0, 0],
				                  n[0, 0], p[0, 0], mc[0, 0], hc[0, 0], g_exc[0], isyn, _g, _rhs])
			# update compartments
			for segment in range(segs):
				if test_inter:
					# fixme Vm[nrn_id, segment] += np.random.uniform(-0.5, 0.5)
					# calc K/NA/L currents
					I_Na[nrn_id, segment] = g_Na * m[nrn_id, segment] ** 3 * h[nrn_id, segment] * (
								E_Na - Vm[nrn_id, segment])
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
					I_syn = (I_syn_exc + I_syn_inh) / (np.pi * x * d) * 10000
					#
					if segment == INP:
						if ref_time_timer[nrn_id] > 0:
							I_syn = 0
						Vm[nrn_id, segment] += (dt / Cm) * (I_leak + I_ionic + I_axonal + I_syn)
					else:
						dV = (dt / Cm) * (I_leak + I_ionic + I_axonal)
						Vm[nrn_id, segment] += dV
					#fixme
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
					#todo return after NEURON hard testing
					# nanew = gnabar * m[nrn_id, segment] ** 3 * h[nrn_id, segment] * (Vm[nrn_id, segment] - E_Na)
					# iknew1 = gkrect * n[nrn_id, segment] ** 4 * (Vm[nrn_id, segment] - E_K)
					# iknew2 = gcak * cai[nrn_id, segment] ** 2 / (cai[nrn_id, segment] ** 2 + 0.014 ** 2) * (Vm[nrn_id, segment] - E_K)
					# ilnew = g_L * (Vm[nrn_id, segment] - E_L)
					# icanew1 = gcaN * mc[nrn_id, segment] ** 2 * hc[nrn_id, segment] * (Vm[nrn_id, segment] - E_Ca[nrn_id, segment])
					# icanew2 = gcaL * p[nrn_id, segment] * (Vm[nrn_id, segment] - E_Ca[nrn_id, segment])
					# ecanew = (1000 * R * 309.15 / (2 * F)) * np.log(ca0 / cai[nrn_id, segment])

					# calc K/NA/L currents
					I_Na[nrn_id, segment] = I_Na_NRN  # nanew
					I_K[nrn_id, segment] = I_K_NRN  # iknew1 + iknew2
					I_L[nrn_id, segment] = I_L_NRN  # ilnew
					# Ca channel
					I_Ca[nrn_id, segment] = I_Ca_NRN  # icanew1 + icanew2
					E_Ca[nrn_id, segment] = E_Ca_NRN  # ecanew
					#todo return after NEURON hard testing
					# first segment
					# if segment == 0:
					# 	vv1 = 0
					# 	vv2 = Vm[nrn_id, segment + 1] - Vm[nrn_id, segment]
					# # the last segment
					# elif segment == segs - 1:
					# 	vv1 = Vm[nrn_id, segment - 1] - Vm[nrn_id, segment]
					# 	vv2 = 0
					# else:
					# 	vv1 = Vm[nrn_id, segment - 1] - Vm[nrn_id, segment]
					# 	vv2 = Vm[nrn_id, segment + 1] - Vm[nrn_id, segment]
					# if segment == 0 and nrn_id == 0 and debug:
					# 	log.info(strformat.format(t, Vm[nrn_id, segment], vv1, vv2,
					# 	                          nanew, iknew1, iknew2, ilnew, ecanew, icanew1, icanew2,
					# 	                          m[nrn_id, segment], h[nrn_id, segment], p[nrn_id, segment],
					# 	                          n[nrn_id, segment], mc[nrn_id, segment], hc[nrn_id, segment],
					# 	                          cai[nrn_id, segment]))
					# I_axonal = 0 #d / (4 * x * x * Ra) * (vv1 + vv2)
					template = f"ERROR, step {t} | {nrn_id} neuron [{segment}] seg"
					assert 0 <= cai[nrn_id, segment] <= 0.1, f"cai " + template
					assert -200 <= Vm[nrn_id, segment] <= 200, f"Vm( " + template
					assert 0 <= m[nrn_id, segment] <= 1, f"m " + template
					assert 0 <= n[nrn_id, segment] <= 1, f"n " + template
					assert 0 <= h[nrn_id, segment] <= 1, f"h " + template
					assert 0 <= p[nrn_id, segment] <= 1, f"p " + template
					assert 0 <= mc[nrn_id, segment] <= 1, f"mc " + template
					assert 0 <= hc[nrn_id, segment] <= 1, f"hc " + template
					# - [mA / cm2]
					I_leak = I_L[nrn_id, segment]
					I_ionic = (I_K[nrn_id, segment] + I_Na[nrn_id, segment] + I_Ca[nrn_id, segment])
					# I inj = I syn [mA / cm2]
					# https://www.neuron.yale.edu/phpBB/viewtopic.php?t=4081
					# [mA / cm2] = [nA] / (area [um2] * 0.01)
					area = np.pi * x * d
					I_syn = i_NRN #(I_syn_exc + I_syn_inh) / (area * 0.01)

					if segment == INP:
						Vm[nrn_id, segment] = Voltage_NRN # += (dt / Cm) * -(I_leak + I_ionic + I_syn)
					#todo fix axonal after test
					# else:
					# 	dV = (dt / Cm) * (I_leak + I_ionic)
					# 	Vm[nrn_id, segment] += dV
					# 	# if segment == MID:
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
					#
					m[nrn_id, segment] = m_NRN  # += (1 - np.exp(-dt * (1 / tau_m))) * (m_inf / tau_m / (1 / tau_m) - m[nrn_id, segment])
					h[nrn_id, segment] = h_NRN  # += (1 - np.exp(-dt * (1 / tau_h))) * (h_inf / tau_h / (1 / tau_h) - h[nrn_id, segment])
					p[nrn_id, segment] = p_NRN  # += (1 - np.exp(-dt * (1 / tau_p))) * (p_inf / tau_p / (1 / tau_p) - p[nrn_id, segment])
					n[nrn_id, segment] = n_NRN  # += (1 - np.exp(-dt * (1 / tau_n))) * (n_inf / tau_n / (1 / tau_n) - n[nrn_id, segment])
					mc[nrn_id, segment] = mc_NRN  # += (1 - np.exp(-dt * (1 / tau_mc))) * (mc_inf / tau_mc / (1 / tau_mc) - mc[nrn_id, segment])
					hc[nrn_id, segment] = hc_NRN  # += (1 - np.exp(-dt * (1 / tau_hc))) * (hc_inf / tau_hc / (1 / tau_hc) - hc[nrn_id, segment])
					cai[nrn_id, segment] = cai_NRN  # += (1 - np.exp(-dt * 0.04)) * (-0.01 * I_Ca[nrn_id, segment] / 0.04 - cai[nrn_id, segment])
			# end FOR segment
			# recalc conductivity
			# [uS] -= [uS] / [ms] * [ms]
			g_exc[nrn_id] -= g_exc[nrn_id] / tau_syn_exc * dt
			g_inh[nrn_id] -= g_inh[nrn_id] / tau_syn_inh * dt
			#todo
			# check on spike
			# if ref_time_timer[nrn_id] == 0 and -55 <= Vm[nrn_id, 0]:
			# 	ref_time_timer[nrn_id] = ref_time
			# spikes.append(t * dt)
			# update refractory period timer
			if ref_time_timer[nrn_id] > 0:
				ref_time_timer[nrn_id] -= 1


def plot(gras_data, neuron_data):
	plt.close()
	fig, ax = plt.subplots(rows, cols)
	xticks = np.arange(neuron_data.shape[0]) * dt
	# plot NEURON and GRAS
	for index, (neuron_data, gras_data, name) in enumerate(zip(neuron_data.T, gras_data.T, axes_names.split())):
		row = int(index / cols)
		col = index % cols
		ax[row, col].plot(xticks, neuron_data, label='NEURON', lw=3)
		ax[row, col].plot(xticks, gras_data, label='GRAS', lw=1, color='r')
		ax[row, col].plot(spikes, [np.mean(neuron_data)] * len(spikes), '.', color='r', ms=10)
		ax[row, col].plot(spikes, [np.mean(gras_data)] * len(spikes), '.', color='b', ms=5)
		ax[row, col].set_title(name)
		ax[row, col].set_xlim(0, sim_time)
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
