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
stimulus = (np.arange(11, sim_time, 25) / dt).astype(int)

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

Vm = np.full(nrn_shape, -70, dtype=np.float)        # [mV] - array for three compartments volatge
n = np.full(nrn_shape, 0.105, dtype=np.float)       # [0..1] compartments channel
m = np.full(nrn_shape, 0.079, dtype=np.float)       # [0..1] compartments channel
h = np.full(nrn_shape, 0.67, dtype=np.float)        # [0..1] compartments channel
cai = np.full(nrn_shape, 0.0001, dtype=np.float)    # [0..1] compartments channel
hc = np.full(nrn_shape, 0.982, dtype=np.float)      # [0..1] compartments channel
mc = np.full(nrn_shape, 0.0005, dtype=np.float)     # [0..1] compartments channel
p = np.full(nrn_shape, 0.021, dtype=np.float)       # [0..1] compartments channel

I_K = np.full(nrn_shape, 0, dtype=np.float)         # [nA] ionic currents
I_Na = np.full(nrn_shape, 0, dtype=np.float)        # [nA] ionic currents
I_L = np.full(nrn_shape, 0, dtype=np.float)         # [nA] ionic currents
I_Ca = np.full(nrn_shape, -0.0004, dtype=np.float)  # [nA] ionic currents
g_exc = np.full(nrns, 0, dtype=np.float)            # [S] conductivity level
g_inh = np.full(nrns, 0, dtype=np.float)            # [S] conductivity level
ref_time_timer = np.full(nrns, 0, dtype=np.float)   # [steps] refractory period timer

E_Ca = np.full(nrn_shape, 131, dtype=np.float)      # [mV]
old_Vm = np.full(nrns, -70, dtype=np.float)         # [mV] old value of Vm

NODE_RHS = np.zeros(shape=nrn_shape, dtype=np.float) # right hand side in node equation
NODE_D = np.zeros(shape=nrn_shape, dtype=np.float)   # diagonal element in node equation
NODE_A = np.zeros(shape=nrn_shape, dtype=np.float)   # is the effect of this node on the parent node's equation
NODE_B = np.zeros(shape=nrn_shape, dtype=np.float)   # is the effect of the parent node on this node's equation

_nt_ncell = 1
_nt_end = 3
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


def current(nrn, seg, voltage):
	global m, h, p, n, mc, hc, cai, I_Na, I_K, I_L, I_Ca, E_Ca

	I_Na[nrn, seg] = gnabar * m[nrn, seg] ** 3 * h[nrn, seg] * (voltage - E_Na)
	I_K[nrn, seg] = gkrect * n[nrn, seg] ** 4 * (voltage - E_K) + \
	                gcak * cai[nrn, seg] ** 2 / (cai[nrn, seg] ** 2 + 0.014 ** 2) * (voltage - E_K)
	I_L[nrn, seg] = g_L * (voltage - E_L)
	I_Ca[nrn, seg] = gcaN * mc[nrn, seg] ** 2 * hc[nrn, seg] * (voltage - E_Ca[nrn, seg]) + \
	                 gcaL * p[nrn, seg] * (voltage - E_Ca[nrn, seg])
	E_Ca[nrn, seg] = (1000 * R * 309.15 / (2 * F)) * np.log(ca0 / cai[nrn, seg])
	return I_Na[nrn, seg] + I_K[nrn, seg] + I_L[nrn, seg] + I_Ca[nrn, seg]

def recalc_channels(nrn, seg, V, NRN=None):
	global m, h, p, n, mc, hc, cai

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
	if NRN:
		m[nrn, seg], h[nrn, seg], p[nrn, seg], n[nrn, seg], mc[nrn, seg], hc[nrn, seg], cai[nrn, seg] = NRN
	else:
		m[nrn, seg] += (1 - np.exp(-dt * (1 / tau_m))) * (m_inf / tau_m / (1 / tau_m) - m[nrn, seg])
		h[nrn, seg] += (1 - np.exp(-dt * (1 / tau_h))) * (h_inf / tau_h / (1 / tau_h) - h[nrn, seg])
		p[nrn, seg] += (1 - np.exp(-dt * (1 / tau_p))) * (p_inf / tau_p / (1 / tau_p) - p[nrn, seg])
		n[nrn, seg] += (1 - np.exp(-dt * (1 / tau_n))) * (n_inf / tau_n / (1 / tau_n) - n[nrn, seg])
		mc[nrn, seg] += (1 - np.exp(-dt * (1 / tau_mc))) * (mc_inf / tau_mc / (1 / tau_mc) - mc[nrn, seg])
		hc[nrn, seg] += (1 - np.exp(-dt * (1 / tau_hc))) * (hc_inf / tau_hc / (1 / tau_hc) - hc[nrn, seg])
		cai[nrn, seg] += (1 - np.exp(-dt * 0.04)) * (-0.01 * I_Ca[nrn, seg] / 0.04 - cai[nrn, seg])

	template = f"ERROR {nrn} neuron [{seg}] seg"
	assert 0 <= cai[nrn, seg] <= 0.1, f"cai " + template
	assert -200 <= Vm[nrn, seg] <= 200, f"Vm " + template
	assert 0 <= m[nrn, seg] <= 1, f"m " + template
	assert 0 <= n[nrn, seg] <= 1, f"n " + template
	assert 0 <= h[nrn, seg] <= 1, f"h " + template
	assert 0 <= p[nrn, seg] <= 1, f"p " + template
	assert 0 <= mc[nrn, seg] <= 1, f"mc " + template
	assert 0 <= hc[nrn, seg] <= 1, f"hc " + template

# three segments as default [CAP seg CAP]
def simulation():
	global NODE_RHS, NODE_D, NODE_A, NODE_B, Vm, GRAS_data
	i1 = 0
	i2 = i1 + _nt_ncell
	i3 = _nt_end
	# a is the effect of this node on the parent node's equation
	# b is the effect of the parent node on this node's equation
	A = -39.269908
	B = -0.5
	# for t in range(sim_time_steps):
	# get_neuron_data()[:, 5] is hard inserted voltage from NEURON data
	for t, NRN_DATA in zip(range(sim_time_steps), get_neuron_data()):
		time_NRN, cai_NRN, I_L_NRN, I_Na_NRN, I_K_NRN, \
		I_Ca_NRN, E_Ca_NRN, Voltage_NRN, m_NRN, h_NRN, \
		n_NRN, p_NRN, mc_NRN, hc_NRN, g_NRN, syn_NRN, _g_NRN, _rhs_NRN = list(NRN_DATA)

		if t % 10 == 0 and debug:
			log.info(headformat.format(*debug_headers))

		for nrn in range(nrns):
			# if t in stimulus:
			# 	# [uS]
			# 	g_exc[0] += 5.5
			####################################################
			################ setup_tree_matrix #################
			####################################################
			'''calculate right hand side of
			cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
			cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
			This is a common operation for fixed step, cvode, and daspk methods'''

			"""void nrn_rhs(NrnThread *_nt)"""
			# make _rhs zero
			for i in range(i1, i3):
				NODE_RHS[nrn, i] = 0
			# update rhs from MOD, CAPS has 0 current!
			for seg in range(segs):
				if seg == INP or seg == OUT:
					continue
				V = Voltage_NRN # Vm[nrn, seg]
				# calc K/NA/L currents
				recalc_channels(nrn, seg, V)
				# calc additional stuff
				_rhs = current(nrn, seg, V)
				print(_rhs, _rhs_NRN)
				_g = current(nrn, seg, V + 0.001)
				_g = (_g - _rhs) / 0.001
				NODE_RHS[nrn, seg] += _rhs
			# end FOR segments
			'''
			# activsynapse_rhs()
			NODE_rhs[nrn, seg] += 0
			# if EXTRA
			# Cannot have any axial terms yet so that i(vm) can be calculated from
			# i(vm)+is(vi) and is(vi) which are stored in rhs vector
			# nrn_rhs_ext(_nt);
			NODE_rhs[nrn, seg] += 0
			# nrn_rhs_ext has also computed the the internal axial current
			# for those nodes containing the extracellular mechanism
			# activstim_rhs()
			NODE_rhs[nrn, seg] += 0
			# activclamp_rhs()
			NODE_rhs[nrn, seg] += 0
			'''
			for i in range(i2, i3):
				nd = i
				pnd = i - 1
				# double dv = NODEV(pnd) - NODEV(nd);
				dv = Vm[nrn, pnd] - Vm[nrn, nd]
				NODE_RHS[nrn, nd] -= B * dv
				NODE_RHS[nrn, pnd] -= A * dv
			print(f"NODE_rhs[MID] = {NODE_RHS[nrn, MID]}")

			"""void nrn_lhs(NrnThread *_nt)"""
			# make _lhs zero
			for i in range(i1, i3):
				NODE_D[nrn, i] = 0
			# update rhs from MOD, CAPS has 0 current!
			for seg in range(segs):
				if seg == INP or seg == OUT:
					continue
				# note that CAP has no jacob
				#todo check info about _g updating (where is it stored?)
				NODE_D[nrn, MID] += _g
			'''
			if (secondorder) { nt->cj = 2.0/dt; }
			else { nt->cj = 1.0/dt; }
			'''
			# nrn_cap_jacob(_nt, _nt->tml->ml);
			cj = 1 / dt
			# cfac = .001 * _nt->cj
			cfac = 0.001 * cj
			# or (i=0; i < count; ++i) {
			for i in range(_nt_ncell):
				NODE_D[nrn, i] += cfac * Cm
			'''
			# activsynapse_lhs()
			NODE_D[nrn, seg] += 0
			#nrn_setup_ext(_nt);
			NODE_D[nrn, seg] += 0
			# activclamp_lhs();
			NODE_D[nrn, seg] += 0
			'''
			for i in range(i2, i3):
				nd = i2
				pnd = i2 - 1
				# NODED(_nt->_v_node[i]) -= NODEB(_nt->_v_node[i]);
				# NODED(_nt->_v_parent[i]) -= NODEA(_nt->_v_node[i]);
				NODE_D[nrn, nd] -= B
				NODE_D[nrn, pnd] -= A

			####################################################
			################### nrn_solve ######################
			####################################################
			'''void nrn_solve(NrnThread* _nt)'''
			# triang(_nt);
			for i in range(i3 - 1, i2 + 1, -1):
				nd = i
				pnd = i - 1
				'''
				p = NODEA(nd) / NODED(nd);
				NODED(pnd) -= p * NODEB(nd);
				NODERHS(pnd) -= p * NODERHS(nd);
				'''
				ppp = A / NODE_D[nrn, nd]
				NODE_D[nrn, pnd] -= ppp * B
				NODE_RHS[nrn, pnd] -= ppp * NODE_RHS[nd]

			'''void bksub(NrnThread* _nt)'''
			# bksub(_nt);
			for i in range(i1, i2):
				NODE_RHS[nrn, i] /= NODE_D[nrn, i]
			for i in range(i2, i3):
				nd = i
				pnd = i - 1
				'''
				NODERHS(cnd) -= NODEB(cnd) * NODERHS(nd);
				NODERHS(cnd) /= NODED(cnd);
				'''
				NODE_RHS[nrn, nd] -= B * NODE_RHS[nrn, pnd]
				NODE_RHS[nrn, nd] /= NODE_D[nrn, nd]

			print(f"NODE_D[MID] = {NODE_D[nrn, MID]}")

			####################################################
			##################### update #######################
			####################################################
			for i in range(i1, i2):
				Vm[nrn, i] += NODE_RHS[nrn, i]

			# nrn_update_2d(_nt);

			g_exc[nrn] -= g_exc[nrn] / tau_syn_exc * dt
			g_inh[nrn] -= g_inh[nrn] / tau_syn_inh * dt
			# todo
			# check on spike
			# if ref_time_timer[nrn_id] == 0 and -55 <= Vm[nrn_id, 0]:
			# 	ref_time_timer[nrn_id] = ref_time
			# spikes.append(t * dt)
			# update refractory period timer
			if ref_time_timer[nrn] > 0:
				ref_time_timer[nrn] -= 1
			print("= " * 10)

		GRAS_data.append([t * dt, cai[0, 0], I_L[0, 0], I_Na[0, 0], I_K[0, 0],
		                  I_Ca[0, 0], E_Ca[0, 0], Vm[0, 0], m[0, 0], h[0, 0],
		                  n[0, 0], p[0, 0], mc[0, 0], hc[0, 0], g_exc[0], 0, _g, NODE_RHS[0, 0]])


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
		passed = np.mean(np.abs(neuron_data - gras_data)) / (np.max(neuron_data) - np.min(neuron_data)) * 100
		ax[row, col].set_title(f"{name} [{passed:.2f}%]")
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


"""
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
"""
