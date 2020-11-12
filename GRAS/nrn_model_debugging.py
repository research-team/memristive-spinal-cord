"""
Formulas and value units were taken from:

Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011).
Principles of Computational Modelling in Neuroscience. Cambridge: Cambridge University Press.
DOI:10.1017/CBO9780511975899

Based on the NEURON repository
"""
import numpy as np
import logging as log
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)

dt = 0.025  # [ms] - sim step
sim_time = 50
sim_time_steps = int(sim_time / dt)
save_neuron_id = 1
stimulus = (np.arange(10, sim_time, 25) / dt).astype(int)

"""common const"""
V_th = -40
E_ex = 50           # [mV] reversal potential
E_in = -80          # [mV] reverse inh
V_adj = -63         # [mV] adjust voltage for -55 threshold
tau_syn_exc = 0.3   # [ms] todo
tau_syn_inh = 2.0   # [ms] todo
"""moto const"""
ca0 = 2     # const ??? todo
amA = 0.4   # const ??? todo
amB = 66    # const ??? todo
amC = 5     # const ??? todo
bmA = 0.4   # const ??? todo
bmB = 32    # const ??? todo
bmC = 5     # const ??? todo
R_const = 8.314472    # [k-mole] or [joule/degC] const
F_const = 96485.34    # [faraday] or [kilocoulombs] const
"""muscle const"""
g_kno = 0.01    # [S/cm2] ?? todo
g_kir = 0.03    # [S/cm2] ?? todo
# Boltzman steady state curve
vhalfl = -98.92 # [mV] fitted to patch data, Stegen et al. 2012
kl = 10.89      # [mV] Stegen et al. 2012
# tau_infty
vhalft = 67.0828    # [mV] fitted #100 uM sens curr 350a, Stegen et al. 2012
at = 0.00610779     # [/ ms] Stegen et al. 2012
bt = 0.0817741      # [/ ms] Note: typo in Stegen et al. 2012
# Temperature dependence
q10 = 1         # temperature scaling
celsius = 36    # [degC]

# fixme
nodecount = 3   # segments
_nt_ncell = 1   # neurons count
# fixme ONLY FOR REALIZATION WITH ONE NEURON/ARRAY
_nt_end = nodecount + 2  # 1 + position of last in v_node array)
nnode = nodecount + 1    # number of nodes for ith section
segs = list(range(_nt_end))
i1 = 0
i2 = i1 + _nt_ncell
i3 = _nt_end

def init0(shape, dtype=np.float):
	return np.zeros(shape, dtype=dtype)

nrns_number = 0
nrn_models = []
# common properties
Cm = init0(100)         # [uF/cm2] membrane capacity
gnabar = init0(100)     # [S / cm2] todo
gkbar = init0(100)      # [S / cm2]g_K todo
gl = init0(100)         # [S / cm2] todo
Ra = init0(100)         # [Ohm cm]
diam = init0(100)       # [um] diameter
length = init0(100)         # [um] compartment length
ena = init0(100)        # [mV] todo
ek = init0(100)         # [mV] todo
el = init0(100)         # [mV] todo
# moto properties
gkrect = init0(100)     # [S / cm2] todo
gcaN = init0(100)       # [S / cm2] todo
gcaL = init0(100)       # [S / cm2] todo
gcak = init0(100)       # [S / cm2] todo
# synapses data
syn_pre_nrn = []
syn_post_nrn = []
syn_weight = []
syn_delay = []
syn_delay_timer = []

def create(number, model='inter'):
	"""

	"""
	global nrns_number, nrn_models
	ids = list(range(nrns_number, nrns_number + number))
	#
	__Cm = None
	__gnabar = None
	__gkbar = None
	__gl = None
	__Ra = None
	__ena = None
	__ek = None
	__el = None
	__diam = None
	__dx = None
	__gkrect = None
	__gcaN = None
	__gcaL = None
	__gcak = None
	# without random at first stage of debugging
	for nrnid in ids:
		if model == 'inter':
			__Cm = 1
			__gnabar = 120
			__gkbar = 36
			__gl = 0.3
			__Ra = 100
			__ena = 50
			__ek = -80
			__el = -70
			__diam = 6
			__dx = 6
		elif model == 'moto':
			__Cm = 2
			__gnabar = 0.05
			__gl = 0.002
			__Ra = 200
			__ena = 50
			__ek = -80
			__el = -70
			__diam = 50
			__dx = 50
			__gkrect = 0.3
			__gcaN = 0.05
			__gcaL = 0.0001
			__gcak = 0.3
		elif model == 'muscle':
			__Cm = 3.6
			__gnabar = 0.15
			__gkbar = 0.03
			__gl = 0.0002
			__Ra = 1.1
			__ena = 55
			__ek = -80
			__el = -72
			__diam = 40
			__dx = 3000
		elif model == 'generator':
			pass
		else:
			raise Exception("Choose the model")

		# common properties
		Cm[nrnid] = __Cm
		gnabar[nrnid] = __gnabar
		gkbar[nrnid] = __gkbar
		gl[nrnid] = __gl
		el[nrnid] = __el
		ena[nrnid] = __ena
		ek[nrnid] = __ek
		Ra[nrnid] = __Ra
		diam[nrnid] = __diam
		length[nrnid] = __dx
		gkrect[nrnid] = __gkrect
		gcaN[nrnid] = __gcaN
		gcaL[nrnid] = __gcaL
		gcak[nrnid] = __gcak

	nrns_number += number
	nrn_models += [model] * number
	return ids

def connect(pre_nrns, post_nrns, delay, weight, typ='all-to-all'):
	"""

	"""
	if typ == 'all-to-all':
		for pre in pre_nrns:
			for post in post_nrns:
				syn_pre_nrn.append(pre)
				syn_post_nrn.append(post)
				syn_weight.append(weight)
				syn_delay.append(int(delay / dt))
				syn_delay_timer.append(-1)

gen = create(1, model='generator')
# n1 = create(1, model='moto')
m1 = create(1, model='muscle')
connect(gen, m1, delay=1, weight=40.5, typ='all-to-all')
# connect(n1, m1, delay=1, weight=40.5, typ='all-to-all')

nrns = list(range(nrns_number))
nrn_shape = (nrns_number, _nt_end)

# global variables
Vm = init0(nrn_shape)           # [mV] array for three compartments volatge
n = init0(nrn_shape)            # [0..1] compartments channel
m = init0(nrn_shape)            # [0..1] compartments channel
h = init0(nrn_shape)            # [0..1] compartments channel
l = init0(nrn_shape)            # [0..1] compartments channel
s = init0(nrn_shape)            # [0..1] compartments channel
p = init0(nrn_shape)            # [0..1] compartments channel
hc = init0(nrn_shape)           # [0..1] compartments channel
mc = init0(nrn_shape)           # [0..1] compartments channel
cai = init0(nrn_shape)          # [0..1] compartments channel
I_L = init0(nrn_shape)          # [nA] leak ionic currents
I_K = init0(nrn_shape)          # [nA] K ionic currents
I_Na = init0(nrn_shape)         # [nA] Na ionic currents
I_Ca = init0(nrn_shape)         # [nA] Ca ionic currents
E_Ca = init0(nrn_shape)         # [mV] Ca reversal potential
g_exc = init0(nrns_number)      # [S] excitatory conductivity level
g_inh = init0(nrns_number)      # [S] inhibitory conductivity level
NODE_A = init0(nrn_shape)       # the effect of this node on the parent node's equation
NODE_B = init0(nrn_shape)       # the effect of the parent node on this node's equation
NODE_D = init0(nrn_shape)       # diagonal element in node equation
NODE_RHS = init0(nrn_shape)     # right hand side in node equation
NODE_RINV = init0(nrn_shape)    # conductance uS from node to parent
NODE_AREA = init0(nrn_shape)    # area of a node in um^2
has_spike = init0(nrns_number, dtype=bool)  # spike flag for each neuron
spike_on = init0(nrns_number, dtype=bool)  # spike flag for each neuron

spikes = []
GRAS_data = []

def get_neuron_data():
	# with open("/home/alex/NRNTEST/classic/motoneuron.log") as file:
	# 	file.readline()
	# 	neuron_data = [line.split("\t") for line in file]
	# 	neuron_data.append(neuron_data[-1])
	# 	neuron_data = np.array(neuron_data[::6]).astype(np.float)
	# return neuron_data
	with open("/home/alex/NRNTEST/muscle/kek2") as file:
		# il, ina, ik, m, h, n, v
		neuron_data = []
		for line in file:
			neuron_data.append(line.replace('BREAKPOINT currents ', '').split("\t"))
		neuron_data = neuron_data[::6]
		neuron_data = np.array(neuron_data).astype(np.float)
	return neuron_data

def save_data(to_end=False):
	inrn = save_neuron_id
	MID = 2

	if to_end:
		# GRAS_data[-1] += [NODE_A[inrn, 0], NODE_B[inrn, 0], NODE_D[inrn, 0], NODE_RINV[inrn, 0], Vm[inrn, 0], NODE_RHS[inrn, 0],
		#                   NODE_A[inrn, 1], NODE_B[inrn, 1], NODE_D[inrn, 1], NODE_RINV[inrn, 1], Vm[inrn, 1], NODE_RHS[inrn, 1],
		#                   NODE_A[inrn, 2], NODE_B[inrn, 2], NODE_D[inrn, 2], NODE_RINV[inrn, 2], Vm[inrn, 2], NODE_RHS[inrn, 2]]
		GRAS_data[-1] += [Vm[inrn, MID]]
	else:
		# il, ina, ik, m, h, n, l, s, v
		GRAS_data.append([I_L[inrn, MID], I_Na[inrn, MID], I_K[inrn, MID], m[inrn, MID], h[inrn, MID], n[inrn, MID], l[inrn, MID], s[inrn, MID]])

def Exp(volt):
	return 0 if volt < -100 else np.exp(volt)

def alpham(volt):
	if abs((volt + amB) / amC) < 1e-6:
		return amA * amC
	return amA * (volt + amB) / (1 - Exp(-(volt + amB) / amC))

def betam(volt):
	if abs((volt + bmB) / bmC) < 1e-6:
		return -bmA * bmC
	return -bmA * (volt + bmB) / (1 - Exp((volt + bmB) / bmC))

def syn_current(nrn, voltage):
	"""

	"""
	return g_exc[nrn] * (voltage - E_ex) + g_inh[nrn] * (voltage - E_in)

def nrn_moto_current(nrn, seg, voltage):
	"""

	"""
	ina = gnabar[nrn] * m[nrn, seg] ** 3 * h[nrn, seg] * (voltage - ena[nrn])
	ik = gkrect[nrn] * n[nrn, seg] ** 4 * (voltage - ek[nrn]) + gcak[nrn] * cai[nrn, seg] ** 2 / (cai[nrn, seg] ** 2 + 0.014 ** 2) * (voltage - ek[nrn])
	il = gl[nrn] * (voltage - el[nrn])
	Eca = (1000 * R_const * 309.15 / (2 * F_const)) * np.log(ca0 / cai[nrn, seg])
	ica = gcaN[nrn] * mc[nrn, seg] ** 2 * hc[nrn, seg] * (voltage - Eca) + gcaL[nrn] * p[nrn, seg] * (voltage - Eca)
	# save
	I_Na[nrn, seg] = ina
	I_K[nrn, seg] = ik
	I_L[nrn, seg] = il
	E_Ca[nrn, seg] = Eca
	I_Ca[nrn, seg] = ica

	return ina + ik + il + ica

def nrn_muscle_current(nrn, seg, voltage):
	"""

	"""
	I_Na[nrn, seg] = gnabar[nrn] * m[nrn, seg] ** 3 * h[nrn, seg] * (voltage - ena[nrn])
	I_K[nrn, seg] = gkbar[nrn] * n[nrn, seg] ** 4 * (voltage - ek[nrn])
	I_L[nrn, seg] = gl[nrn] * (voltage - el[nrn])
	return I_Na[nrn, seg] + I_K[nrn, seg] + I_L[nrn, seg]

def recalc_synaptic(nrn):
	"""

	"""
	if g_exc[nrn] != 0:
		g_exc[nrn] -= g_exc[nrn] / tau_syn_exc * dt
		# fixme is worth?
		if g_exc[nrn] < 1e-10:
			g_exc[nrn] = 0
	if g_inh[nrn] != 0:
		g_inh[nrn] -= g_inh[nrn] / tau_syn_inh * dt
		# fixme is worth?
		if g_inh[nrn] < 1e-10:
			g_inh[nrn] = 0

def nrn_inter_initial(nrn, seg, V):
	"""
	evaluate_fct cropped
	"""
	V_mem = V - V_adj
	a = 0.32 * (13 - V_mem) / (np.exp((13 - V_mem) / 4) - 1)
	b = 0.28 * (V_mem - 40) / (np.exp((V_mem - 40) / 5) - 1)
	m_inf = a / (a + b)

	a = 0.128 * np.exp((17 - V_mem) / 18)
	b = 4 / (1 + np.exp((40 - V_mem) / 5))
	h_inf = a / (a + b)

	a = 0.032 * (15 - V_mem) / (np.exp((15 - V_mem) / 5) - 1)
	b = 0.5 * np.exp((10 - V_mem) / 40)
	n_inf = a / (a + b)
	"""INITIAL"""
	m[nrn, seg] = m_inf
	h[nrn, seg] = h_inf
	n[nrn, seg] = n_inf

def nrn_moto_initial(nrn, seg, V):
	"""
	evaluate_fct cropped
	"""
	a = alpham(V)
	b = betam(V)
	m_inf = a / (a + b)
	h_inf = 1 / (1 + Exp((V + 65) / 7))
	n_inf = 1 / (1 + Exp(-(V + 38) / 15))
	mc_inf = 1 / (1 + Exp(-(V + 32) / 5))
	hc_inf = 1 / (1 + Exp((V + 50) / 5))
	p_inf = 1 / (1 + Exp(-(V + 55.8) / 3.7))
	"""INITIAL"""
	m[nrn, seg] = m_inf
	h[nrn, seg] = h_inf
	p[nrn, seg] = p_inf
	n[nrn, seg] = n_inf
	mc[nrn, seg] = mc_inf
	hc[nrn, seg] = hc_inf
	cai[nrn, seg] = 0.0001

def nrn_muslce_initial(nrn, seg, V):
	"""
	evaluate_fct cropped
	"""
	V_mem = V - V_adj
	a = 0.32 * (13 - V_mem) / (np.exp((13 - V_mem) / 4) - 1)
	b = 0.28 * (V_mem - 40) / (np.exp((V_mem - 40) / 5) - 1)
	m_inf = a / (a + b)

	a = 0.128 * np.exp((17 - V_mem) / 18)
	b = 4 / (1 + np.exp((40 - V_mem) / 5))
	h_inf = a / (a + b)

	a = 0.032 * (15 - V_mem) / (np.exp((15 - V_mem) / 5) - 1)
	b = 0.5 * np.exp((10 - V_mem) / 40)
	n_inf = a / (a + b)
	"""INITIAL"""
	m[nrn, seg] = m_inf
	h[nrn, seg] = h_inf
	n[nrn, seg] = n_inf

def recalc_moto_channels(nrn, seg, V):
	"""

	"""
	# BREAKPOINT -> states -> evaluate_fct
	"""evaluate_fct"""
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
	"""states"""
	m[nrn, seg] += (1 - np.exp(-dt * (1 / tau_m))) * (m_inf / tau_m / (1 / tau_m) - m[nrn, seg])
	h[nrn, seg] += (1 - np.exp(-dt * (1 / tau_h))) * (h_inf / tau_h / (1 / tau_h) - h[nrn, seg])
	p[nrn, seg] += (1 - np.exp(-dt * (1 / tau_p))) * (p_inf / tau_p / (1 / tau_p) - p[nrn, seg])
	n[nrn, seg] += (1 - np.exp(-dt * (1 / tau_n))) * (n_inf / tau_n / (1 / tau_n) - n[nrn, seg])
	mc[nrn, seg] += (1 - np.exp(-dt * (1 / tau_mc))) * (mc_inf / tau_mc / (1 / tau_mc) - mc[nrn, seg])
	hc[nrn, seg] += (1 - np.exp(-dt * (1 / tau_hc))) * (hc_inf / tau_hc / (1 / tau_hc) - hc[nrn, seg])
	cai[nrn, seg] += (1 - np.exp(-dt * 0.04)) * (-0.01 * I_Ca[nrn, seg] / 0.04 - cai[nrn, seg])

	assert 0 <= cai[nrn, seg] <= 0.1
	assert -200 <= Vm[nrn, seg] <= 200
	assert 0 <= m[nrn, seg] <= 1
	assert 0 <= n[nrn, seg] <= 1
	assert 0 <= h[nrn, seg] <= 1
	assert 0 <= p[nrn, seg] <= 1
	assert 0 <= mc[nrn, seg] <= 1
	assert 0 <= hc[nrn, seg] <= 1

def recalc_muslce_channels(nrn, seg, V):
	"""

	"""
	# BREAKPOINT -> states -> evaluate_fct
	"""evaluate_fct"""
	V_mem = V - V_adj
	#
	a = 0.32 * (13.0 - V_mem) / (np.exp((13.0 - V_mem) / 4.0) - 1.0)
	b = 0.28 * (V_mem - 40.0) / (np.exp((V_mem - 40.0) / 5.0) - 1.0)
	tau_m = 1.0 / (a + b)
	m_inf = a / (a + b)
	#
	a = 0.128 * np.exp((17.0 - V_mem) / 18.0)
	b = 4.0 / (1.0 + np.exp((40.0 - V_mem) / 5.0))
	tau_h = 1.0 / (a + b)
	h_inf = a / (a + b)
	#
	a = 0.032 * (15.0 - V_mem) / (np.exp((15.0 - V_mem) / 5.0) - 1.0)
	b = 0.5 * np.exp((10.0 - V_mem) / 40.0)
	tau_n = 1 / (a + b)
	n_inf = a / (a + b)
	#
	qt = q10 ** ((celsius - 33.0) / 10.0)
	linf = 1.0 / (1.0 + np.exp((V - vhalfl) / kl))
	taul = 1.0 / (qt * (at * np.exp(-V / vhalft) + bt * np.exp(V / vhalft)))
	alpha = 0.3 / (1.0 + np.exp((V + 43.0) / - 5.0))
	beta = 0.03 / (1.0 + np.exp((V + 80.0) / - 1.0))
	summ = alpha + beta
	stau = 1.0 / summ
	sinf = alpha / summ
	"""states"""
	m[nrn, seg] += (1 - np.exp(dt * (-1 / tau_m))) * (-(m_inf / tau_m) / (-1 / tau_m) - m[nrn, seg])
	h[nrn, seg] += (1 - np.exp(dt * (-1 / tau_h))) * (-(h_inf / tau_h) / (-1 / tau_h) - h[nrn, seg])
	n[nrn, seg] += (1 - np.exp(dt * (-1 / tau_n))) * (-(n_inf / tau_n) / (-1 / tau_n) - n[nrn, seg])
	l[nrn, seg] += (1 - np.exp(dt * (-1 / taul))) * (-(linf / taul) / (-1 / taul) - l[nrn, seg])
	s[nrn, seg] += (1 - np.exp(dt * (-1 / stau))) * (-(sinf / stau) / (-1 / stau) - s[nrn, seg])

	assert -200 <= Vm[nrn, seg] <= 200
	assert 0 <= m[nrn, seg] <= 1
	assert 0 <= n[nrn, seg] <= 1
	assert 0 <= h[nrn, seg] <= 1
	assert 0 <= l[nrn, seg] <= 1
	assert 0 <= s[nrn, seg] <= 1

def nrn_rhs(nrn):
	"""
	void nrn_rhs(NrnThread *_nt) combined with the first part of nrn_lhs
	calculate right hand side of
	cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	This is a common operation for fixed step, cvode, and daspk methods
	"""
	# init _rhs and _lhs (NODE_D) as zero
	for nd in range(i1, i3):
		NODE_RHS[nrn, nd] = 0
		NODE_D[nrn, nd] = 0
	# update MOD rhs, CAPS has no current [CAP MOD CAP]!
	for seg in segs:
		if seg == segs[0] or seg == segs[-1]:
			continue
		V = Vm[nrn, seg]

		# SYNAPTIC update
		if seg == 2: #fixme only for muscles [0, 1, 2MID, 3, 4]
			# static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
			_g = syn_current(nrn, V + 0.001)
			_rhs = syn_current(nrn, V)
			_g = (_g - _rhs) / .001
			_g *= 1.e2 / NODE_AREA[nrn, seg]
			_rhs *= 1.e2 / NODE_AREA[nrn, seg]
			NODE_RHS[nrn, seg] -= _rhs
			# static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
			NODE_D[nrn, seg] += _g

		# NEURON update
		# static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		if nrn_models[nrn] == 'inter':
			# muscle and inter has the same fast_channel function
			_g = nrn_muscle_current(nrn, seg, V + 0.001)
			_rhs = nrn_muscle_current(nrn, seg, V)
		elif nrn_models[nrn] == 'moto':
			_g = nrn_moto_current(nrn, seg, V + 0.001)
			_rhs = nrn_moto_current(nrn, seg, V)
		elif nrn_models[nrn] == 'muscle':
			# muscle and inter has the same fast_channel function
			_g = nrn_muscle_current(nrn, seg, V + 0.001)
			_rhs = nrn_muscle_current(nrn, seg, V)
		else:
			raise Exception('No nrn model found')
		# save data like in NEURON (after .mod nrn_cur)
		if nrn == save_neuron_id and seg == 2:
			save_data()
		_g = (_g - _rhs) / 0.001
		NODE_RHS[nrn, seg] -= _rhs
		# static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
		NODE_D[nrn, seg] += _g
	# end FOR segments

	# activsynapse_rhs()
	# NODE_rhs[nrn, seg] += 0
	# if EXTRA: nrn_rhs_ext(_nt);
	# NODE_rhs[nrn, seg] += 0
	# activstim_rhs()
	# NODE_rhs[nrn, seg] += 0
	# activclamp_rhs()
	# NODE_rhs[nrn, seg] += 0

	# todo always 0, because Vm0 = Vm1 = Vm2 at [CAP node CAP] model (1 section)
	for nd in range(i2, i3):
		pnd = nd - 1
		dv = Vm[nrn, pnd] - Vm[nrn, nd]
		# our connection coefficients are negative so
		NODE_RHS[nrn, nd] -= NODE_B[nrn, nd] * dv
		NODE_RHS[nrn, pnd] += NODE_A[nrn, nd] * dv
	print("After RHS JACOB NODED", NODE_D[nrn, :])

def nrn_lhs(nrn):
	"""
	void nrn_lhs(NrnThread *_nt)
	NODE_D[nrn, nd] updating is located at nrn_rhs, because _g is not the global variable
	"""
	# nt->cj = 2/dt if (secondorder) else 1/dt
	# note, the first is CAP
	# function nrn_cap_jacob(_nt, _nt->tml->ml);
	print(f"After LHS JACOB NODED (above) {NODE_D[nrn, :]}")

	cj = 1 / dt
	cfac = 0.001 * cj
	# fixme +1 for nodelist
	for nd in range(0 + 1, nodecount + 1):
		NODE_D[nrn, nd] += cfac * Cm[nrn]
	print(f"before activsynapse_lhs NODED= {NODE_D[nrn, :]}")

	# activsynapse_lhs()
	# NODE_D[nrn, seg] += 0
	# nrn_setup_ext(_nt);
	# NODE_D[nrn, seg] += 0
	# activclamp_lhs();
	# NODE_D[nrn, seg] += 0

	# updating NODED
	for nd in range(i2, i3):
		pnd = nd - 1
		NODE_D[nrn, nd] -= NODE_B[nrn, nd]
		NODE_D[nrn, pnd] -= NODE_A[nrn, nd]
	print(f"After AXIAL currents NODED {NODE_D[nrn, :]}")

def bksub(nrn):
	"""
	void bksub(NrnThread* _nt)
	"""
	for nd in range(i1, i2):
		NODE_RHS[nrn, nd] /= NODE_D[nrn, nd]
	for nd in range(i2, i3):
		pnd = nd - 1
		NODE_RHS[nrn, nd] -= NODE_B[nrn, nd] * NODE_RHS[nrn, pnd]
		NODE_RHS[nrn, nd] /= NODE_D[nrn, nd]

def triang(nrn):
	"""
	void triang(NrnThread* _nt)
	"""
	nd = i3 - 1
	while nd >= i2:
		pnd = nd - 1
		ppp = NODE_A[nrn, nd] / NODE_D[nrn, nd]
		NODE_D[nrn, pnd] -= ppp * NODE_B[nrn, nd]
		NODE_RHS[nrn, pnd] -= ppp * NODE_RHS[nrn, nd]
		nd -= 1

def nrn_solve(nrn):
	"""
	void nrn_solve(NrnThread* _nt)
	"""
	triang(nrn)
	bksub(nrn)

def setup_tree_matrix(nrn):
	"""
	void setup_tree_matrix(NrnThread* _nt)
	"""
	nrn_rhs(nrn)
	nrn_lhs(nrn)

def update(nrn):
	"""
	void update(NrnThread* _nt)
	"""
	for nd in range(i1, _nt_end):
		Vm[nrn, nd] += NODE_RHS[nrn, nd]

	# save data like in NEURON (after .mod nrn_cur)
	if nrn == save_neuron_id:
		save_data(to_end=True)

def deliver_net_events():
	"""
	void deliver_net_events(NrnThread* nt)
	"""
	for index, pre_nrn in enumerate(syn_pre_nrn):
		if has_spike[pre_nrn] and syn_delay_timer[index] == -1:
			syn_delay_timer[index] = syn_delay[index] + 1
		if syn_delay_timer[index] == 0:
			post_id = syn_post_nrn[index]
			weight = syn_weight[index]
			if weight >= 0:
				g_exc[post_id] += weight
			else:
				g_inh[post_id] += -weight
			syn_delay_timer[index] = -1
		if syn_delay_timer[index] > 0:
			syn_delay_timer[index] -= 1
	# reset spikes
	has_spike[:] = False

def nrn_deliver_events(nrn, t):
	"""
	void nrn_deliver_events(NrnThread* nt)
	"""
	if spike_on[nrn] == 0 and Vm[nrn, 1] > V_th:
		spike_on[nrn] = True
		has_spike[nrn] = True
	elif Vm[nrn, 1] < V_th:
		spike_on[nrn] = False

def nrn_fixed_step_lastpart(nrn, t):
	"""
	void *nrn_fixed_step_lastpart(NrnThread *nth)
	"""
	recalc_synaptic(nrn)
	for seg in segs:
		if nrn_models[nrn] == 'inter':
			recalc_muslce_channels(nrn, seg, Vm[nrn, seg])
		elif nrn_models[nrn] == 'moto':
			recalc_moto_channels(nrn, seg, Vm[nrn, seg])
		elif nrn_models[nrn] == 'muscle':
			recalc_muslce_channels(nrn, seg, Vm[nrn, seg])
		else:
			raise Exception("No model")
	nrn_deliver_events(nrn, t)

def nrn_area_ri():
	"""
	void nrn_area_ri(Section *sec) [790] treeset.c
	area for right circular cylinders. Ri as right half of parent + left half of this
	"""
	for nrn in nrns:
		# dx = section_length(sec) / ((double) (sec->nnode - 1));
		dx = length[nrn] / nodecount # divide by the last index of node (or segments count)
		rright = 0
		# todo sec->pnode needs +1 index
		for nd in range(0 + 1, nnode - 1 + 1):
			# area for right circular cylinders. Ri as right half of parent + left half of this
			NODE_AREA[nrn, nd] = np.pi * dx * diam[nrn]
			rleft = 1e-2 * Ra[nrn] * (dx / 2) / (np.pi * diam[nrn] * diam[nrn] / 4) # left half segment Megohms
			NODE_RINV[nrn, nd] = 1 / (rleft + rright) # uS
			rright = rleft
		nd = nnode - 1 + 1
		# last segment has 0 length. area is 1e2 in dimensionless units
		NODE_AREA[:, 0] = 100
		NODE_AREA[:, nd] = 100
		NODE_RINV[nrn, nd] = 1 / rright

def connection_coef():
	"""
	void connection_coef(void)  [854] treeset.c
	"""
	nrn_area_ri()
	# set NODE_A and NODE_B
	# NODE_A is the effect of this node on the parent node's equation
	# NODE_B is the effect of the parent node on this node's equation
	for nrn in nrns:
		# first the effect of node on parent equation. Note that last nodes have area = 1.e2 in dimensionless
		# units so that last nodes have units of microsiemens
		#todo sec->pnode needs +1 index
		nd = 1
		# sec->prop->dparam[4].val = 1, what is dparam[4].val
		NODE_A[nrn, nd] = -1.e2 * 1 * NODE_RINV[nrn, nd] / NODE_AREA[nrn, nd - 1]
		# todo sec->pnode needs +1 index
		for nd in range(1 + 1, nnode + 1):
			pnd = nd - 1
			NODE_A[nrn, nd] = -1.e2 * NODE_RINV[nrn, nd] / NODE_AREA[nrn, pnd]
		# now the effect of parent on node equation
		# todo sec->pnode needs +1 index
		for nd in range(0 + 1, nnode + 1):
			NODE_B[nrn, nd] = -1.e2 * NODE_RINV[nrn, nd] / NODE_AREA[nrn, nd]

def finitialize(v_init=-70):
	"""

	"""
	# todo do not invoke for generators
	connection_coef()
	# for different models -- different init function
	for nrn in nrns:
		# do not init neuron state for generator
		if nrn_models[nrn] == 'generator':
			continue
		# for each segment init the neuron model
		for seg in segs:
			Vm[nrn, seg] = v_init
			if nrn_models[nrn] == 'inter':
				nrn_inter_initial(nrn, seg, v_init)
			elif nrn_models[nrn] == 'moto':
				nrn_moto_initial(nrn, seg, v_init)
			elif nrn_models[nrn] == 'muscle':
				nrn_muslce_initial(nrn, seg, v_init)
			else:
				raise Exception("No nrn model found")
		# init RHS/LHS
		setup_tree_matrix(nrn)
	# initialization process should not be recorderd
	GRAS_data.clear()

def nrn_fixed_step_thread(t):
	"""
	void *nrn_fixed_step_thread(NrnThread *nth)
	"""
	# update data for each neuron
	deliver_net_events()
	for nrn in nrns:
		if nrn_models[nrn] == 'generator':
			has_spike[nrn] = t in stimulus
		else:
			print(f"INIT before setup_tree_matrix nrn {nrn} NODE_RHS: {NODE_RHS[nrn, :]}")
			setup_tree_matrix(nrn)
			print(f"after setup_tree_matrix nrn {nrn} NODE_RHS: {NODE_RHS[nrn, :]}")
			nrn_solve(nrn)
			print(f"after nrn_solve nrn {nrn}")
			print(f"NODE_RHS: {NODE_RHS[nrn, :]}")
			print(f"NODE_D: {NODE_D[nrn, :]}")
			update(nrn)
			nrn_fixed_step_lastpart(nrn, t)
		# save spikes
		if has_spike[nrn]:
			spikes.append(t)

def simulation():
	"""
	Notes: NrnThread represent collection of cells or part of a cell computed by single thread within NEURON process
	"""
	# create_nrns()
	finitialize()
	# start simulation loop
	for t in range(sim_time_steps):
		print("=======", t * dt)
		nrn_fixed_step_thread(t)

def plot(gras_data, neuron_data):
	"""

	"""

	# names = 'cai il ina ik ica Eca m h n p mc hc A0 B0 D0 RINV0 Vm0 RHS0 A1 B1 D1 RINV1 Vm1 RHS1 A2 B2 D2 RINV2 Vm2 RHS2'
	names = 'il, ina, ik, m, h, n, l, s, v'
	rows = 5
	cols = 6

	plt.close()
	fig, ax = plt.subplots(rows, cols, sharex='all')
	xticks = np.arange(neuron_data.shape[0]) * dt
	# plot NEURON and GRAS
	for index, (neuron_d, gras_d, name) in enumerate(zip(neuron_data.T, gras_data.T, names.split())):
		row = int(index / cols)
		col = index % cols
		if row >= rows:
			break
		# if all(np.abs(neuron_d - gras_d) <= 1e-5):
		# 	ax[row, col].plot(xticks, neuron_d, label='NEURON', lw=3, ls='--')
		# else:
		ax[row, col].plot(xticks, neuron_d, label='NEURON', lw=3)
		ax[row, col].plot(xticks, gras_d, label='GRAS', lw=1, color='r')
		ax[row, col].plot(np.array(spikes) * dt, [np.mean(neuron_d)] * len(spikes), '.', ms=10, color='orange')
		# ax[row, col].plot(spikes, [np.mean(neuron_data)] * len(spikes), '.', color='r', ms=10)
		# ax[row, col].plot(spikes, [np.mean(gras_data)] * len(spikes), '.', color='b', ms=5)
		passed = np.abs(np.mean(neuron_d) / np.mean(gras_d)) * 100 - 100
		ax[row, col].set_title(f"{name} max diff {passed:.3f}")
		ax[row, col].set_xlim(0, sim_time)
	plt.show()


if __name__ == "__main__":
	simulation()
	GRAS_data = np.array(GRAS_data)
	xlength = GRAS_data.shape[0]
	NEURON_data = get_neuron_data()[:xlength, :]
	log.info(f"GRAS shape {GRAS_data.shape}")
	log.info(f"NEURON shape {NEURON_data.shape}")
	plot(GRAS_data, NEURON_data)
