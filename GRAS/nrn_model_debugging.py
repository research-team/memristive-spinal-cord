"""
Formulas and value units were taken from:

Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011).
Principles of Computational Modelling in Neuroscience. Cambridge: Cambridge University Press.
DOI:10.1017/CBO9780511975899

Based on the NEURON repository
"""
import random
import numpy as np
import logging as log
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)

EXTRACELLULAR = True
DEBUG = False
dt = 0.025  # [ms] - sim step
sim_time = 50
sim_time_steps = int(sim_time / dt)
save_neuron_id = 1

if DEBUG:
	stimulus = (np.arange(10, sim_time, 25) / dt).astype(int)
else:
	stimulus = (np.arange(10, sim_time, 10) / dt).astype(int)

"""common const"""
V_th = -40
V_adj = -63         # [mV] adjust voltage for -55 threshold
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

# i_membrane [mA/cm2]
e_extracellular = 0 # [mV]
xraxial = 1e9       # [MOhm/cm]
# todo find the initialization
xg = [0, 1e9, 1e9, 1e9, 0]  # [S/cm2]
xc = [0, 0, 0, 0, 0]        # [uF/cm2]

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
tmp = 205
# common properties
models = init0(tmp, np.object) # model's names
Cm = init0(tmp)         # [uF / cm2] membrane capacity
gnabar = init0(tmp)     # [S / cm2] todo
gkbar = init0(tmp)      # [S / cm2]g_K todo
gl = init0(tmp)         # [S / cm2] todo
Ra = init0(tmp)         # [Ohm cm]
diam = init0(tmp)       # [um] diameter
length = init0(tmp)     # [um] compartment length
ena = init0(tmp)        # [mV] todo
ek = init0(tmp)         # [mV] todo
el = init0(tmp)         # [mV] todo
# moto properties
gkrect = init0(tmp)     # [S / cm2] todo
gcaN = init0(tmp)       # [S / cm2] todo
gcaL = init0(tmp)       # [S / cm2] todo
gcak = init0(tmp)       # [S / cm2] todo
# synapses data
E_ex = init0(tmp)       # [S / cm2] todo
E_inh = init0(tmp)      # [S / cm2] todo
syn_pre_nrn = []
syn_post_nrn = []
syn_weight = []
syn_delay = []
syn_delay_timer = []
tau_syn_exc = init0(tmp)    # [S / cm2] todo
tau_syn_inh = init0(tmp)    # [S / cm2] todo

def create(number, model='inter'):
	"""

	"""
	global nrns_number
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
	__e_ex = None
	__e_inh = None
	__tau_syn_exc = None
	__tau_syn_inh = None
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
			__e_ex = 50
			__e_inh = -80
			__tau_syn_exc = 0.3
			__tau_syn_inh = 2.0
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
			__e_ex = 50
			__e_inh = -80
			__tau_syn_exc = 0.3
			__tau_syn_inh = 2.0
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
			__e_ex = 0  # todo check -- 9
			__e_inh = -80
			__tau_syn_exc = 0.3
			__tau_syn_inh = 0.1
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
		E_ex[nrnid] = __e_ex
		E_inh[nrnid] = __e_inh
		tau_syn_exc[nrnid] = __tau_syn_exc
		tau_syn_inh[nrnid] = __tau_syn_inh
	models[nrns_number:nrns_number + number] = model
	nrns_number += number
	return ids

def conn_a2a(pre_nrns, post_nrns, delay, weight):
	"""

	"""
	for pre in pre_nrns:
		for post in post_nrns:
			syn_pre_nrn.append(pre)
			syn_post_nrn.append(post)
			syn_weight.append(weight)
			syn_delay.append(int(delay / dt))
			syn_delay_timer.append(-1)


def conn_fixed_outdegree(pre_nrns, post_nrns, delay, weight, outdegree=50):
	"""

	"""
	for post in post_nrns:
		for j in range(outdegree):
			pre = random.choice(pre_nrns)
			weight = weight #random.gauss(weight, weight / 10)
			delay = delay #random.gauss(delay, delay / 10)

			syn_pre_nrn.append(pre)
			syn_post_nrn.append(post)
			syn_weight.append(weight)
			syn_delay.append(int(delay / dt))
			syn_delay_timer.append(-1)

if DEBUG:
	gen = create(1, model='generator')
	m1 = create(1, model='muscle')
	conn_a2a(gen, m1, delay=1, weight=40.5)
	# m1 = create(1, model='moto')
	# connect(gen, m1, delay=1, weight=5.5, conn_type='all-to-all')
else:
	gen = create(1, model='generator')
	OM1 = create(50, model='inter')
	OM2 = create(50, model='inter')
	OM3 = create(50, model='inter')
	moto = create(50, model='moto')
	muscle = create(1, model='muscle')

	conn_a2a(gen, OM1, delay=1, weight=1.5)

	conn_fixed_outdegree(OM1, OM2, delay=2, weight=1.85)
	conn_fixed_outdegree(OM2, OM1, delay=3, weight=1.85)
	conn_fixed_outdegree(OM2, OM3, delay=3, weight=0.00055)
	conn_fixed_outdegree(OM1, OM3, delay=3, weight=0.00005)
	conn_fixed_outdegree(OM3, OM2, delay=1, weight=2.5)
	conn_fixed_outdegree(OM3, OM1, delay=1, weight=2.5)
	conn_fixed_outdegree(OM2, moto, delay=2, weight=1.5)
	conn_fixed_outdegree(moto, muscle, delay=2, weight=15.5)

nrns = list(range(nrns_number))
nrn_shape = (nrns_number, _nt_end)

# global variables
Vm = init0(nrn_shape)           # [mV] array for three compartments volatge
n = init0(nrn_shape)            # [0..1] compartments channel
m = init0(nrn_shape)            # [0..1] compartments channel
h = init0(nrn_shape)            # [0..1] compartments channel
l = init0(nrn_shape)            # [0..1] inward rectifier potassium (Kir) channel
s = init0(nrn_shape)            # [0..1] nodal slow potassium channel
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
spike_on = init0(nrns_number, dtype=bool)   # special flag to prevent fake spike detecting

nlayer = 2
ext_shape = (nrns_number, _nt_end, nlayer)
ext_rhs = init0(ext_shape)   # extracellular
ext_v = init0(ext_shape)     # extracellular
ext_a = init0(ext_shape)     # extracellular
ext_b = init0(ext_shape)     # extracellular
ext_d = init0(ext_shape)     # extracellular

spikes = []
GRAS_data1 = []
GRAS_data2 = []

def get_neuron_data():
	with open("/home/alex/NRNTEST/muscle/output") as file:
		neuron_data = []
		while 1:
			line = file.readline()
			if not line:
				break
			if 'SYNAPTIC time' in line:
				line = file.readline()
				line = line[line.index('i:')+2:line.index(', e')]
				neuron_data.append([line])
			if 'BREAKPOINT currents' in line:
				# il, ina, ik, m, h, n, v
				file.readline()
				line = file.readline()
				line = line.replace('BREAKPOINT currents ', '').strip().split("\t")[:-1] # without Vm
				[file.readline() for _ in range(3)]
				neuron_data[-1] += line
			if 'A	B	D	INV	Vm	RHS' in line:
				file.readline()
				file.readline()
				line = file.readline()
				line = line.strip().split("\t")
				neuron_data[-1] += line

		# neuron_data.append(line)
		neuron_data = neuron_data[1:-1]
		neuron_data.insert(0, neuron_data[0])
		neuron_data.insert(0, neuron_data[0])
		neuron_data = np.array(neuron_data).astype(np.float)
	return neuron_data

def save_data(to_end=False):
	"""

	"""
	inrn = save_neuron_id
	MID = 2
	global GRAS_data

	if to_end:
		GRAS_data1.append([NODE_A[inrn, MID], NODE_B[inrn, MID], NODE_D[inrn, MID],
		                  NODE_RINV[inrn, MID], Vm[inrn, MID], NODE_RHS[inrn, MID], ext_v[inrn, MID, 0]])
	else:
		# syn il, ina, ik, m, h, n, l, s, v
		isyn = g_exc[inrn] * (Vm[inrn, MID] - E_ex[inrn])
		GRAS_data2.append([isyn, I_L[inrn, MID], I_Na[inrn, MID], I_K[inrn, MID], m[inrn, MID], h[inrn, MID],
		                  n[inrn, MID], l[inrn, MID], s[inrn, MID]])

def Exp(volt):
	return 0 if volt < -100 else np.exp(volt)

def alpham(volt):
	"""

	"""
	if abs((volt + amB) / amC) < 1e-6:
		return amA * amC
	return amA * (volt + amB) / (1 - Exp(-(volt + amB) / amC))

def betam(volt):
	"""

	"""
	if abs((volt + bmB) / bmC) < 1e-6:
		return -bmA * bmC
	return -bmA * (volt + bmB) / (1 - Exp((volt + bmB) / bmC))

def syn_current(nrn, voltage):
	"""

	"""
	return g_exc[nrn] * (voltage - E_ex[nrn]) + g_inh[nrn] * (voltage - E_inh[nrn])

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
	# exc synaptic conductance
	if g_exc[nrn] != 0:
		g_exc[nrn] -= g_exc[nrn] * (1 - np.exp(-dt / tau_syn_exc[nrn]))
		if g_exc[nrn] < 1e-10:
			g_exc[nrn] = 0
	# inh synaptic conductance
	if g_inh[nrn] != 0:
		g_inh[nrn] -= g_inh[nrn] * (1 - np.exp(-dt / tau_syn_inh[nrn]))
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
	# initial
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
	# initial
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
	# initial
	m[nrn, seg] = m_inf
	h[nrn, seg] = h_inf
	n[nrn, seg] = n_inf

def recalc_moto_channels(nrn, seg, V):
	"""
	evaluate_fct
	"""
	# BREAKPOINT -> states -> evaluate_fct
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
	# states
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
	evaluate_fct
	"""
	# BREAKPOINT -> states -> evaluate_fct
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
	linf = 1.0 / (1.0 + np.exp((V - vhalfl) / kl))  # l_steadystate
	taul = 1.0 / (qt * (at * np.exp(-V / vhalft) + bt * np.exp(V / vhalft)))
	alpha = 0.3 / (1.0 + np.exp((V + 43.0) / - 5.0))
	beta = 0.03 / (1.0 + np.exp((V + 80.0) / - 1.0))
	summ = alpha + beta
	stau = 1.0 / summ
	sinf = alpha / summ
	# states
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

def nrn_rhs_ext(nrn):
	"""
	void nrn_rhs_ext(NrnThread* _nt)
	"""
	# nd rhs contains -membrane current + stim current
	# nde rhs contains stim current
	# todo passed
	for nd in segs:
		ext_rhs[nrn, nd, 0] -= NODE_RHS[nrn, nd]
	#
	for nd in range(1, _nt_end):
		pnd = nd - 1
		# for j in range(nlayer):
		# 	# dv = 0
		# 	dv = ext_v[nrn, pnd, j] - ext_v[nrn, nd, j]
		# 	ext_rhs[nrn, nd, j] -= ext_b[nrn, nd, j] * dv
		# 	ext_rhs[nrn, pnd, j] += ext_a[nrn, nd, j] * dv
		# series resistance and battery to ground between nlayer-1 and ground
		j = nlayer - 1
		# print(f"??V0 {ext_v[nrn, nd, 0]} V1 {ext_v[nrn, nd, 1]} RHS0 {ext_rhs[nrn, nd, 0]} RHS1 {ext_rhs[nrn, nd, 1]}")

		ext_rhs[nrn, nd, j] -= xg[nd] * (ext_v[nrn, nd, j] - e_extracellular)
		# for (--j; j >= 0; --j) { // between j and j+1 layer
		j = 0
		# print(f"V0 {ext_v[nrn, nd, 0]} V1 {ext_v[nrn, nd, 1]} RHS0 {ext_rhs[nrn, nd, 0]} RHS1 {ext_rhs[nrn, nd, 1]}")
		x = xg[nd] * (ext_v[nrn, nd, j] - ext_v[nrn, nd, j+1])
		ext_rhs[nrn, nd, j] -= x
		ext_rhs[nrn, nd, j+1] += x

		# print(f"==>V0 {ext_v[nrn, nd, 0]} V1 {ext_v[nrn, nd, 1]} RHS0 {ext_rhs[nrn, nd, 0]} RHS1 {ext_rhs[nrn, nd, 1]}")
	# print()

def nrn_setup_ext(nrn):
	"""
	void nrn_setup_ext(NrnThread* _nt)
	"""
	cj = 1 / dt
	cfac = 0.001 * cj

	# todo find the place where it is zeroed
	ext_d[nrn, :, :] = 0

	# d contains all the membrane conductances (and capacitance)
	# i.e. (cm/dt + di/dvm - dis/dvi)*[dvi] and (dis/dvi)*[dvx]
	for nd in segs:
		# nde->_d only has -ELECTRODE_CURRENT contribution
		ext_d[nrn, nd, 0] += NODE_D[nrn, nd]
	# D[0] = [0 0.1442 0.1442 0.1442 0 ]

	# series resistance, capacitance, and axial terms
	for nd in range(1, _nt_end):
		pnd = nd - 1
		# series resistance and capacitance to ground
		j = 0
		while True:
			mfac = xg[nd] + xc[nd] * cfac
			ext_d[nrn, nd, j] += mfac
			j += 1
			if j == nlayer:
				break
			ext_d[nrn, nd, j] += mfac
		# axial connections
		for j in range(nlayer):
			ext_d[nrn, nd, j] -= ext_b[nrn, nd, j]
			ext_d[nrn, pnd, j] -= ext_a[nrn, nd, j]
	# D[0] = [2e-08 1e+09 1e+09 1e+09 2e-08 ]
	# D[1] = [2e-08 2e+09 2e+09 2e+09 2e-08 ]

def nrn_update_2d(nrn):
	"""
	void nrn_update_2d(NrnThread* nt)

	update has already been called so modify nd->v based on dvi we only need to
	update extracellular nodes and base the corresponding nd->v on dvm (dvm = dvi - dvx)
	"""
	for nd in range(i1, _nt_end):
		for j in range(nlayer):
			ext_v[nrn, nd, j] += ext_rhs[nrn, nd, j]

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
	#fixme
	ext_rhs[nrn, :, :] = 0

	# update MOD rhs, CAPS has no current [CAP MOD CAP]!
	for seg in segs:
		if seg == segs[0] or seg == segs[-1]:
			continue
		V = Vm[nrn, seg]
		# SYNAPTIC update
		# todo 2nd seg only for muscles [0, 1, 2MID, 3, 4]
		seg_update = 2 if models[nrn] == 'muscle' else 1
		if seg == seg_update:
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
		if models[nrn] == 'inter':
			# muscle and inter has the same fast_channel function
			_g = nrn_muscle_current(nrn, seg, V + 0.001)
			_rhs = nrn_muscle_current(nrn, seg, V)
		elif models[nrn] == 'moto':
			_g = nrn_moto_current(nrn, seg, V + 0.001)
			_rhs = nrn_moto_current(nrn, seg, V)
		elif models[nrn] == 'muscle':
			# muscle and inter has the same fast_channel function
			_g = nrn_muscle_current(nrn, seg, V + 0.001)
			_rhs = nrn_muscle_current(nrn, seg, V)
		else:
			raise Exception('No nrn model found')
		# save data like in NEURON (after .mod nrn_cur)
		if nrn == save_neuron_id and seg == 1:
			save_data()
		_g = (_g - _rhs) / 0.001

		NODE_RHS[nrn, seg] -= _rhs
		# static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
		NODE_D[nrn, seg] += _g
	# end FOR segments

	# activsynapse_rhs()
	if EXTRACELLULAR:
		# Cannot have any axial terms yet so that i(vm) can be calculated from
		# i(vm)+is(vi) and is(vi) which are stored in rhs vector.
		nrn_rhs_ext(nrn)
		# nrn_rhs_ext has also computed the the internal axial current for those
		# nodes containing the extracellular mechanism
	# activstim_rhs()
	# activclamp_rhs()

	# todo always 0, because Vm0 = Vm1 = Vm2 at [CAP node CAP] model (1 section)
	for nd in range(i2, i3):
		pnd = nd - 1
		dv = Vm[nrn, pnd] - Vm[nrn, nd]
		# our connection coefficients are negative so
		NODE_RHS[nrn, nd] -= NODE_B[nrn, nd] * dv
		NODE_RHS[nrn, pnd] += NODE_A[nrn, nd] * dv

def nrn_lhs(nrn):
	"""
	void nrn_lhs(NrnThread *_nt)
	NODE_D[nrn, nd] updating is located at nrn_rhs, because _g is not the global variable
	"""
	# nt->cj = 2/dt if (secondorder) else 1/dt
	# note, the first is CAP
	# function nrn_cap_jacob(_nt, _nt->tml->ml);
	cj = 1 / dt
	cfac = 0.001 * cj
	# fixme +1 for nodelist
	for nd in range(0 + 1, nodecount + 1):
		NODE_D[nrn, nd] += cfac * Cm[nrn]

	# activsynapse_lhs()
	if EXTRACELLULAR:
		nrn_setup_ext(nrn)
	# activclamp_lhs();

	# updating NODED
	for nd in range(i2, i3):
		pnd = nd - 1
		NODE_D[nrn, nd] -= NODE_B[nrn, nd]
		NODE_D[nrn, pnd] -= NODE_A[nrn, nd]
		# extra
		# _a_matelm += NODE_A[nrn, nd]
		# _b_matelm += NODE_B[nrn, nd]

def bksub(nrn):
	"""
	void bksub(NrnThread* _nt)
	"""
	# intracellular
	for nd in range(i1, i2):
		NODE_RHS[nrn, nd] /= NODE_D[nrn, nd]
	for nd in range(i2, i3):
		pnd = nd - 1
		NODE_RHS[nrn, nd] -= NODE_B[nrn, nd] * NODE_RHS[nrn, pnd]
		NODE_RHS[nrn, nd] /= NODE_D[nrn, nd]

	# extracellular
	for nd in range(i1, i2):
		for j in range(nlayer):
			ext_rhs[nrn, nd, j] /= ext_d[nrn, nd, j]
	for nd in range(i2, i3):
		pnd = nd - 1
		for j in range(nlayer):
			ext_rhs[nrn, nd, j] -= ext_b[nrn, nd, j] * ext_rhs[nrn, pnd, j]
			ext_rhs[nrn, nd, j] /= ext_d[nrn, nd, j]

def triang(nrn):
	"""
	void triang(NrnThread* _nt)
	"""
	# intracellular
	nd = i3 - 1
	while nd >= i2:
		pnd = nd - 1
		ppp = NODE_A[nrn, nd] / NODE_D[nrn, nd]
		NODE_D[nrn, pnd] -= ppp * NODE_B[nrn, nd]
		NODE_RHS[nrn, pnd] -= ppp * NODE_RHS[nrn, nd]
		nd -= 1
	# extracellular
	nd = i3 - 1
	while nd >= i2:
		pnd = nd - 1
		for j in range(nlayer):
			ppp = ext_a[nrn, nd, j] / ext_d[nrn, nd, j]
			ext_d[nrn, pnd, j] -= ppp * ext_b[nrn, nd, j]
			ext_rhs[nrn, pnd, j] -= ppp * ext_rhs[nrn, nd, j]
		nd -= 1

def nrn_solve(nrn):
	"""
	void nrn_solve(NrnThread* _nt)
	"""
	# NODED [228.479 0.416927 0.326018 0.416927 228.479]
	# spFactor
	# print("NODED", NODE_D[1, :])
	# print("ext_d", ext_d[1, :, 0])

	# NODED [0.00437676 4.76737 3.06731 4.83803 0.00437676]

	triang(nrn)
	bksub(nrn)
	# print("NODED", NODE_D[1, :])
	# print("must  [0.004376 4.76737 3.06731 4.83803 0.004376]")
	# exit()


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
	nrn_update_2d(nrn)

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

def nrn_deliver_events(nrn):
	"""
	void nrn_deliver_events(NrnThread* nt)
	"""
	if not spike_on[nrn] and Vm[nrn, 1] > V_th:
		spike_on[nrn] = True
		has_spike[nrn] = True
	elif Vm[nrn, 1] < V_th:
		spike_on[nrn] = False

def nrn_fixed_step_lastpart(nrn):
	"""
	void *nrn_fixed_step_lastpart(NrnThread *nth)
	"""
	recalc_synaptic(nrn)
	for seg in segs:
		if models[nrn] == 'inter':
			recalc_muslce_channels(nrn, seg, Vm[nrn, seg])
		elif models[nrn] == 'moto':
			recalc_moto_channels(nrn, seg, Vm[nrn, seg])
		elif models[nrn] == 'muscle':
			recalc_muslce_channels(nrn, seg, Vm[nrn, seg])
		else:
			raise Exception("No model")
	nrn_deliver_events(nrn)

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

def ext_con_coef():
	"""
	void ext_con_coef(void)
	setup a and b
	"""
	layer = 0
	# todo: extracellular only for those neurons who need
	for nrn in nrns:
		# temporarily store half segment resistances in rhs
		# todo sec->pnode needs +1 index, also xraxial is common
		for nd in range(0 + 1, nnode - 1 + 1):
			dx = length[nrn] / nodecount
			ext_rhs[nrn, nd, layer] = 1e-4 * xraxial * dx / 2  # Megohms
		# last segment has 0 length
		ext_rhs[nrn, -1, layer] = 0
		# NEURON RHS = [5e+07 5e+07 5e+07 0 ]

		# node half resistances in general get added to the node and to the node's "child node in the same section".
		# child nodes in different sections don't involve parent node's resistance
		ext_b[nrn, 0+1, layer] = ext_rhs[nrn, 0+1, layer]
		# todo sec->pnode needs +1 index
		for nd in range(1 + 1, nnode + 1):
			pnd = nd - 1
			ext_b[nrn, nd, layer] = ext_rhs[nrn, nd, layer] + ext_rhs[nrn, pnd, layer]  # Megohms
		# NEURON B = [5e+07 1e+08 1e+08 5e+07 ]

		# first the effect of node on parent equation. Note That last nodes have area = 1.e2 in
		# dimensionless units so that last nodes have units of microsiemens's
		area = NODE_AREA[nrn, 0]    # parentnode index of sec is 0
		rall_branch = 1  # sec->prop->dparam[4].val
		ext_a[nrn, 0+1, layer] = -1.e2 * rall_branch / (ext_b[nrn, 0+1, layer] * area)
		# todo sec->pnode needs +1 index
		for nd in range(1 + 1, nnode + 1):
			area = NODE_AREA[nrn, nd - 1]  # pnd = nd - 1
			ext_a[nrn, nd, layer] = -1.e2 / (ext_b[nrn, nd, layer] * area)
		# NEURON A = [-2e-08 -7.95775e-12 -7.95775e-12 -1.59155e-11 ]

		# now the effect of parent on node equation
		# todo sec->pnode needs +1 index
		for nd in range(0 + 1, nnode + 1):
			ext_b[nrn, nd, layer] = -1.e2 / (ext_b[nrn, nd, layer] * NODE_AREA[nrn, nd])
		# NEURON B = [-1.59155e-11 -7.95775e-12 -7.95775e-12 -2e-08 ]

		# the same for other layers
		ext_a[nrn, :, 1] = ext_a[nrn, :, 0].copy()
		ext_b[nrn, :, 1] = ext_b[nrn, :, 0].copy()
		ext_rhs[nrn, :, 1] = ext_rhs[nrn, :, 0].copy()
		# todo recheck: RHS initially is zero!
		ext_rhs[nrn, :, :] = 0

def connection_coef():
	"""
	void connection_coef(void) treeset.c
	"""
	nrn_area_ri()
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
	# for extracellular
	ext_con_coef()

def finitialize(v_init=-70):
	"""

	"""
	# todo do not invoke for generators
	connection_coef()
	# for different models -- different init function
	for nrn in nrns:
		# do not init neuron state for generator
		if models[nrn] == 'generator':
			continue
		# for each segment init the neuron model
		for seg in segs:
			Vm[nrn, seg] = v_init
			if models[nrn] == 'inter':
				nrn_inter_initial(nrn, seg, v_init)
			elif models[nrn] == 'moto':
				nrn_moto_initial(nrn, seg, v_init)
			elif models[nrn] == 'muscle':
				nrn_muslce_initial(nrn, seg, v_init)
			else:
				raise Exception("No nrn model found")
		# init RHS/LHS
		setup_tree_matrix(nrn)
	# initialization process should not be recorderd
	GRAS_data1.clear()
	GRAS_data2.clear()

def nrn_fixed_step_thread(t):
	"""
	void *nrn_fixed_step_thread(NrnThread *nth)
	"""
	# update data for each neuron
	deliver_net_events()
	for nrn in nrns:
		if models[nrn] == 'generator':
			has_spike[nrn] = t in stimulus
		else:
			setup_tree_matrix(nrn)
			nrn_solve(nrn)
			update(nrn)
			nrn_fixed_step_lastpart(nrn)
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
		print(t * dt)
		nrn_fixed_step_thread(t)

def plot(gras_data, neuron_data):
	"""

	"""
	# names = 'cai il ina ik ica Eca m h n p mc hc A0 B0 D0 RINV0 Vm0 RHS0 A1 B1 D1 RINV1 Vm1 RHS1 A2 B2 D2 RINV2 Vm2 RHS2'
	names = 'isyn il ina ik m h n l s A B D RINV Vm RHS EXT'
	rows = 4
	cols = 5

	plt.close()
	fig, ax = plt.subplots(rows, cols, sharex='all')
	xticks = np.arange(gras_data.shape[0]) * dt
	# plot NEURON and GRAS
	for index, (neuron_d, gras_d, name) in enumerate(zip(neuron_data.T, gras_data.T, names.split())):
		row = int(index / cols)
		col = index % cols
		if row >= rows:
			break
		# if all(np.abs(neuron_d - gras_d) <= 1e-3):
		# 	ax[row, col].plot(xticks, neuron_d, label='NEURON', lw=3, ls='--')
		# else:
		ax[row, col].plot(xticks, neuron_d, label='NEURON', lw=3)
		# else:
		ax[row, col].plot(xticks, gras_d, label='GRAS', lw=1, color='r')
		# ax[row, col].plot(np.array(spikes) * dt, [np.mean(neuron_d)] * len(spikes), '.', ms=10, color='orange')
		# ax[row, col].plot(spikes, [np.mean(neuron_data)] * len(spikes), '.', color='r', ms=10)
		# ax[row, col].plot(spikes, [np.mean(gras_data)] * len(spikes), '.', color='b', ms=5)
		# passed = np.abs(np.mean(neuron_d) / np.mean(gras_d)) * 100 - 100
		ax[row, col].set_title(f"{name} max diff")
		ax[row, col].set_xlim(0, sim_time)
	plt.show()

if __name__ == "__main__":
	simulation()
	GRAS_data = np.array(list(sum(d, []) for d in zip(GRAS_data2, GRAS_data1)))
	if DEBUG:
		xlength = GRAS_data.shape[0]
		NEURON_data = get_neuron_data()[:xlength, :]
		log.info(f"GRAS shape {GRAS_data.shape}")
		log.info(f"NEURON shape {NEURON_data.shape}")
		plot(GRAS_data, NEURON_data)
	else:
		pass
