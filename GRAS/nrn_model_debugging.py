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
from time import time

log.basicConfig(level=log.INFO)

DEBUG = False
EXTRACELLULAR = False
GENERATOR = 'generator'
INTER = 'interneuron'
MOTO = 'motoneuron'
MUSCLE = 'muscle'
neurons_in_ip = 196

dt = 0.025      # [ms] - sim step
sim_time = 50   # [ms] - simulation time
sim_time_steps = int(sim_time / dt) # [steps] converted time into steps

skin_time = 25  # duration of layer 25 = 21 cm/s; 50 = 15 cm/s; 125 = 6 cm/s
cv_fr = 200     # frequency of CV
ees_fr = 100     # frequency of EES

cv_int = 1000 / cv_fr
ees_int = 1000 / ees_fr

EES_stimulus = (np.arange(0, sim_time, ees_int) / dt).astype(int)
CV1_stimulus = (np.arange(skin_time * 0, skin_time * 1, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV2_stimulus = (np.arange(skin_time * 1, skin_time * 2, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV3_stimulus = (np.arange(skin_time * 2, skin_time * 3, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV4_stimulus = (np.arange(skin_time * 3, skin_time * 5, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV5_stimulus = (np.arange(skin_time * 5, skin_time * 6, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)

# common neuron constants
k = 0.017           # synaptic coef
V_th = -40          # [mV] voltage threshold
V_adj = -63         # [mV] adjust voltage for -55 threshold
# moto neuron constants
ca0 = 2             # initial calcium concentration
amA = 0.4           # const ??? todo
amB = 66            # const ??? todo
amC = 5             # const ??? todo
bmA = 0.4           # const ??? todo
bmB = 32            # const ??? todo
bmC = 5             # const ??? todo
R_const = 8.314472  # [k-mole] or [joule/degC] const
F_const = 96485.34  # [faraday] or [kilocoulombs] const
# muscle fiber constants
g_kno = 0.01        # [S/cm2] conductance of the todo
g_kir = 0.03        # [S/cm2] conductance of the Inwardly Rectifying Potassium K+ (Kir) channel
# Boltzman steady state curve
vhalfl = -98.92     # [mV] inactivation half-potential
kl = 10.89          # [mV] Stegen et al. 2012
# tau_infty
vhalft = 67.0828    # [mV] fitted #100 uM sens curr 350a, Stegen et al. 2012
at = 0.00610779     # [/ ms] Stegen et al. 2012
bt = 0.0817741      # [/ ms] Note: typo in Stegen et al. 2012
# temperature dependence
q10 = 1             # temperature scaling (sensitivity)
celsius = 36        # [degC] temperature of the cell
# i_membrane [mA/cm2]
e_extracellular = 0 # [mV]
xraxial = 1e9       # [MOhm/cm]
# todo find the initialization
xg = [0, 1e9, 1e9, 1e9, 0]  # [S/cm2]
xc = [0, 0, 0, 0, 0]        # [uF/cm2]

def init0(shape, dtype=float):
	return np.zeros(shape, dtype=dtype)

nrns_number = 0
nrns_and_segs = 0
generators_id_end = 0
# common neuron's parameters
# also from https://www.cell.com/neuron/pdfExtended/S0896-6273(16)00010-6
class P:
	nrn_start_seg = []     #
	models = []     # [str] model's names
	Cm = []         # [uF / cm2] membrane capacitance
	gnabar = []     # [S / cm2] the maximal fast Na+ conductance
	gkbar = []      # [S / cm2] the maximal slow K+ conductance
	gl = []         # [S / cm2] the maximal leak conductance
	Ra = []         # [Ohm cm] axoplasmic resistivity
	diam = []       # [um] soma compartment diameter
	length = []     # [um] soma compartment length
	ena = []        # [mV] Na+ reversal (equilibrium, Nernst) potential
	ek = []         # [mV] K+ reversal (equilibrium, Nernst) potential
	el = []         # [mV] Leakage reversal (equilibrium) potential
	# moto neuron's properties
	# https://senselab.med.yale.edu/modeldb/showModel.cshtml?model=189786
	# https://journals.physiology.org/doi/pdf/10.1152/jn.2002.88.4.1592
	gkrect = []     # [S / cm2] the maximal delayed rectifier K+ conductance
	gcaN = []       # [S / cm2] the maximal N-type Ca2+ conductance
	gcaL = []       # [S / cm2] the maximal L-type Ca2+ conductance
	gcak = []       # [S / cm2] the maximal Ca2+ activated K+ conductance
	# synapses' parameters
	E_ex = []       # [mV] excitatory reversal (equilibrium) potential
	E_inh = []      # [mV] inhibitory reversal (equilibrium) potential
	tau_exc = []    # [ms] rise time constant of excitatory synaptic conductance
	tau_inh1 = []   # [ms] rise time constant of inhibitory synaptic conductance
	tau_inh2 = []   # [ms] decay time constant of inhibitory synaptic conductance

# metadata of synapses
syn_pre_nrn = []        # [id] list of pre neurons ids
syn_post_nrn = []       # [id] list of pre neurons ids
syn_weight = []         # [S] list of synaptic weights
syn_delay = []          # [ms * dt] list of synaptic delays in steps
syn_delay_timer = []    # [ms * dt] list of synaptic timers, shows how much left to send signal
# arrays for saving data
spikes = []             # saved spikes
GRAS_data = []          # saved gras data (DEBUGGING)
save_groups = []        # neurons groups that need to save
saved_voltage = []      # saved voltage
save_neuron_ids = []    # neurons id that need to save

def form_group(name, number=50, model=INTER, segs=1):
	"""

	"""
	global nrns_number, nrns_and_segs
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
	__tau_exc = None
	__tau_inh1 = None
	__tau_inh2 = None
	# without random at first stage of debugging
	for _ in ids:
		if model == INTER:
			__Cm = random.gauss(1, 0.01)
			__gnabar = 0.1
			__gkbar = 0.08
			__gl = 0.002
			__Ra = 100
			__ena = 50
			__ek = -90
			__el = -70
			__diam = 10 # random.randint(5, 15)
			__dx = __diam
			__e_ex = 50
			__e_inh = -80
			__tau_exc = 0.35
			__tau_inh1 = 0.5
			__tau_inh2 = 3.5
		elif model == MOTO:
			__Cm = 2
			__gnabar = 0.05
			__gl = 0.002
			__Ra = 200
			__ena = 50
			__ek = -80
			__el = -70
			__diam = random.randint(45, 55)
			__dx = __diam
			__gkrect = 0.3
			__gcaN = 0.05
			__gcaL = 0.0001
			__gcak = 0.3
			__e_ex = 50
			__e_inh = -80
			__tau_exc = 0.3
			__tau_inh1 = 1
			__tau_inh2 = 1.5
			if __diam > 50:
				__gnabar = 0.1
				__gcaL = 0.001
				__gl = 0.003
				__gkrect = 0.2
				__gcak = 0.2
		elif model == MUSCLE:
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
			__e_ex = 0
			__e_inh = -80
			__tau_exc = 0.3
			__tau_inh1 = 1
			__tau_inh2 = 1
		elif model == GENERATOR:
			pass
		else:
			raise Exception("Choose the model")

		# common properties
		P.Cm.append(__Cm)
		P.gnabar.append(__gnabar)
		P.gkbar.append(__gkbar)
		P.gl.append(__gl)
		P.el.append(__el)
		P.ena.append(__ena)
		P.ek.append(__ek)
		P.Ra.append(__Ra)
		P.diam.append(__diam)
		P.length.append(__dx)
		P.gkrect.append(__gkrect)
		P.gcaN.append(__gcaN)
		P.gcaL.append(__gcaL)
		P.gcak.append(__gcak)
		P.E_ex.append(__e_ex)
		P.E_inh.append(__e_inh)
		P.tau_exc.append(__tau_exc)
		P.tau_inh1.append(__tau_inh1)
		P.tau_inh2.append(__tau_inh2)

		P.nrn_start_seg.append(nrns_and_segs)
		nrns_and_segs += (segs + 2)

	P.models += [model] * number
	nrns_number += number

	return name, ids

def conn_a2a(pre_nrns, post_nrns, delay, weight):
	"""

	"""
	pre_nrns_ids = pre_nrns[1]
	post_nrns_ids = post_nrns[1]

	for pre in pre_nrns_ids:
		for post in post_nrns_ids:
			# weight = random.gauss(weight, weight / 5)
			# delay = random.gauss(delay, delay / 5)
			syn_pre_nrn.append(pre)
			syn_post_nrn.append(post)
			syn_weight.append(weight)
			syn_delay.append(int(delay / dt))
			syn_delay_timer.append(-1)

def conn_fixed_outdegree(pre_group, post_group, delay, weight, indegree=50):
	"""

	"""
	pre_nrns_ids = pre_group[1]
	post_nrns_ids = post_group[1]

	nsyn = random.randint(indegree - 15, indegree)

	for post in post_nrns_ids:
		for _ in range(nsyn):
			pre = random.choice(pre_nrns_ids)
			# weight = random.gauss(weight, weight / 5)
			# delay = random.gauss(delay, delay / 5)
			syn_pre_nrn.append(pre)
			syn_post_nrn.append(post)
			syn_weight.append(weight)
			syn_delay.append(int(delay / dt))
			syn_delay_timer.append(-1)

def save(nrn_groups):
	global save_neuron_ids, save_groups
	save_groups = nrn_groups
	for group in nrn_groups:
		for nrn in group[1]:
			center = P.nrn_start_seg[nrn] + (2 if P.models[nrn] == MUSCLE else 1)
			save_neuron_ids.append(center)

if DEBUG:
	gen = form_group(1, model=GENERATOR)
	m1 = form_group(1, model=MUSCLE, segs=3)
	conn_a2a(gen, m1, delay=1, weight=40.5)
	groups = [m1]
	save(groups)
	# m1 = create(1, model='moto')
	# connect(gen, m1, delay=1, weight=5.5, conn_type='all-to-all')

	'''
	gen = create(1, model='generator', segs=1)
	OM1 = create(50, model='inter', segs=1)
	OM2 = create(50, model='inter', segs=1)
	OM3 = create(50, model='inter', segs=1)
	moto = create(50, model='moto', segs=1)
	muscle = create(1, model='muscle', segs=3)

	conn_a2a(gen, OM1, delay=1, weight=1.5)

	conn_fixed_outdegree(OM1, OM2, delay=2, weight=1.85)
	conn_fixed_outdegree(OM2, OM1, delay=3, weight=1.85)
	conn_fixed_outdegree(OM2, OM3, delay=3, weight=0.00055)
	conn_fixed_outdegree(OM1, OM3, delay=3, weight=0.00005)
	conn_fixed_outdegree(OM3, OM2, delay=1, weight=-4.5)
	conn_fixed_outdegree(OM3, OM1, delay=1, weight=-4.5)
	conn_fixed_outdegree(OM2, moto, delay=2, weight=1.5)
	conn_fixed_outdegree(moto, muscle, delay=2, weight=15.5)
	'''
else:
	gen = form_group("gen", 1, model=GENERATOR, segs=1)
	OM1 = form_group("OM1", 50, model=INTER, segs=1)
	OM2 = form_group("OM2", 50, model=INTER, segs=1)
	OM3 = form_group("OM3", 50, model=INTER, segs=1)
	moto = form_group("moto", 50, model=MOTO, segs=1)
	muscle = form_group("muscle", 1, model=MUSCLE, segs=3)

	conn_a2a(gen, OM1, delay=1, weight=1.5)

	conn_fixed_outdegree(OM1, OM2, delay=2, weight=1.85)
	conn_fixed_outdegree(OM2, OM1, delay=3, weight=1.85)
	conn_fixed_outdegree(OM2, OM3, delay=3, weight=0.00055)
	conn_fixed_outdegree(OM1, OM3, delay=3, weight=0.00005)
	conn_fixed_outdegree(OM3, OM2, delay=1, weight=-4.5)
	conn_fixed_outdegree(OM3, OM1, delay=1, weight=-4.5)
	conn_fixed_outdegree(OM2, moto, delay=2, weight=1.5)
	conn_fixed_outdegree(moto, muscle, delay=2, weight=15.5)

	groups = [OM1, OM2, OM3, moto, muscle]
	save(groups)

	P.nrn_start_seg.append(nrns_and_segs)
	#
	# EES = form_group("EES", 1, model=GENERATOR)
	# E1 = form_group("E1", 1, model=GENERATOR)
	# E2 = form_group("E2", 1, model=GENERATOR)
	# E3 = form_group("E3", 1, model=GENERATOR)
	# E4 = form_group("E4", 1, model=GENERATOR)
	# E5 = form_group("E5", 1, model=GENERATOR)
	# #
	# CV1 = form_group("CV1", 1, model=GENERATOR)
	# CV2 = form_group("CV2", 1, model=GENERATOR)
	# CV3 = form_group("CV3", 1, model=GENERATOR)
	# CV4 = form_group("CV4", 1, model=GENERATOR)
	# CV5 = form_group("CV5", 1, model=GENERATOR)
	#
	# C_0 = form_group("C_0")
	# C_1 = form_group("C_1")
	# V0v = form_group("V0v")
	# OM1_0E = form_group("OM1_0E")
	# OM1_0F = form_group("OM1_0F")
	# #
	# OM1_0 = form_group("OM1_0")
	# OM1_1 = form_group("OM1_1")
	# OM1_2_E = form_group("OM1_2_E")
	# OM1_2_F = form_group("OM1_2_F")
	# OM1_3 = form_group("OM1_3")
	# '''
	# #
	# OM2_0 = form_group("OM2_0")
	# OM2_1 = form_group("OM2_1")
	# OM2_2_E = form_group("OM2_2_E")
	# OM2_2_F = form_group("OM2_2_F")
	# OM2_3 = form_group("OM2_3")
	# #
	# OM3_0 = form_group("OM3_0")
	# OM3_1 = form_group("OM3_1")
	# OM3_2_E = form_group("OM3_2_E")
	# OM3_2_F = form_group("OM3_2_F")
	# OM3_3 = form_group("OM3_3")
	# #
	# OM4_0 = form_group("OM4_0")
	# OM4_1 = form_group("OM4_1")
	# OM4_2_E = form_group("OM4_2_E")
	# OM4_2_F = form_group("OM4_2_F")
	# OM4_3 = form_group("OM4_3")
	# #
	# OM5_0 = form_group("OM5_0")
	# OM5_1 = form_group("OM5_1")
	# OM5_2_E = form_group("OM5_2_E")
	# OM5_2_F = form_group("OM5_2_F")
	# OM5_3 = form_group("OM5_3")
	# #
	# '''
	# '''
	# Ia_E = form_group("Ia_E", neurons_in_ip)
	# iIP_E = form_group("iIP_E", neurons_in_ip)
	# R_E = form_group("R_E")
	#
	# Ia_F = form_group("Ia_F", neurons_in_ip)
	# iIP_F = form_group("iIP_F", neurons_in_ip)
	# R_F = form_group("R_F")
	#
	# MN_E = form_group("MN_E", 210, model=MOTO)
	# MN_F = form_group("MN_F", 180, model=MOTO)
	# sens_aff = form_group("sens_aff", 120)
	# Ia_aff_E = form_group("Ia_aff_E", 120)
	# Ia_aff_F = form_group("Ia_aff_F", 120)
	# eIP_E_1 = form_group("eIP_E_1", 40)
	# eIP_E_2 = form_group("eIP_E_2", 40)
	# eIP_E_3 = form_group("eIP_E_3", 40)
	# eIP_E_4 = form_group("eIP_E_4", 40)
	# eIP_E_5 = form_group("eIP_E_5", 40)
	# eIP_F = form_group("eIP_F", neurons_in_ip)
	# # muscle_E = form_group("muscle_E", 150 * 210, model=MUSCLE)
	# # muscle_F = form_group("muscle_F", 100 * 180, model=MUSCLE)
	# '''
	# conn_fixed_outdegree(EES, CV1, delay=1, weight=15)
	# conn_fixed_outdegree(EES, OM1_0, delay=2, weight=0.00075 * k * skin_time)
	# conn_fixed_outdegree(CV1, OM1_0, delay=2, weight=0.00048)
	# # conn_fixed_outdegree(CV1, CV2, delay=1, weight=15)
	#
	# # OM1
	# conn_fixed_outdegree(OM1_0, OM1_1, delay=3, weight=2.95)
	# conn_fixed_outdegree(OM1_1, OM1_2_E, delay=3, weight=2.85)
	# conn_fixed_outdegree(OM1_2_E, OM1_1, delay=3, weight=1.95)
	# conn_fixed_outdegree(OM1_2_E, OM1_3, delay=3, weight=0.0007)
	# # conn_fixed_outdegree(OM1_2_F, OM2_2_F, delay=1.5, weight=2)
	# conn_fixed_outdegree(OM1_1, OM1_3, delay=3, weight=0.00005)
	# conn_fixed_outdegree(OM1_3, OM1_2_E, delay=3, weight=-4.5)
	# conn_fixed_outdegree(OM1_3, OM1_1, delay=3, weight=-4.5)
	#
	# groups = [OM1_0, OM1_1, OM1_2_E, OM1_3]
	# save(groups)
	'''
	# OM2
	conn_fixed_outdegree(OM2_0, OM2_1, delay=3, weight=2.95)
	conn_fixed_outdegree(OM2_1, OM2_2_E, delay=3, weight=2.85)
	conn_fixed_outdegree(OM2_2_E, OM2_1, delay=3, weight=1.95)
	conn_fixed_outdegree(OM2_2_E, OM2_3, delay=3, weight=0.0007)
	conn_fixed_outdegree(OM2_2_F, OM3_2_F, delay=1.5, weight=2)
	conn_fixed_outdegree(OM2_1, OM2_3, delay=3, weight=0.00005)
	conn_fixed_outdegree(OM2_3, OM2_2_E, delay=3, weight=-4.5)
	conn_fixed_outdegree(OM2_3, OM2_1, delay=3, weight=-4.5)
	# OM3
	conn_fixed_outdegree(OM3_0, OM3_1, delay=3, weight=2.95)
	conn_fixed_outdegree(OM3_1, OM3_2_E, delay=3, weight=2.85)
	conn_fixed_outdegree(OM3_2_E, OM3_1, delay=3, weight=1.95)
	conn_fixed_outdegree(OM3_2_E, OM3_3, delay=3, weight=0.0007)
	conn_fixed_outdegree(OM3_2_F, OM4_2_F, delay=1.5, weight=2)
	conn_fixed_outdegree(OM3_1, OM3_3, delay=3, weight=0.00005)
	conn_fixed_outdegree(OM3_3, OM3_2_E, delay=3, weight=-4.5)
	conn_fixed_outdegree(OM3_3, OM3_1, delay=3, weight=-4.5)
	# OM4
	conn_fixed_outdegree(OM4_0, OM4_1, delay=3, weight=2.95)
	conn_fixed_outdegree(OM4_1, OM4_2_E, delay=3, weight=2.85)
	conn_fixed_outdegree(OM4_2_E, OM4_1, delay=3, weight=1.95)
	conn_fixed_outdegree(OM4_2_E, OM4_3, delay=3, weight=0.0007)
	conn_fixed_outdegree(OM4_2_F, OM5_2_F, delay=1.5, weight=2)
	conn_fixed_outdegree(OM4_1, OM4_3, delay=3, weight=0.00005)
	conn_fixed_outdegree(OM4_3, OM4_2_E, delay=3, weight=-4.5)
	conn_fixed_outdegree(OM4_3, OM4_1, delay=3, weight=-4.5)
	# OM5
	conn_fixed_outdegree(OM5_0, OM5_1, delay=3, weight=2.95)
	conn_fixed_outdegree(OM5_1, OM5_2_E, delay=3, weight=2.85)
	conn_fixed_outdegree(OM5_2_E, OM5_1, delay=3, weight=1.95)
	conn_fixed_outdegree(OM5_2_E, OM5_3, delay=3, weight=0.0007)
	conn_fixed_outdegree(OM5_1, OM5_3, delay=3, weight=0.00005)
	conn_fixed_outdegree(OM5_3, OM5_2_E, delay=3, weight=-4.5)
	conn_fixed_outdegree(OM5_3, OM5_1, delay=3, weight=-4.5)
	'''
# ids of neurons
nrns = list(range(nrns_number))
class S:
	Vm = init0(nrns_and_segs)           # [mV] array for three compartments volatge
	n = init0(nrns_and_segs)            # [0..1] compartments channel, providing the kinetic pattern of the L conductance
	m = init0(nrns_and_segs)            # [0..1] compartments channel, providing the kinetic pattern of the Na conductance
	h = init0(nrns_and_segs)            # [0..1] compartments channel, providing the kinetic pattern of the Na conductance
	l = init0(nrns_and_segs)            # [0..1] inward rectifier potassium (Kir) channel
	s = init0(nrns_and_segs)            # [0..1] nodal slow potassium channel
	p = init0(nrns_and_segs)            # [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	hc = init0(nrns_and_segs)           # [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	mc = init0(nrns_and_segs)           # [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	cai = init0(nrns_and_segs)          #
	I_Ca = init0(nrns_and_segs)         # [nA] Ca ionic currents
	NODE_A = init0(nrns_and_segs)       # the effect of this node on the parent node's equation
	NODE_B = init0(nrns_and_segs)       # the effect of the parent node on this node's equation
	NODE_D = init0(nrns_and_segs)       # diagonal element in node equation
	const_NODE_D = init0(nrns_and_segs) # const diagonal element in node equation (performance)
	NODE_RHS = init0(nrns_and_segs)     # right hand side in node equation
	NODE_RINV = init0(nrns_and_segs)    # conductance uS from node to parent
	NODE_AREA = init0(nrns_and_segs)    # area of a node in um^2

class U:
	has_spike = init0(nrns_number, dtype=bool)  # spike flag for each neuron
	spike_on = init0(nrns_number, dtype=bool)   # special flag to prevent fake spike detecting
	# synapses
	g_exc = init0(nrns_number)        # [S] excitatory conductivity level
	g_inh_A = init0(nrns_number)      # [S] inhibitory conductivity level
	g_inh_B = init0(nrns_number)      # [S] inhibitory conductivity level
	factor = init0(nrns_number)       # [const] todo

# extracellular
nlayer = 2
ext_shape = (nrns_and_segs, nlayer)
ext_rhs = init0(ext_shape)   # extracellular right hand side in node equation
ext_v = init0(ext_shape)     # extracellular membrane potential
ext_a = init0(ext_shape)     # extracellular effect of node in parent equation
ext_b = init0(ext_shape)     # extracellular effect of parent in node equation
ext_d = init0(ext_shape)     # extracellular diagonal element in node equation

def get_neuron_data():
	"""
	please note, that this file should contain only specified debugging output
	"""
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
				line = line.replace('BREAKPOINT currents ', '').strip().split("\t")[3:-1] # without Vm
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

def save_data():
	"""
	for debugging with NEURON
	"""
	global GRAS_data
	for nrn_seg in save_neuron_ids:
		# syn il, ina, ik, m, h, n, l, s, v
		isyn = 0 # S.g_exc[nrn_seg] * (S.Vm[nrn_seg] - P.E_ex[nrn_seg])
		GRAS_data.append([S.NODE_A[nrn_seg], S.NODE_B[nrn_seg], S.NODE_D[nrn_seg], S.NODE_RINV[nrn_seg], S.Vm[nrn_seg],
		                  S.NODE_RHS[nrn_seg], ext_v[nrn_seg, 0], isyn, S.m[nrn_seg], S.h[nrn_seg], S.n[nrn_seg],
		                  S.l[nrn_seg], S.s[nrn_seg]])

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
	calculate synaptic current
	"""
	return U.g_exc[nrn] * (voltage - P.E_ex[nrn]) + (U.g_inh_B[nrn] - U.g_inh_A[nrn]) * (voltage - P.E_inh[nrn])

def nrn_moto_current(nrn, nrn_seg_index, voltage):
	"""
	calculate channels current
	"""
	iNa = P.gnabar[nrn] * S.m[nrn_seg_index] ** 3 * S.h[nrn_seg_index] * (voltage - P.ena[nrn])
	iK = P.gkrect[nrn] * S.n[nrn_seg_index] ** 4 * (voltage - P.ek[nrn]) + \
	     P.gcak[nrn] * S.cai[nrn_seg_index] ** 2 / (S.cai[nrn_seg_index] ** 2 + 0.014 ** 2) * (voltage - P.ek[nrn])
	iL = P.gl[nrn] * (voltage - P.el[nrn])
	eCa = (1000 * R_const * 309.15 / (2 * F_const)) * np.log(ca0 / S.cai[nrn_seg_index])
	S.I_Ca[nrn_seg_index] = P.gcaN[nrn] * S.mc[nrn_seg_index] ** 2 * S.hc[nrn_seg_index] * (voltage - eCa) + \
	                      P.gcaL[nrn] * S.p[nrn_seg_index] * (voltage - eCa)
	return iNa + iK + iL + S.I_Ca[nrn_seg_index]

def nrn_fastchannel_current(nrn, nrn_seg_index, voltage):
	"""
	calculate channels current
	"""
	iNa = P.gnabar[nrn] * S.m[nrn_seg_index] ** 3 * S.h[nrn_seg_index] * (voltage - P.ena[nrn])
	iK = P.gkbar[nrn] * S.n[nrn_seg_index] ** 4 * (voltage - P.ek[nrn])
	iL = P.gl[nrn] * (voltage - P.el[nrn])
	return iNa + iK + iL

def recalc_synaptic(nrn):
	"""
	updating conductance (summed) of neurons' post-synaptic conenctions
	"""
	# exc synaptic conductance
	if U.g_exc[nrn] != 0:
		U.g_exc[nrn] -= (1 - np.exp(-dt / P.tau_exc[nrn])) * U.g_exc[nrn]
		if U.g_exc[nrn] < 1e-5:
			U.g_exc[nrn] = 0
	# inh1 synaptic conductance
	if U.g_inh_A[nrn] != 0:
		U.g_inh_A[nrn] -= (1 - np.exp(-dt / P.tau_inh1[nrn])) * U.g_inh_A[nrn]
		if U.g_inh_A[nrn] < 1e-5:
			U.g_inh_A[nrn] = 0
	# inh2 synaptic conductance
	if U.g_inh_B[nrn] != 0:
		U.g_inh_B[nrn] -= (1 - np.exp(-dt / P.tau_inh2[nrn])) * U.g_inh_B[nrn]
		if U.g_inh_B[nrn] < 1e-5:
			U.g_inh_B[nrn] = 0

def syn_initial(nrn):
	"""
	initialize tau (rise/decay time, ms) and factor (const) variables
	"""
	if P.tau_inh1[nrn] / P.tau_inh2[nrn] > 0.9999:
		P.tau_inh1[nrn] = 0.9999 * P.tau_inh2[nrn]
	if P.tau_inh1[nrn] / P.tau_inh2[nrn] < 1e-9:
		P.tau_inh1[nrn] = P.tau_inh2[nrn] * 1e-9
	#
	tp = (P.tau_inh1[nrn] * P.tau_inh2[nrn]) / (P.tau_inh2[nrn] - P.tau_inh1[nrn]) * np.log(P.tau_inh2[nrn] / P.tau_inh1[nrn])
	U.factor[nrn] = -np.exp(-tp / P.tau_inh1[nrn]) + np.exp(-tp / P.tau_inh2[nrn])
	U.factor[nrn] = 1 / U.factor[nrn]

def nrn_inter_initial(nrn_seg_index, V):
	"""
	initialize channels, based on cropped evaluate_fct function
	"""
	V_mem = V - V_adj
	#
	a = 0.32 * (13 - V_mem) / (np.exp((13 - V_mem) / 4) - 1)
	b = 0.28 * (V_mem - 40) / (np.exp((V_mem - 40) / 5) - 1)
	S.m[nrn_seg_index] = a / (a + b)   # m_inf
	#
	a = 0.128 * np.exp((17 - V_mem) / 18)
	b = 4 / (1 + np.exp((40 - V_mem) / 5))
	S.h[nrn_seg_index] = a / (a + b)   # h_inf
	#
	a = 0.032 * (15 - V_mem) / (np.exp((15 - V_mem) / 5) - 1)
	b = 0.5 * np.exp((10 - V_mem) / 40)
	S.n[nrn_seg_index] = a / (a + b)   # n_inf

def nrn_moto_initial(nrn_seg_index, V):
	"""
	initialize channels, based on cropped evaluate_fct function
	"""
	a = alpham(V)
	S.m[nrn_seg_index] = a / (a + betam(V))                # m_inf
	S.h[nrn_seg_index] = 1 / (1 + Exp((V + 65) / 7))       # h_inf
	S.p[nrn_seg_index] = 1 / (1 + Exp(-(V + 55.8) / 3.7))  # p_inf
	S.n[nrn_seg_index] = 1 / (1 + Exp(-(V + 38) / 15))     # n_inf
	S.mc[nrn_seg_index] = 1 / (1 + Exp(-(V + 32) / 5))     # mc_inf
	S.hc[nrn_seg_index] = 1 / (1 + Exp((V + 50) / 5))      # hc_inf
	S.cai[nrn_seg_index] = 0.0001

def nrn_muslce_initial(nrn_seg_index, V):
	"""
	initialize channels, based on cropped evaluate_fct function
	"""
	V_mem = V - V_adj
	#
	a = 0.32 * (13 - V_mem) / (np.exp((13 - V_mem) / 4) - 1)
	b = 0.28 * (V_mem - 40) / (np.exp((V_mem - 40) / 5) - 1)
	S.m[nrn_seg_index] = a / (a + b)   # m_inf
	#
	a = 0.128 * np.exp((17 - V_mem) / 18)
	b = 4 / (1 + np.exp((40 - V_mem) / 5))
	S.h[nrn_seg_index] = a / (a + b)   # h_inf
	#
	a = 0.032 * (15 - V_mem) / (np.exp((15 - V_mem) / 5) - 1)
	b = 0.5 * np.exp((10 - V_mem) / 40)
	S.n[nrn_seg_index] = a / (a + b)   # n_inf

def recalc_inter_channels(nrn_seg_index, V):
	"""
	calculate new states of channels (evaluate_fct)
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
	# states
	S.m[nrn_seg_index] += (1 - np.exp(-dt / tau_m)) * (m_inf - S.m[nrn_seg_index])
	S.h[nrn_seg_index] += (1 - np.exp(-dt / tau_h)) * (h_inf - S.h[nrn_seg_index])
	S.n[nrn_seg_index] += (1 - np.exp(-dt / tau_n)) * (n_inf - S.n[nrn_seg_index])

def recalc_moto_channels(nrn_seg_index, V):
	"""
	calculate new states of channels (evaluate_fct)
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
	mc_inf = 1 / (1 + Exp(-(V + 32) / 5))
	hc_inf = 1 / (1 + Exp((V + 50) / 5))
	# CALCIUM DYNAMICS L-type
	tau_p = 400
	p_inf = 1 / (1 + Exp(-(V + 55.8) / 3.7))
	# states
	S.m[nrn_seg_index] += (1 - np.exp(-dt / tau_m)) * (m_inf - S.m[nrn_seg_index])
	S.h[nrn_seg_index] += (1 - np.exp(-dt / tau_h)) * (h_inf - S.h[nrn_seg_index])
	S.p[nrn_seg_index] += (1 - np.exp(-dt / tau_p)) * (p_inf - S.p[nrn_seg_index])
	S.n[nrn_seg_index] += (1 - np.exp(-dt / tau_n)) * (n_inf - S.n[nrn_seg_index])
	S.mc[nrn_seg_index] += (1 - np.exp(-dt / 15)) * (mc_inf - S.mc[nrn_seg_index])    # tau_mc = 15
	S.hc[nrn_seg_index] += (1 - np.exp(-dt / 50)) * (hc_inf - S.hc[nrn_seg_index])    # tau_hc = 50
	S.cai[nrn_seg_index] += (1 - np.exp(-dt * 0.04)) * (-0.01 * S.I_Ca[nrn_seg_index] / 0.04 - S.cai[nrn_seg_index])

def recalc_muslce_channels(nrn_seg_index, V):
	"""
	calculate new states of channels (evaluate_fct)
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
	S.m[nrn_seg_index] += (1 - np.exp(-dt / tau_m)) * (m_inf - S.m[nrn_seg_index])
	S.h[nrn_seg_index] += (1 - np.exp(-dt / tau_h)) * (h_inf - S.h[nrn_seg_index])
	S.n[nrn_seg_index] += (1 - np.exp(-dt / tau_n)) * (n_inf - S.n[nrn_seg_index])
	S.l[nrn_seg_index] += (1 - np.exp(-dt / taul)) * (linf - S.l[nrn_seg_index])
	S.s[nrn_seg_index] += (1 - np.exp(-dt / stau)) * (sinf - S.s[nrn_seg_index])

def nrn_rhs_ext(nrn):
	"""
	void nrn_rhs_ext(NrnThread* _nt)
	"""
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]
	# nd rhs contains -membrane current + stim current
	# nde rhs contains stim current
	# todo passed
	for nrn_seg in range(i1, i3):
		ext_rhs[nrn_seg, 0] -= S.NODE_RHS[nrn_seg]
	#
	for nrn_seg in range(i1 + 1, i3):
		# for j in range(nlayer):
		# 	# dv = 0
		# 	dv = ext_v[nrn, pnd, j] - ext_v[nrn, nd, j]
		# 	ext_rhs[nrn, nd, j] -= ext_b[nrn, nd, j] * dv
		# 	ext_rhs[nrn, pnd, j] += ext_a[nrn, nd, j] * dv
		# series resistance and battery to ground between nlayer-1 and ground
		j = nlayer - 1
		# print(f"??V0 {ext_v[nrn, nd, 0]} V1 {ext_v[nrn, nd, 1]} RHS0 {ext_rhs[nrn, nd, 0]} RHS1 {ext_rhs[nrn, nd, 1]}")
		nd = nrn_seg - i1
		ext_rhs[nrn_seg, j] -= xg[nd] * (ext_v[nrn_seg, j] - e_extracellular)
		# for (--j; j >= 0; --j) { // between j and j+1 layer
		j = 0
		# print(f"V0 {ext_v[nrn, nd, 0]} V1 {ext_v[nrn, nd, 1]} RHS0 {ext_rhs[nrn, nd, 0]} RHS1 {ext_rhs[nrn, nd, 1]}")
		x = xg[nd] * (ext_v[nrn_seg, j] - ext_v[nrn_seg, j + 1])
		ext_rhs[nrn_seg, j] -= x
		ext_rhs[nrn_seg, j + 1] += x

		# print(f"==>V0 {ext_v[nrn, nd, 0]} V1 {ext_v[nrn, nd, 1]} RHS0 {ext_rhs[nrn, nd, 0]} RHS1 {ext_rhs[nrn, nd, 1]}")

def nrn_setup_ext(nrn):
	"""
	void nrn_setup_ext(NrnThread* _nt)
	"""
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]

	cj = 1 / dt
	cfac = 0.001 * cj

	# todo find the place where it is zeroed
	ext_d[i1:i3, :] = 0

	# d contains all the membrane conductances (and capacitance)
	# i.e. (cm/dt + di/dvm - dis/dvi)*[dvi] and (dis/dvi)*[dvx]
	for nrn_seg in range(i1, i3):
		# nde->_d only has -ELECTRODE_CURRENT contribution
		ext_d[nrn_seg, 0] += S.NODE_D[nrn_seg]
	# D[0] = [0 0.1442 0.1442 0.1442 0 ]

	# series resistance, capacitance, and axial terms
	for nrn_seg in range(i1 + 1, i3):
		# series resistance and capacitance to ground
		j = 0
		nd = nrn_seg - i1   # start indexing from 0
		while True:
			mfac = xg[nd] + xc[nd] * cfac
			ext_d[nrn_seg, j] += mfac
			j += 1
			if j == nlayer:
				break
			ext_d[nrn_seg, j] += mfac
		# axial connections
		for j in range(nlayer):
			ext_d[nrn_seg, j] -= ext_b[nrn_seg, j]
			ext_d[nrn_seg - 1, j] -= ext_a[nrn_seg, j]
	# D[0] = [2e-08 1e+09 1e+09 1e+09 2e-08 ]
	# D[1] = [2e-08 2e+09 2e+09 2e+09 2e-08 ]

def nrn_update_2d(nrn):
	"""
	void nrn_update_2d(NrnThread* nt)

	update has already been called so modify nd->v based on dvi we only need to
	update extracellular nodes and base the corresponding nd->v on dvm (dvm = dvi - dvx)
	"""
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]
	# final voltage updating
	for nrn_seg in range(i1, i3):
		for j in range(nlayer):
			ext_v[nrn_seg, j] += ext_rhs[nrn_seg, j]

def nrn_rhs(nrn):
	"""
	void nrn_rhs(NrnThread *_nt) combined with the first part of nrn_lhs
	calculate right hand side of
	cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	This is a common operation for fixed step, cvode, and daspk methods
	"""
	# init _rhs and _lhs (NODE_D) as zero
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]
	S.NODE_RHS[i1:i3] = 0
	S.NODE_D[i1:i3] = 0
	ext_rhs[i1:i3, :] = 0

	# update MOD rhs, CAPS has no current [CAP MOD CAP]!
	center_segment = i1 + (2 if P.models[nrn] == MUSCLE else 1)
	# update segments except CAPs
	for nrn_seg in range(i1 + 1, i3 - 1):
		V = S.Vm[nrn_seg]
		# SYNAPTIC update
		if nrn_seg == center_segment:
			# static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
			_g = syn_current(nrn, V + 0.001)
			_rhs = syn_current(nrn, V)
			_g = (_g - _rhs) / .001
			_g *= 1.e2 / S.NODE_AREA[nrn_seg]
			_rhs *= 1.e2 / S.NODE_AREA[nrn_seg]
			S.NODE_RHS[nrn_seg] -= _rhs
			# static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
			S.NODE_D[nrn_seg] += _g

		# NEURON update
		# static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		if P.models[nrn] == INTER:
			# muscle and inter has the same fast_channel function
			_g = nrn_fastchannel_current(nrn, nrn_seg, V + 0.001)
			_rhs = nrn_fastchannel_current(nrn, nrn_seg, V)
		elif P.models[nrn] == MOTO:
			_g = nrn_moto_current(nrn, nrn_seg, V + 0.001)
			_rhs = nrn_moto_current(nrn, nrn_seg, V)
		elif P.models[nrn] == MUSCLE:
			# muscle and inter has the same fast_channel function
			_g = nrn_fastchannel_current(nrn, nrn_seg, V + 0.001)
			_rhs = nrn_fastchannel_current(nrn, nrn_seg, V)
		else:
			raise Exception('No nrn model found')
		# save data like in NEURON (after .mod nrn_cur)
		_g = (_g - _rhs) / 0.001
		S.NODE_RHS[nrn_seg] -= _rhs
		# static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
		S.NODE_D[nrn_seg] += _g
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

	# todo: always 0, because Vm0 = Vm1 = Vm2 at [CAP node CAP] model (1 section)
	for nrn_seg in range(i1 + 1, i3):
		dv = S.Vm[nrn_seg - 1] - S.Vm[nrn_seg]
		# our connection coefficients are negative so
		S.NODE_RHS[nrn_seg] -= S.NODE_B[nrn_seg] * dv
		S.NODE_RHS[nrn_seg - 1] += S.NODE_A[nrn_seg] * dv

def nrn_lhs(nrn):
	"""
	void nrn_lhs(NrnThread *_nt)
	NODE_D[nrn, nd] updating is located at nrn_rhs, because _g is not the global variable
	"""
	S.NODE_D[nrn, :] += S.const_NODE_D[nrn, :]
	raise NotImplemented

def bksub(nrn):
	"""
	void bksub(NrnThread* _nt)
	"""
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]
	# intracellular
	# note that loop from i1 to i1 + 1 is always SINGLE element
	S.NODE_RHS[i1] /= S.NODE_D[i1]
	#
	for nrn_seg in range(i1 + 1, i3):
		S.NODE_RHS[nrn_seg] -= S.NODE_B[nrn_seg] * S.NODE_RHS[nrn_seg - 1]
		S.NODE_RHS[nrn_seg] /= S.NODE_D[nrn_seg]
	# extracellular
	if EXTRACELLULAR:
		for j in range(nlayer):
			ext_rhs[i1, j] /= ext_d[i1, j]
		for nrn_seg in range(i1 + 1, i3):
			for j in range(nlayer):
				ext_rhs[nrn_seg, j] -= ext_b[nrn_seg, j] * ext_rhs[nrn_seg - 1, j]
				ext_rhs[nrn_seg, j] /= ext_d[nrn_seg, j]

def triang(nrn):
	"""
	void triang(NrnThread* _nt)
	"""
	# intracellular
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]
	nrn_seg = i3 - 1
	while nrn_seg >= i1 + 1:
		ppp = S.NODE_A[nrn_seg] / S.NODE_D[nrn_seg]
		S.NODE_D[nrn_seg - 1] -= ppp * S.NODE_B[nrn_seg]
		S.NODE_RHS[nrn_seg - 1] -= ppp * S.NODE_RHS[nrn_seg]
		nrn_seg -= 1

	# extracellular
	if EXTRACELLULAR:
		nrn_seg = i3 - 1
		while nrn_seg >= i1 + 1:
			for j in range(nlayer):
				ppp = ext_a[nrn_seg, j] / ext_d[nrn_seg, j]
				ext_d[nrn_seg - 1, j] -= ppp * ext_b[nrn_seg, j]
				ext_rhs[nrn_seg - 1, j] -= ppp * ext_rhs[nrn_seg, j]
			nrn_seg -= 1

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
	# simplified nrn_lhs(nrn)
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]
	S.NODE_D[i1:i3] += S.const_NODE_D[i1:i3]

def update(nrn):
	"""
	void update(NrnThread* _nt)
	"""
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]
	# final voltage updating
	for nrn_seg in range(i1, i3):
		S.Vm[nrn_seg] += S.NODE_RHS[nrn_seg]
	# save data like in NEURON (after .mod nrn_cur)
	if DEBUG and nrn in save_neuron_ids:
		save_data()
	# extracellular
	nrn_update_2d(nrn)

def deliver_net_events():
	"""
	void deliver_net_events(NrnThread* nt)
	"""
	for index, pre_nrn in enumerate(syn_pre_nrn):
		if U.has_spike[pre_nrn] and syn_delay_timer[index] == -1:
			syn_delay_timer[index] = syn_delay[index] + 1
		if syn_delay_timer[index] == 0:
			post_id = syn_post_nrn[index]
			weight = syn_weight[index]
			if weight >= 0:
				U.g_exc[post_id] += weight
			else:
				U.g_inh_A[post_id] += -weight * U.factor[post_id]
				U.g_inh_B[post_id] += -weight * U.factor[post_id]
			syn_delay_timer[index] = -1
		if syn_delay_timer[index] > 0:
			syn_delay_timer[index] -= 1
	# reset spikes
	U.has_spike[:] = False

def nrn_deliver_events(nrn):
	"""
	void nrn_deliver_events(NrnThread* nt)
	"""
	# get the central segment (for detecting spikes): i1 + (2 or 1)
	seg_update = P.nrn_start_seg[nrn] + (2 if P.models[nrn] == MUSCLE else 1)
	# check if neuron has spike with special flag for avoidance multi-spike detecting
	if not U.spike_on[nrn] and S.Vm[seg_update] > V_th:
		U.spike_on[nrn] = True
		U.has_spike[nrn] = True
	elif S.Vm[seg_update] < V_th:
		U.spike_on[nrn] = False

def nrn_fixed_step_lastpart(nrn):
	"""
	void *nrn_fixed_step_lastpart(NrnThread *nth)
	"""
	i1 = P.nrn_start_seg[nrn]
	i3 = P.nrn_start_seg[nrn + 1]
	# update synapses' state
	recalc_synaptic(nrn)
	# update neurons' segment state
	if P.models[nrn] == INTER:
		for nrn_seg in range(i1, i3):
			recalc_inter_channels(nrn_seg, S.Vm[nrn_seg])
	elif P.models[nrn] == MOTO:
		for nrn_seg in range(i1, i3):
			recalc_moto_channels(nrn_seg, S.Vm[nrn_seg])
	elif P.models[nrn] == MUSCLE:
		for nrn_seg in range(i1, i3):
			recalc_muslce_channels(nrn_seg, S.Vm[nrn_seg])
	else:
		raise Exception("No model")
	# spike detection for
	nrn_deliver_events(nrn)

def nrn_area_ri():
	"""
	void nrn_area_ri(Section *sec) [790] treeset.c
	area for right circular cylinders. Ri as right half of parent + left half of this
	"""
	for nrn in nrns:
		if P.models[nrn] == GENERATOR:
			continue
		i1 = P.nrn_start_seg[nrn]
		i3 = P.nrn_start_seg[nrn + 1]
		segments = (i3 - i1 - 2)
		# dx = section_length(sec) / ((double) (sec->nnode - 1));
		dx = P.length[nrn] / segments # divide by the last index of node (or segments count)
		rright = 0
		# todo sec->pnode needs +1 index
		for nrn_seg in range(i1 + 1, i1 + segments + 1):
			# area for right circular cylinders. Ri as right half of parent + left half of this
			S.NODE_AREA[nrn_seg] = np.pi * dx * P.diam[nrn]
			rleft = 1e-2 * P.Ra[nrn] * (dx / 2) / (np.pi * P.diam[nrn] ** 2 / 4) # left half segment Megohms
			S.NODE_RINV[nrn_seg] = 1 / (rleft + rright) # uS
			rright = rleft
		# the first and last segments has zero length. Area is 1e2 in dimensionless units
		S.NODE_AREA[i1] = 100
		nrn_seg = i1 + segments + 1 # the last segment
		S.NODE_AREA[nrn_seg] = 100
		S.NODE_RINV[nrn_seg] = 1 / rright

def ext_con_coef():
	"""
	void ext_con_coef(void)
	setup a and b
	"""
	layer = 0
	# todo: extracellular only for those neurons who need
	for nrn in nrns:
		if P.models[nrn] == GENERATOR:
			continue
		i1 = P.nrn_start_seg[nrn]
		i3 = P.nrn_start_seg[nrn + 1]
		segments = (i3 - i1 - 2)
		# temporarily store half segment resistances in rhs
		# todo sec->pnode needs +1 index, also xraxial is common
		for nrn_seg in range(i1 + 1, i1 + segments + 1):
			dx = P.length[nrn] / segments
			ext_rhs[nrn_seg, layer] = 1e-4 * xraxial * dx / 2  # Megohms
		# last segment has 0 length
		ext_rhs[i3 - 1, layer] = 0 # todo i3 -1 or just i3
		# NEURON RHS = [5e+07 5e+07 5e+07 0 ]

		# node half resistances in general get added to the node and to the node's "child node in the same section".
		# child nodes in different sections don't involve parent node's resistance
		ext_b[i1 + 1, layer] = ext_rhs[i1 + 1, layer]
		# todo sec->pnode needs +1 index
		for nrn_seg in range(i1 + 1 + 1, i1 + segments + 1 + 1):
			ext_b[nrn_seg, layer] = ext_rhs[nrn_seg, layer] + ext_rhs[nrn_seg - 1, layer]  # Megohms
		# NEURON B = [5e+07 1e+08 1e+08 5e+07 ]

		# first the effect of node on parent equation. Note That last nodes have area = 1.e2 in
		# dimensionless units so that last nodes have units of microsiemens's
		area = S.NODE_AREA[i1]    # parentnode index of sec is 0
		rall_branch = 1  # sec->prop->dparam[4].val
		ext_a[i1 + 1, layer] = -1.e2 * rall_branch / (ext_b[i1 + 1, layer] * area)
		# todo sec->pnode needs +1 index
		for nrn_seg in range(i1 + 1+ 1, i1 + segments + 1 + 1):
			area = S.NODE_AREA[nrn_seg]  # pnd = nd - 1
			ext_a[nrn_seg, layer] = -1.e2 / (ext_b[nrn_seg, layer] * area)
		# NEURON A = [-2e-08 -7.95775e-12 -7.95775e-12 -1.59155e-11 ]

		# now the effect of parent on node equation
		# todo sec->pnode needs +1 index
		for nrn_seg in range(i1 + 1, i1 + segments + 1 + 1):
			ext_b[nrn_seg, layer] = -1.e2 / (ext_b[nrn_seg, layer] * S.NODE_AREA[nrn_seg])
		# NEURON B = [-1.59155e-11 -7.95775e-12 -7.95775e-12 -2e-08 ]

		# the same for other layers
		ext_a[i1:i3, 1] = ext_a[i1:i3, 0].copy()
		ext_b[i1:i3, 1] = ext_b[i1:i3, 0].copy()
		ext_rhs[i1:i3, 1] = ext_rhs[i1:i3, 0].copy()
		# todo recheck: RHS initially is zero!
		ext_rhs[i1:i3, :] = 0

def connection_coef():
	"""
	void connection_coef(void) treeset.c
	"""
	nrn_area_ri()
	# NODE_A is the effect of this node on the parent node's equation
	# NODE_B is the effect of the parent node on this node's equation
	for nrn in nrns:
		if P.models[nrn] == GENERATOR:
			continue
		i1 = P.nrn_start_seg[nrn]
		i3 = P.nrn_start_seg[nrn + 1]
		segments = (i3 - i1 - 2)
		# first the effect of node on parent equation. Note that last nodes have area = 1.e2 in dimensionless
		# units so that last nodes have units of microsiemens
		#todo sec->pnode needs +1 index
		nrn_seg = i1 + 1
		# sec->prop->dparam[4].val = 1, what is dparam[4].val
		S.NODE_A[nrn_seg] = -1.e2 * 1 * S.NODE_RINV[nrn_seg] / S.NODE_AREA[nrn_seg - 1]
		# todo sec->pnode needs +1 index
		for nrn_seg in range(i1 + 1 + 1, i1 + segments + 1 + 1):
			S.NODE_A[nrn_seg] = -1.e2 * S.NODE_RINV[nrn_seg] / S.NODE_AREA[nrn_seg - 1]
		# now the effect of parent on node equation
		# todo sec->pnode needs +1 index
		for nrn_seg in range(i1 + 1, i1 + segments + 1 + 1):
			S.NODE_B[nrn_seg] = -1.e2 * S.NODE_RINV[nrn_seg] / S.NODE_AREA[nrn_seg]
	# for extracellular
	ext_con_coef()
	# note: from LHS, this functions just recalc each time the constant NODED (!)
	"""
	void nrn_lhs(NrnThread *_nt)
	NODE_D[nrn, nd] updating is located at nrn_rhs, because _g is not the global variable
	"""
	# nt->cj = 2/dt if (secondorder) else 1/dt
	# note, the first is CAP
	# function nrn_cap_jacob(_nt, _nt->tml->ml);
	cj = 1 / dt
	cfac = 0.001 * cj
	for nrn in nrns:
		if P.models[nrn] == GENERATOR:
			continue
		i1 = P.nrn_start_seg[nrn]
		i3 = P.nrn_start_seg[nrn + 1]
		segments = (i3 - i1 - 2)
		for nrn_seg in range(i1 + 1, i1 + segments + 1):  # added +1 for nodelist
			S.const_NODE_D[nrn_seg] += cfac * P.Cm[nrn]
		# updating NODED
		for nrn_seg in range(i1 + 1, i3):
			S.const_NODE_D[nrn_seg] -= S.NODE_B[nrn_seg]
			S.const_NODE_D[nrn_seg - 1] -= S.NODE_A[nrn_seg]

	# extra
	# _a_matelm += NODE_A[nrn, nd]
	# _b_matelm += NODE_B[nrn, nd]

def finitialize(v_init=-70):
	"""

	"""
	# todo do not invoke for generators
	connection_coef()
	# for different models -- different init function
	for nrn in nrns:
		# do not init neuron state for generator
		if P.models[nrn] == GENERATOR:
			continue
		i1 = P.nrn_start_seg[nrn]
		i3 = P.nrn_start_seg[nrn + 1]
		# for each segment init the neuron model
		for nrn_seg in range(i1, i3):
			S.Vm[nrn_seg] = v_init
			if P.models[nrn] == INTER:
				nrn_inter_initial(nrn_seg, v_init)
			elif P.models[nrn] == MOTO:
				nrn_moto_initial(nrn_seg, v_init)
			elif P.models[nrn] == MUSCLE:
				nrn_muslce_initial(nrn_seg, v_init)
			else:
				raise Exception("No nrn model found")
		# init RHS/LHS
		setup_tree_matrix(nrn)
		# init tau synapses
		syn_initial(nrn)

	# initialization process should not be recorderd
	GRAS_data.clear()

def nrn_fixed_step_thread(t):
	"""
	void *nrn_fixed_step_thread(NrnThread *nth)
	"""
	# update synapses
	deliver_net_events()

	U.has_spike[gen[1]] = t in EES_stimulus
	# has_spike[EES[1]] = t in EES_stimulus
	# has_spike[CV1[1]] = t in CV1_stimulus
	# has_spike[CV2[1]] = t in CV2_stimulus
	# has_spike[CV3[1]] = t in CV3_stimulus
	# has_spike[CV4[1]] = t in CV4_stimulus
	# has_spike[CV5[1]] = t in CV5_stimulus

	# update data for each neuron
	for nrn in nrns[generators_id_end:]:
		setup_tree_matrix(nrn)
		nrn_solve(nrn)
		update(nrn)
		nrn_fixed_step_lastpart(nrn)

	for nrn in nrns:
		if U.has_spike[nrn]:
			spikes.append(t * dt)

def simulation():
	"""
	Notes: NrnThread represent collection of cells or part of a cell computed by single thread within NEURON process
	"""
	# create_nrns()
	finitialize()
	# start simulation loop
	for t in range(sim_time_steps):
		#
		if t % 100 == 0:
			print(t * dt)
		#
		nrn_fixed_step_thread(t)
		#
		if not DEBUG:
			saved_voltage.append(S.Vm[save_neuron_ids])

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

def start():
	global saved_voltage, generators_id_end, GRAS_data
	# get the last ID of the generatr (generators are created firstly)
	generators_id_end = P.models.count(GENERATOR)
	# start the simulation
	t_start = time()
	simulation()
	t_end = time()
	print(t_end - t_start)
	#
	GRAS_data = np.array(GRAS_data)
	if DEBUG:
		xlength = GRAS_data.shape[0]
		NEURON_data = get_neuron_data()[:xlength, :]
		log.info(f"GRAS shape {GRAS_data.shape}")
		log.info(f"NEURON shape {NEURON_data.shape}")
		plot(GRAS_data, NEURON_data)
	else:
		plt.close()
		saved_voltage = np.array(saved_voltage)
		xticks = np.arange(sim_time_steps) * dt
		ind = 0
		for group in groups:
			size = len(group[1])
			data = np.mean(saved_voltage[:, ind:ind + size], axis=1)
			plt.plot(xticks, data, label=group[0])
			ind += size
		plt.legend()
		plt.show()

if __name__ == "__main__":
	start()
