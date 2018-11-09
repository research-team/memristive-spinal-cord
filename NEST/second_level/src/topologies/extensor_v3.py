import random
from nest import Create, Connect, CopyModel
from NEST.second_level.src.data import *
from NEST.second_level.src.tools.multimeter import add_multimeter
from NEST.second_level.src.tools.spike_detector import add_spike_detector
from NEST.second_level.src.namespace import *

HEBBIAN = {
	'alpha': 1.0,  # Asymmetry parameter (scales depressing increments as  alpha*lambda)
	'lambda': 0.1,  # Step size
	'mu_plus': 0.0,  # Weight dependence exponent, potentiation
	'mu_minus': 0.0  # Weight dependence exponent, depression
}
ANTI_HEBBIAN = {
	'alpha': 1.0,
	'lambda': -0.1,
	'mu_plus': 0.0,
	'mu_minus': 0.0
}
SOMBRERO = {
	'alpha': -1.0,
	'lambda': 0.1,
	'mu_plus': 0.0,
	'mu_minus': 0.0
}
ANTI_SOMBRERO = {
	'alpha': -1.0,
	'lambda': -0.1,
	'mu_plus': 0.0,
	'mu_minus': 0.0
}

"""
class Delay:
	def __init__(self, time=None):
		global delay_module_number
		node1 = self.create_with_mmeter('Delay_N{}_node1'.format(delay_module_number))
		node2 = self.create_with_mmeter('Delay_N{}_node2'.format(delay_module_number))
		node3 = self.create_with_mmeter('Delay_N{}_node3'.format(delay_module_number))
		node4 = self.create_with_mmeter('Delay_N{}_node3'.format(delay_module_number))

		connect(node1, node2, weight=10, delay=2)
		connect(node1, node3, weight=10, delay=2)
		connect(node2, node1, weight=10, delay=2)
		connect(node2, node3, weight=10, delay=2)
		connect(node3, node1, weight=-10, delay=2)
		connect(node3, node2, weight=-10, delay=2)
		connect(node4, node3, weight=-10, delay=2)

		delay_module_number += 1

	@property
	def reset(self):
		return self.node1 + self.node2

	@property
	def mod(self):
		return self.node4

	@property
	def input(self):
		return self.node1

	@property
	def output(self):
		return self.node3
class Generator:
	def __init__(self):
		global generator_module_number
		node1 = self.create_with_mmeter('Generator_N{}_node1'.format(generator_module_number))
		node2 = self.create_with_mmeter('Generator_N{}_node2'.format(generator_module_number))
		node3 = self.create_with_mmeter('Generator_N{}_node3'.format(generator_module_number))

		connect(node1, node2, weight=10, delay=2)
		connect(node1, node3, weight=10, delay=2)
		connect(node2, node1, weight=10, delay=2)
		connect(node2, node3, weight=10, delay=2)
		connect(node3, node1, weight=-10, delay=2)
		connect(node3, node2, weight=-10, delay=2)

		generator_module_number += 1

	@property
	def output(self):
		return self.node1 + self.node2

	@property
	def input(self):
		return self.node1
class Subthreshold:
	def __init__(self):
		global subthreshold_module_number
		node1 = self.create_with_mmeter('Subthreshold_N{}_node1'.format(subthreshold_module_number))
		node2 = self.create_with_mmeter('Subthreshold_N{}_node2'.format(subthreshold_module_number))
		node3 = self.create_with_mmeter('Subthreshold_N{}_node3'.format(subthreshold_module_number))

		connect(node1, node2, weight=10, delay=2)
		connect(node1, node3, weight=10, delay=2)
		connect(node2, node1, weight=10, delay=2)
		connect(node2, node3, weight=-10, delay=2)

		subthreshold_module_number += 1

	@property
	def output(self):
		return self.node3

	@property
	def reset(self):
		return self.node1 + self.node2

	@property
	def input(self):
		return self.node1 + self.node3
"""

class Topology:
	def __init__(self, simulation_params, test_iteration=0):
		self.iteration = test_iteration
		self.multitest = simulation_params[Params.MULTITEST.value]
		self.inh_coef = simulation_params[Params.INH_COEF.value]
		self.sim_time = simulation_params[Params.SIM_TIME.value]
		self.ees_rate = simulation_params[Params.EES_RATE.value]

		neurons_in_moto = 169

		C1 = self.create_with_mmeter("C1")
		C2 = self.create_with_mmeter("C2")
		C3 = self.create_with_mmeter("C3")
		C4 = self.create_with_mmeter("C4")
		C5 = self.create_with_mmeter("C5")

		group1 = self.create_with_mmeter("group1")
		group2 = self.create_with_mmeter("group2")
		group3 = self.create_with_mmeter("group3")
		group4 = self.create_with_mmeter("group4")

		D1_1 = self.create_with_mmeter("D1_1")
		D1_2 = self.create_with_mmeter("D1_2")
		D1_3 = self.create_with_mmeter("D1_3")
		D1_4 = self.create_with_mmeter("D1_4")

		D2_1 = self.create_with_mmeter("D2_1")
		D2_2 = self.create_with_mmeter("D2_2")
		D2_3 = self.create_with_mmeter("D2_3")
		D2_4 = self.create_with_mmeter("D2_4")

		D3_1 = self.create_with_mmeter("D3_1")
		D3_2 = self.create_with_mmeter("D3_2")
		D3_3 = self.create_with_mmeter("D3_3")
		D3_4 = self.create_with_mmeter("D3_4")

		D4_1 = self.create_with_mmeter("D4_1")
		D4_2 = self.create_with_mmeter("D4_2")
		D4_3 = self.create_with_mmeter("D4_3")
		D4_4 = self.create_with_mmeter("D4_4")

		D5_1 = self.create_with_mmeter("D5_1")
		D5_2 = self.create_with_mmeter("D5_2")
		D5_3 = self.create_with_mmeter("D5_3")
		D5_4 = self.create_with_mmeter("D5_4")

		G1_1 = self.create_with_mmeter("G1_1")
		G1_2 = self.create_with_mmeter("G1_2")
		G1_3 = self.create_with_mmeter("G1_3")

		G2_1 = self.create_with_mmeter("G2_1")
		G2_2 = self.create_with_mmeter("G2_2")
		G2_3 = self.create_with_mmeter("G2_3")
#
#		G3_1 = self.create_with_mmeter("G3_1")
#		G3_2 = self.create_with_mmeter("G3_2")
#		G3_3 = self.create_with_mmeter("G3_3")
#
#		G4_1 = self.create_with_mmeter("G4_1")
#		G4_2 = self.create_with_mmeter("G4_2")
#		G4_3 = self.create_with_mmeter("G4_3")
#
#		G5_1 = self.create_with_mmeter("G5_1")
#		G5_2 = self.create_with_mmeter("G5_2")
#		G5_3 = self.create_with_mmeter("G5_3")

		IP_E = self.create_with_mmeter("IP_E", neurons_in_moto)
#		IP_F = self.create_with_mmeter("IP_F", neurons_in_moto)

		MP_E = self.create_with_mmeter("MP_E", neurons_in_moto)
#		MP_F = self.create_with_mmeter("MP_F", neurons_in_moto)
#
#		R_E = self.create_with_mmeter("R_E", neurons_in_moto)
#		R_F = self.create_with_mmeter("R_F", neurons_in_moto)
#
#		Ia_E = self.create_with_mmeter("Ia_E", neurons_in_moto)
#		Ia_F = self.create_with_mmeter("Ia_F", neurons_in_moto)
#
#		Ib_E = self.create_with_mmeter("Ib_E", neurons_in_moto)
#		Ib_F = self.create_with_mmeter("Ib_F", neurons_in_moto)
#
#		Extensor = self.create_with_mmeter("Extensor", neurons_in_moto)
#		Flexor = self.create_with_mmeter("Flexor", neurons_in_moto)
#
#		Ia = self.create_with_mmeter("Ia")
		EES = self.create_with_mmeter("EES")
#		C_0 = self.create_with_mmeter("C=0")
#		C_1 = self.create_with_mmeter("C=1")

		self.connect_spike_generator(C1, t_start=0, t_end=25, rate=200)
		self.connect_spike_generator(C2, t_start=25, t_end=50, rate=200)
		self.connect_spike_generator(C3, t_start=50, t_end=75, rate=200)
		self.connect_spike_generator(C4, t_start=75, t_end=125, rate=200)
		self.connect_spike_generator(C5, t_start=125, t_end=150, rate=200)
		self.connect_spike_generator(EES, t_start=0, t_end=150, rate=self.ees_rate)


		''' D1 '''
		# input from
		connect(C1, D1_1, delay=1.0, weight=3.0)
		connect(C1, D1_4, delay=1.0, weight=3.0)
		connect(C2, D1_1, delay=1.0, weight=3.0)
		connect(C2, D1_4, delay=1.0, weight=3.0)
		# EES
		connect(EES, D1_1, delay=1.0, weight=17.0)
		connect(EES, D1_4, delay=1.0, weight=17.0)

		# inner connectomes
		connect(D1_1, D1_2, delay=2, weight=7.0) # 7
		connect(D1_1, D1_3, delay=2, weight=16.0)
		connect(D1_2, D1_1, delay=2, weight=7.0)
		connect(D1_2, D1_3, delay=2, weight=20.0)
		connect(D1_3, D1_1, delay=2, weight=-10.0 * self.inh_coef)
		connect(D1_3, D1_2, delay=2, weight=-10.0 * self.inh_coef)
		connect(D1_4, D1_3, delay=3, weight=-10.0 * self.inh_coef)
		# output to
		connect(D1_3, G1_1, delay=2, weight=30.0)
		connect(D1_3, group1, delay=3.0, weight=30) # 13.5

		connect(group1, group2, delay=2.0, weight=20.0)

		''' D2 '''
		# input from
		connect(C2, D2_1, delay=1.0, weight=4.0)
		connect(C2, D2_4, delay=1.0, weight=4.0)
		connect(C3, D2_1, delay=1.0, weight=4.0)
		connect(C3, D2_4, delay=1.0, weight=4.0)

		connect(group1, D2_1, delay=1.0, weight=6.0)
		connect(group1, D2_4, delay=1.0, weight=6.0)

		# inner connectomes
		connect(D2_1, D2_2, delay=2.0, weight=7.0) # 7
		connect(D2_1, D2_3, delay=2.0, weight=20.0)
		connect(D2_2, D2_1, delay=2.0, weight=7.0)
		connect(D2_2, D2_3, delay=2.0, weight=20.0)
		connect(D2_3, D2_1, delay=2.0, weight=-10.0 * self.inh_coef)
		connect(D2_3, D2_2, delay=2.0, weight=-10.0 * self.inh_coef)
		connect(D2_4, D2_3, delay=3.0, weight=-10.0 * self.inh_coef)
		# output to
		connect(D2_3, G2_1, delay=2.0, weight=30.0)



		''' D3 '''
		# input from
		connect(C3, D3_1, delay=1.0, weight=4.0)
		connect(C3, D3_4, delay=1.0, weight=4.0)
		connect(C4, D3_1, delay=1.0, weight=4.0)
		connect(C4, D3_4, delay=1.0, weight=4.0)

		connect(group2, D3_1, delay=2.0, weight=6.0)
		# inner connectomes
		connect(D3_1, D3_2, delay=2.0, weight=7.0)
		connect(D3_1, D3_3, delay=2.0, weight=20.0)
		connect(D3_2, D3_1, delay=2.0, weight=7.0)
		connect(D3_2, D3_3, delay=2.0, weight=20.0)
		connect(D3_3, D3_1, delay=2.0, weight=-10.0 * self.inh_coef)
		connect(D3_3, D3_2, delay=2.0, weight=-10.0 * self.inh_coef)
		connect(D3_4, D3_3, delay=2.0, weight=-10.0 * self.inh_coef)
		# output to
		#connect(D3_3, G3_1, delay=3.0, weight=30.0)
		# suppression
		connect(D3_3, G1_3, delay=1.0, weight=30.0)

		connect(group2, group3, delay=2.0, weight=20.0)

		''' D4 '''
		# input from
		connect(C4, D4_1, delay=1.0, weight=5.0) # 4
		connect(C4, D4_4, delay=1.0, weight=5.0) # 4
		connect(C5, D4_1, delay=1.0, weight=5.0) # 4
		connect(C5, D4_4, delay=1.0, weight=5.0) # 4

		connect(group3, D4_1, delay=2.0, weight=6.0)
		# inner connectomes
		connect(D4_1, D4_2, delay=2.0, weight=7.0)
		connect(D4_1, D4_3, delay=2.0, weight=20.0)
		connect(D4_2, D4_1, delay=2.0, weight=7.0)
		connect(D4_2, D4_3, delay=2.0, weight=20.0)
		connect(D4_3, D4_1, delay=2.0, weight=-10.0 * self.inh_coef)
		connect(D4_3, D4_2, delay=2.0, weight=-10.0 * self.inh_coef)
		connect(D4_4, D4_3, delay=2.0, weight=-10.0 * self.inh_coef)
		# output to
		#connect(D4_3, G4_1, delay=3.0, weight=20.0)
		# suppression
		connect(D4_3, G2_3, delay=1.0, weight=30.0)

		connect(group3, group4, delay=2.0, weight=20.0)

		''' D5 '''
		# input from
		connect(C5, D5_1, delay=1.0, weight=4.0)
		connect(C5, D5_4, delay=1.0, weight=4.0)
		connect(group4, D5_1, delay=2.0, weight=5.0)
		# inner connectomes
		connect(D5_1, D5_2, delay=2.0, weight=7.0)
		connect(D5_1, D5_3, delay=2.0, weight=20.0)
		connect(D5_2, D5_1, delay=2.0, weight=7.0)
		connect(D5_2, D5_3, delay=2.0, weight=20.0)
		connect(D5_3, D5_1, delay=2.0, weight=-10.0 * self.inh_coef)
		connect(D5_3, D5_2, delay=2.0, weight=-10.0 * self.inh_coef)
		connect(D5_4, D5_3, delay=2.0, weight=-10.0 * self.inh_coef)
		# output to
		#connect(D5_3, G5_1, 4.0, 30.0)
		# suppression
		connect(D5_3, G1_3, delay=1.0, weight=30.0)
		connect(D5_3, G2_3, delay=1.0, weight=30.0)
		#connect(D5_3, G3_3, delay=1.0, weight=30.0)
		#connect(D5_3, G4_3, delay=1.0, weight=30.0)

		''' G1 '''
		# inner connectomes
		connect(G1_1, G1_2, delay=2.0, weight=10.0)
		connect(G1_1, G1_3, delay=1.0, weight=10.0)
		connect(G1_2, G1_1, delay=2.0, weight=10.0)
		connect(G1_2, G1_3, delay=1.0, weight=10.0)
		connect(G1_3, G1_1, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(G1_3, G1_2, delay=1.0, weight=-10.0 * self.inh_coef)
		# output to IP_E
		connect(G1_1, IP_E, delay=2.0, weight=20.0) # 25
		connect(G1_2, IP_E, delay=2.0, weight=20.0) # 25

		#''' G2 '''
		# inner connectomes
		connect(G2_1, G2_2, delay=2.0, weight=10.0)
		connect(G2_1, G2_3, delay=2.0, weight=10.0)
		connect(G2_2, G2_1, delay=2.0, weight=10.0)
		connect(G2_2, G2_3, delay=2.0, weight=10.0)
		connect(G2_3, G2_1, delay=2.0, weight=-30.0 * self.inh_coef)
		connect(G2_3, G2_2, delay=2.0, weight=-30.0 * self.inh_coef)
		# output to IP_E
		connect(G2_1, IP_E, delay=2.0, weight=20.0) # 25
		connect(G2_2, IP_E, delay=2.0, weight=20.0) # 25
#
		#''' G3 '''
		## inner connectomes
		#connect(G3_1, G3_2, delay=2.0, weight=7.0)
		#connect(G3_1, G3_3, delay=2.0, weight=10.0)
		#connect(G3_2, G3_1, delay=2.0, weight=7.0)
		#connect(G3_2, G3_3, delay=2.0, weight=10.0)
		#connect(G3_3, G3_1, delay=1.0, weight=-30.0 * self.inh_coef)
		#connect(G3_3, G3_2, delay=1.0, weight=-30.0 * self.inh_coef)
		## output to IP_E
		#connect(G3_1, IP_E, delay=1.0, weight=30.0)
		#connect(G3_2, IP_E, delay=1.0, weight=30.0)
#
		#''' G4 '''
		## inner connectomes
		#connect(G4_1, G4_2, delay=2.0, weight=5.0)
		#connect(G4_1, G4_3, delay=2.0, weight=10.0)
		#connect(G4_2, G4_1, delay=2.0, weight=5.0)
		#connect(G4_2, G4_3, delay=2.0, weight=10.0)
		#connect(G4_3, G4_1, delay=1.0, weight=-30.0 * self.inh_coef)
		#connect(G4_3, G4_2, delay=1.0, weight=-30.0 * self.inh_coef)
		## output to IP_E
		#connect(G4_1, IP_E, delay=3.0, weight=30.0)
		#connect(G4_2, IP_E, delay=3.0, weight=30.0)
#
		#''' G5 '''
		## inner connectomes
		#connect(G5_1, G5_2, delay=2.0, weight=7.0)
		#connect(G5_1, G5_3, delay=2.0, weight=10.0)
		#connect(G5_2, G5_1, delay=2.0, weight=7.0)
		#connect(G5_2, G5_3, delay=2.0, weight=10.0)
		#connect(G5_3, G5_1, delay=1.0, weight=-30.0 * self.inh_coef)
		#connect(G5_3, G5_2, delay=1.0, weight=-30.0 * self.inh_coef)
		## output to IP_E
		#connect(G5_1, IP_E, delay=5.0, weight=30.0)
		#connect(G5_2, IP_E, delay=5.0, weight=30.0)

		connect(IP_E, MP_E, delay=1, weight=20)
		connect(EES, MP_E, delay=2, weight=30)
		#connect(Ia, MP_E, delay=1, weight=50)


	def connect_spike_generator(self, node, t_start, t_end, rate, offset=0):
		"""

		Args:
			node (tuple or list):
				 GIDs of the neurons
			t_start (int):
				start stimulation time
			t_end (int):
				end stimulation time
			rate (int):
				frequency rate
			offset (int or float):
				time offset
		"""
		# total possible number of spikes without t_start and t_end at current rate
		num_spikes = int(self.sim_time // (1000 / rate))
		# get spike times only in [t_start, t_end) interval
		spike_times = []
		for spike_index in range(num_spikes):
			time = offset + spike_index * round(1000 / rate, 1)
			if t_start <= time < t_end:
				if time == 0:
					time = 0.1
				spike_times.append(time)

		# parameters
		spike_generator_params = {
			'spike_times': spike_times,
			'spike_weights': [450.] * len(spike_times)
		}

		syn_spec = {
			'model': 'static_synapse',
			'weight': 1.0,
			'delay': 0.1
		}

		conn_spec = {
			'rule': 'fixed_outdegree',
			'outdegree': len(node),
			'autapses': False,
			'multapses': False
		}

		# create the spike generator
		spike_generator = Create(model='spike_generator', params=spike_generator_params)
		# connect the spike generator with node
		Connect(pre=spike_generator, post=node, syn_spec=syn_spec, conn_spec=conn_spec)


	def create_with_mmeter(self, name, n=40):
		"""
		Function for creating new neruons
		Args:
			name (str): neurons group name
			n (int): number of neurons
		Returns:
			list: global IDs of created neurons
		"""
		name = "{}-{}".format(self.iteration, name)
		gids = create(n)
		# decrease useless data recording for 'multitest' case
		# if self.multitest and "moto" not in name:
		#	return gids

		mm_id = add_multimeter(name, record_from="V_m")
		sd_id = add_spike_detector(name)
		spikedetectors_dict[name] = sd_id
		multimeters_dict[name] = mm_id

		Connect(pre=mm_id, post=gids)
		Connect(pre=gids, post=sd_id)

		return gids


def rand(a, b):
	return random.uniform(a, b)


def create(neuron_number):
	"""
	Function for creating new neruons without multimeter
	Args:
		neuron_number (int):
			number of neurons
	Returns:
		list: neurons GIDs
	"""
	neuron_model = 'hh_cond_exp_traub'
	neuron_params = {'t_ref': 2.0, #rand(1.9, 2.1) if self.multitest else 2.0,  # [ms] refractory period
	                 'V_m': -70.0,  # [mV] starting value of membrane potential
	                 'E_L': -70.0, #rand(-70.5, -69.5) if self.multitest else -70.0,  # [mV] leak reversal potential
	                 'g_L': 75.0,  # [nS] leak conductance
	                 'tau_syn_ex': 0.2,  # [ms]
	                 'tau_syn_in': 3.0}  # [ms]
	neuron_ids = Create(model=neuron_model, n=neuron_number, params=neuron_params)
	return neuron_ids


def connect(pre, post, weight=0, delay=1.0, syn_spec=None, degree=40):
	"""
	Connect group of neurons
	Args:
		syn_spec (dict): STDP synapse specifications, None - for default static_synapse
		pre (list): pre neurons GIDs
		post (list): pre neurons GIDs
		weight (float): synaptic strength
		degree (int): number of outgoing synapses from PRE neurons
		delay (float): synaptic delay
	"""
	# initialize synapse specification
	if syn_spec:
		syn_spec['model'] = 'stdp_synapse'
		syn_spec['delay'] = float(delay)
		syn_spec['weight'] = float(weight)
	else:
		syn_spec = {'model': 'static_synapse',
		            'delay': float(delay),
		            'weight': float(weight)}
	# initialize connection specification
	conn_spec = {'rule': 'fixed_outdegree',  # fixed outgoing synapse number
	             'outdegree': int(degree),  # number of synapses outgoing from PRE neuron
	             'multapses': True,  # allow recurring connections
	             'autapses': False}  # allow self-connection
	# NEST connection
	Connect(pre=pre, post=post, syn_spec=syn_spec, conn_spec=conn_spec)
