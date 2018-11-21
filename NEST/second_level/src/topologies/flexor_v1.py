import random
from nest import Create, Connect, CopyModel
from NEST.second_level.src.data import *
from NEST.second_level.src.tools.multimeter import add_multimeter
from NEST.second_level.src.tools.spike_detector import add_spike_detector
from NEST.second_level.src.namespace import *

'''
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
'''

class Topology:
	def __init__(self, simulation_params, test_iteration=0):
		self.iteration = test_iteration
		self.multitest = simulation_params[Params.MULTITEST.value]
		self.inh_coef = simulation_params[Params.INH_COEF.value]
		self.sim_time = simulation_params[Params.SIM_TIME.value]
		self.ees_rate = simulation_params[Params.EES_RATE.value]
		self.speed = simulation_params[Params.SPEED.value]
		self.c_time = simulation_params[Params.C_TIME.value]
		self.record_from = simulation_params[Params.RECORD_FROM.value]

		neurons_in_moto = 169

		ees_group1 = self.create_with_mmeter("ees_group1")
		ees_group2 = self.create_with_mmeter("ees_group2")
		ees_group3 = self.create_with_mmeter("ees_group3")
		ees_group4 = self.create_with_mmeter("ees_group4")

		inh_group5 = self.create_with_mmeter("inh_group5")

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

		G3_1 = self.create_with_mmeter("G3_1")
		G3_2 = self.create_with_mmeter("G3_2")
		G3_3 = self.create_with_mmeter("G3_3")

		G4_1 = self.create_with_mmeter("G4_1")
		G4_2 = self.create_with_mmeter("G4_2")
		G4_3 = self.create_with_mmeter("G4_3")

		G5_1 = self.create_with_mmeter("G5_1")
		G5_2 = self.create_with_mmeter("G5_2")
		G5_3 = self.create_with_mmeter("G5_3")

		IP_F = self.create_with_mmeter("IP_F", neurons_in_moto)
		MP_F = self.create_with_mmeter("MP_F", neurons_in_moto)

		EES = self.create_with_mmeter("EES")


		self.connect_spike_generator(EES, t_start=0, t_end=self.sim_time, rate=self.ees_rate)

		connect(inh_group5, G1_3, delay=1.0, weight=20.0)
		connect(inh_group5, G2_3, delay=1.0, weight=20.0)
		connect(inh_group5, G3_3, delay=1.0, weight=20.0)
		connect(inh_group5, G4_3, delay=1.0, weight=20.0)

		''' D1 '''
		# input from EES
		connect(EES, D1_1, delay=1.0, weight=17.0)  # 17 Threshold / 7 ST
		connect(EES, D1_4, delay=1.0, weight=17.0)  # 17 Threshold / 7 ST
		# inner connectomes
		connect(D1_1, D1_2, delay=1, weight=7.0)
		connect(D1_1, D1_3, delay=1, weight=16.0)
		connect(D1_2, D1_1, delay=1, weight=7.0)
		connect(D1_2, D1_3, delay=1, weight=20.0)
		connect(D1_3, D1_1, delay=1, weight=-10.0 * self.inh_coef)
		connect(D1_3, D1_2, delay=1, weight=-10.0 * self.inh_coef)
		connect(D1_4, D1_3, delay=2, weight=-10.0 * self.inh_coef)
		# output to G1
		connect(D1_3, G1_1, delay=3, weight=12.5)
		# output to G2
		connect(D1_3, G2_1, delay=3, weight=12.5)
		# output to EES group
		connect(D1_3, ees_group1, delay=1.0, weight=30)

		# EES group connectomes
		connect(ees_group1, ees_group2, delay=1.0, weight=20.0)

		''' D2 '''
		# input from Group (1)
		connect(ees_group1, D2_1, delay=1.0, weight=5.0)
		connect(ees_group1, D2_4, delay=1.0, weight=5.0)
		# inner connectomes
		connect(D2_1, D2_2, delay=1.0, weight=7.0)
		connect(D2_1, D2_3, delay=1.0, weight=20.0)
		connect(D2_2, D2_1, delay=1.0, weight=7.0)
		connect(D2_2, D2_3, delay=1.0, weight=20.0)
		connect(D2_3, D2_1, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(D2_3, D2_2, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(D2_4, D2_3, delay=2.0, weight=-10.0 * self.inh_coef)
		# output to D3
		connect(D2_3, D3_1, delay=1.0, weight=12.5)
		connect(D2_3, D3_4, delay=1.0, weight=12.5)

		# EES group connectomes
		connect(ees_group2, ees_group3, delay=1.0, weight=20.0)

		''' D3 '''
		# input from Group (2)
		connect(ees_group2, D3_1, delay=1, weight=6.0)
		connect(ees_group2, D3_4, delay=1, weight=6.0)
		# inner connectomes
		connect(D3_1, D3_2, delay=1.0, weight=7.0)
		connect(D3_1, D3_3, delay=1.0, weight=20.0)
		connect(D3_2, D3_1, delay=1.0, weight=7.0)
		connect(D3_2, D3_3, delay=1.0, weight=20.0)
		connect(D3_3, D3_1, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(D3_3, D3_2, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(D3_4, D3_3, delay=2.0, weight=-10.0 * self.inh_coef)
		# output to generator
		connect(D3_3, G3_1, delay=1.5, weight=30.0)

		# EES group connectomes
		connect(ees_group3, ees_group4, delay=2.0, weight=20.0)

		''' D4 '''
		# input from Group (3)
		connect(ees_group3, D4_1, delay=2.0, weight=6.0)
		connect(ees_group3, D4_4, delay=2.0, weight=6.0)
		# inner connectomes
		connect(D4_1, D4_2, delay=1.0, weight=7.0)
		connect(D4_1, D4_3, delay=1.0, weight=20.0)
		connect(D4_2, D4_1, delay=1.0, weight=7.0)
		connect(D4_2, D4_3, delay=1.0, weight=20.0)
		connect(D4_3, D4_1, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(D4_3, D4_2, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(D4_4, D4_3, delay=2.0, weight=-10.0 * self.inh_coef)

		''' D5 '''
		# input from Group (4)
		connect(ees_group4, D5_1, delay=2.0, weight=5.0)
		connect(ees_group4, D5_4, delay=2.0, weight=5.0)
		# inner connectomes
		connect(D5_1, D5_2, delay=1.0, weight=7.0)
		connect(D5_1, D5_3, delay=1.0, weight=20.0)
		connect(D5_2, D5_1, delay=1.0, weight=7.0)
		connect(D5_2, D5_3, delay=1.0, weight=20.0)
		connect(D5_3, D5_1, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(D5_3, D5_2, delay=1.0, weight=-10.0 * self.inh_coef)
		connect(D5_4, D5_3, delay=2.0, weight=-10.0 * self.inh_coef)
		# output to the generator
		connect(D5_3, G5_1, delay=1.0, weight=30.0)

		''' G1 '''
		# inner connectomes
		connect(G1_1, G1_2, delay=1.0, weight=10.0)
		connect(G1_1, G1_3, delay=1.0, weight=10.0)
		connect(G1_2, G1_1, delay=1.0, weight=10.0)
		connect(G1_2, G1_3, delay=1.0, weight=10.0)
		connect(G1_3, G1_1, delay=0.5, weight=-20.0 * self.inh_coef)
		connect(G1_3, G1_2, delay=0.5, weight=-20.0 * self.inh_coef)
		# output to IP_E
		connect(G1_1, IP_F, delay=3, weight=25.0) # 18 normal
		connect(G1_2, IP_F, delay=3, weight=25.0) # 18 normal

		''' G2 '''
		# inner connectomes
		connect(G2_1, G2_2, delay=1.0, weight=10.0)
		connect(G2_1, G2_3, delay=1.0, weight=20.0)
		connect(G2_2, G2_1, delay=1.0, weight=10.0)
		connect(G2_2, G2_3, delay=1.0, weight=20.0)
		connect(G2_3, G2_1, delay=0.5, weight=-30.0 * self.inh_coef)
		connect(G2_3, G2_2, delay=0.5, weight=-30.0 * self.inh_coef)
		# output to IP_E
		connect(G2_1, IP_F, delay=1.0, weight=65.0) # 35 normal
		connect(G2_2, IP_F, delay=1.0, weight=65.0) # 35 normal
		# output to D2
		connect(G2_1, D2_1, delay=1.0, weight=65.0) # 35 normal
		connect(G2_2, D2_1, delay=1.0, weight=65.0) # 35 normal
		connect(G2_1, D2_4, delay=1.0, weight=65.0) # 35 normal
		connect(G2_2, D2_4, delay=1.0, weight=65.0) # 35 normal

		''' G3 '''
		# inner connectomes
		connect(G3_1, G3_2, delay=1.0, weight=12.0)
		connect(G3_1, G3_3, delay=1.0, weight=20.0)
		connect(G3_2, G3_1, delay=1.0, weight=12.0)
		connect(G3_2, G3_3, delay=1.0, weight=20.0)
		connect(G3_3, G3_1, delay=0.5, weight=-30.0 * self.inh_coef)
		connect(G3_3, G3_2, delay=0.5, weight=-30.0 * self.inh_coef)
		# output to IP_F
		connect(G3_1, IP_F, delay=2, weight=25.0)   # 20 normal
		connect(G3_2, IP_F, delay=2, weight=25.0)   # 20 normal
		# output to G4
		connect(G3_1, G4_1, delay=1.0, weight=65.0)  # 35 normal
		connect(G3_2, G4_1, delay=1.0, weight=65.0)  # 35 normal

		''' G4 '''
		# inner connectomes
		connect(G4_1, G4_2, delay=1.0, weight=10.0)
		connect(G4_1, G4_3, delay=1.0, weight=10.0)
		connect(G4_2, G4_1, delay=1.0, weight=5.0)
		connect(G4_2, G4_3, delay=1.0, weight=10.0)
		connect(G4_3, G4_1, delay=0.5, weight=-30.0 * self.inh_coef)
		connect(G4_3, G4_2, delay=0.5, weight=-30.0 * self.inh_coef)
		# output to IP_F
		connect(G4_1, IP_F, delay=1.0, weight=17.0)
		connect(G4_2, IP_F, delay=1.0, weight=17.0)
		# output to D4
		connect(G4_1, D4_1, delay=1.0, weight=65.0)  # 35 normal
		connect(G4_2, D4_1, delay=1.0, weight=65.0)  # 35 normal
		connect(G4_1, D4_4, delay=1.0, weight=65.0)  # 35 normal
		connect(G4_2, D4_4, delay=1.0, weight=65.0)  # 35 normal

		''' G5 '''
		# inner connectomes
		connect(G5_1, G5_2, delay=1.0, weight=7.0)
		connect(G5_1, G5_3, delay=1.0, weight=10.0)
		connect(G5_2, G5_1, delay=1.0, weight=7.0)
		connect(G5_2, G5_3, delay=1.0, weight=10.0)
		connect(G5_3, G5_1, delay=0.5, weight=-30.0 * self.inh_coef)
		connect(G5_3, G5_2, delay=0.5, weight=-30.0 * self.inh_coef)
		# output to IP_F
		connect(G5_1, IP_F, delay=1.0, weight=20.0) # normal 18
		connect(G5_2, IP_F, delay=1.0, weight=20.0) # normal 18
		# output to (inh 5)
		connect(G5_1, inh_group5, delay=1.0, weight=20.0)
		connect(G5_2, inh_group5, delay=1.0, weight=20.0)

		# ref arc
		connect(IP_F, MP_F, delay=1, weight=11) # 14
		connect(EES, MP_F, delay=2, weight=50)


	def connect_spike_generator(self, node, t_start, t_end, rate, offset=0):
		"""
		Create and connect spikegenerator to the node
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
			'spike_weights': [450.0] * len(spike_times)
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
		if self.multitest and "MP_F" not in name:
			return gids

		mm_id = add_multimeter(name, record_from=self.record_from)
		sd_id = add_spike_detector(name)
		spikedetectors_dict[name] = sd_id
		multimeters_dict[name] = mm_id

		Connect(pre=mm_id, post=gids)
		Connect(pre=gids, post=sd_id)

		return gids


def rand(a, b):
	return random.uniform(a, b)


def __build_params():
	neuron_params = {'t_ref': 2.0, #rand(1.5, 2.5),  # if self.multitest else 2.0,  # [ms] refractory period
	                 'V_m': -70.0,  # [mV] starting value of membrane potential
	                 'E_L': -70.0, #rand(-75.0, -65.0),  # if self.multitest else -70.0,  # [mV] leak reversal potential
	                 'g_L': 75.0,  # [nS] leak conductance
	                 'tau_syn_ex': 0.2,  # [ms]
	                 'tau_syn_in': 3.0,  # [ms]
	                 'C_m': rand(150.0, 250.0)}
	return neuron_params


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
	neuron_ids = [Create(model=neuron_model, n=1, params=__build_params())[0] for _ in range(neuron_number)]

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
	med = 3
	# initialize synapse specification
	if syn_spec:
		syn_spec['model'] = 'stdp_synapse'
		syn_spec['delay'] = float(delay)
		syn_spec['weight'] = {"distribution": "uniform",
		                      "low": float(weight) - med,
		                      "high": float(weight) + med}
	else:
		syn_spec = {'model': 'static_synapse',
		            'delay': float(delay),
		            'weight': {"distribution": "uniform",
		                       "low": float(weight) - med,
		                       "high": float(weight) + med}
		            }
	# initialize connection specification
	conn_spec = {'rule': 'fixed_outdegree',  # fixed outgoing synapse number
	             'outdegree': int(degree),  # number of synapses outgoing from PRE neuron
	             'multapses': True,  # allow recurring connections
	             'autapses': False}  # allow self-connection
	# NEST connection
	Connect(pre=pre, post=post, syn_spec=syn_spec, conn_spec=conn_spec)
