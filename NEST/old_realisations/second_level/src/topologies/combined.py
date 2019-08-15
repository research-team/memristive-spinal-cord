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


class Topology:
	def __init__(self, simulation_params, test_iteration=0):
		self.iteration = test_iteration
		self.multitest = simulation_params[Params.MULTITEST.value]
		self.inh_coef = simulation_params[Params.INH_COEF.value]
		self.sim_time = simulation_params[Params.SIM_TIME.value]
		self.ees_rate = simulation_params[Params.EES_RATE.value]
		self.speed = simulation_params[Params.SPEED.value]
		self.c_time = simulation_params[Params.C_TIME.value]

		neurons_in_moto = 169

		C1 = self.create_with_mmeter("C1")
		EES = self.create_with_mmeter("EES")
		OM1_0_E = self.create_with_mmeter("OM1_0_E")
		OM1_0_F = self.create_with_mmeter("OM1_0_F")
		OM1_1 = self.create_with_mmeter("OM1_1")
		OM1_2 = self.create_with_mmeter("OM1_2")
		OM1_3 = self.create_with_mmeter("OM1_3")
		OM2_0_E = self.create_with_mmeter("OM2_0_E")
		OM2_0_F = self.create_with_mmeter("OM2_0_F")
		OM2_1 = self.create_with_mmeter("OM2_1")
		OM2_2 = self.create_with_mmeter("OM2_2")
		OM2_3 = self.create_with_mmeter("OM2_3")
		OM3_0 = self.create_with_mmeter("OM3_0")
		OM3_1 = self.create_with_mmeter("OM3_1")
		OM3_2_E = self.create_with_mmeter("OM3_2_E")
		OM3_2_F = self.create_with_mmeter("OM3_2_F")
		OM3_3 = self.create_with_mmeter("OM3_3")
		OM4_0_E = self.create_with_mmeter("OM4_0_E")
		OM4_0_F = self.create_with_mmeter("OM4_0_F")
		OM4_1 = self.create_with_mmeter("OM4_1")
		OM4_2 = self.create_with_mmeter("OM4_2")
		OM4_3 = self.create_with_mmeter("OM4_3")
		OM5_0 = self.create_with_mmeter("OM5_0")
		OM5_1 = self.create_with_mmeter("OM5_1")
		OM5_2 = self.create_with_mmeter("OM5_2")
		OM5_3 = self.create_with_mmeter("OM5_3")
		IP_E = self.create_with_mmeter("IP_E", neurons_in_ip)
		IP_F = self.create_with_mmeter("IP_F", neurons_in_ip)
		MP_E = self.create_with_mmeter("MP_E", neurons_in_moto)
		MP_F = self.create_with_mmeter("MP_F", neurons_in_moto)
		Ia_Extensor = self.create_with_mmeter("Ia_Extensor", neurons_in_afferent)
		Ia_Flexor = self.create_with_mmeter("Ia_Flexor", neurons_in_afferent)
		E1 = self.create_with_mmeter("E1")
		E2 = self.create_with_mmeter("E2")
		E3 = self.create_with_mmeter("E3")
		E4 = self.create_with_mmeter("E4")
		E5 = self.create_with_mmeter("E5")
		R_E = self.create_with_mmeter("R_E")
		R_F = self.create_with_mmeter("R_F")
		Ia_E = self.create_with_mmeter("Ia_E")
		Ia_F = self.create_with_mmeter("Ia_F")
		Ib_E = self.create_with_mmeter("Ib_E")
		Ib_F = self.create_with_mmeter("Ib_F")
		CV1 = self.create_with_mmeter("CV1", 1)
		CV2 = self.create_with_mmeter("CV2", 1)
		CV3 = self.create_with_mmeter("CV3", 1)
		CV4 = self.create_with_mmeter("CV4", 1)
		CV5 = self.create_with_mmeter("CV5", 1)
		CD4 = self.create_with_mmeter("CD4", 1)
		CD5 = self.create_with_mmeter("CD5", 1)
		C_0 = self.create_with_mmeter("C_0", 1)
		C_1 = self.create_with_mmeter("C_1", 1)

		self.connect_spike_generator(CV1, t_start=0, t_end=self.c_time, rate=200)
		self.connect_spike_generator(CV2, t_start=self.c_time, t_end=self.c_time*2, rate=200)
		self.connect_spike_generator(CV3, t_start=self.c_time*2, t_end=self.c_time*3, rate=200)
		self.connect_spike_generator(CV4, t_start=self.c_time*3, t_end=self.c_time*5, rate=200)
		self.connect_spike_generator(CV5, t_start=self.c_time*5, t_end=self.c_time*6, rate=200)
		self.connect_spike_generator(EES, t_start=0, t_end=self.sim_time, rate=self.ees_rate)

		# input from EES
		connect(EES, E1, 2, 500)
		connect(E1, E2, 2, 200)
		connect(E2, E3, 2, 200)
		connect(E3, E4, 3, 200)
		connect(E4, E5, 3, 200)

		# OM 1
		# input from EES group 1
		connect(E1, OM1_0_E, 3, 7)  # ToDo: EXTENSOR
		connect(E1, OM1_0_F, 1, 15)  # ToDo: FLEXOR
		# input from sensory
		connect_one_to_all(CV1, OM1_0_E, 0.5, 18)
		connect_one_to_all(CV2, OM1_0_E, 0.5, 18)
		# [inhibition]
		connect_one_to_all(CV3, OM1_3, 1, 80)
		connect_one_to_all(CV4, OM1_3, 1, 80)
		connect_one_to_all(CV5, OM1_3, 1, 80)
		# inner connectomes
		connect(OM1_0_E, OM1_1, 1, 50)  # ToDo: EXTENSOR
		connect(OM1_0_F, OM1_1, 1, 50)  # ToDo: FLEXOR
		connect(OM1_1, OM1_2, 1, 24)  # 23
		connect(OM1_1, OM1_3, 1, 3)
		connect(OM1_2, OM1_1, 2.5, 23)  # 22
		connect(OM1_2, OM1_3, 1, 3)
		connect(OM1_3, OM1_1, 1, -70 * INH_COEF)
		connect(OM1_3, OM1_2, 1, -70 * INH_COEF)
		# output to OM2, ToDo: FLEXOR
		connect(OM1_0_F, OM2_2, 1, 50)
		# output to IP
		connect(OM1_2, IP_E, 1, 15, neurons_in_ip)
		connect(OM1_2, IP_F, 3, 2, neurons_in_ip)

		# OM 2
		# input from EES group 2
		connect(E2, OM2_0_E, 3, 7)  # ToDo: EXTENSOR
		connect(E2, OM2_0_F, 1, 15)  # ToDo: FLEXOR
		# input from sensory [CV]
		connect_one_to_all(CV2, OM2_0_E, 0.5, 18)
		connect_one_to_all(CV3, OM2_0_E, 0.5, 18)
		# [inhibition]
		connect_one_to_all(CV4, OM2_3, 1, 80)
		connect_one_to_all(CV5, OM2_3, 1, 80)
		# inner connectomes
		connect(OM2_0_E, OM2_1, 1, 50)  # ToDo: EXTENSOR
		connect(OM2_0_F, OM2_1, 1, 10)  # ToDo: FLEXOR
		connect(OM2_1, OM2_2, 1, 23)
		connect(OM2_1, OM2_3, 1, 3)
		connect(OM2_2, OM2_1, 2.5, 22)
		connect(OM2_2, OM2_3, 1, 3)
		connect(OM2_3, OM2_1, 1, -70 * INH_COEF)
		connect(OM2_3, OM2_2, 1, -70 * INH_COEF)
		# output to OM3, ToDo: FLEXOR
		connect(OM2_0_F, OM3_0, 1, 50)
		# output to IP
		connect(OM2_2, IP_E, 2, 15, neurons_in_ip)  # 50
		connect(OM2_2, IP_F, 2, 3, neurons_in_ip)

		# OM 3
		# input from EES group 3
		connect(E3, OM3_0, 3, 7)
		# input from sensory [CV]
		connect_one_to_all(CV3, OM3_0, 0.5, 18)
		connect_one_to_all(CV4, OM3_0, 0.5, 18)
		# [INH]
		connect_one_to_all(CV5, OM3_3, 1, 80)
		# input from sensory [CD]
		connect_one_to_all(CD4, OM3_0, 1, 11)
		# inner connectomes
		connect(OM3_0, OM3_1, 1, 50)
		connect(OM3_1, OM3_2_E, 1, 24)  # ToDo: EXTENSOR
		connect(OM3_1, OM3_2_F, 1, 30)  # ToDo: FLEXOR
		connect(OM3_1, OM3_3, 1, 3)
		connect(OM3_2_E, OM3_1, 2.5, 22)  # ToDo: EXTENSOR
		connect(OM3_2_F, OM3_1, 2.5, 40)  # ToDo: FLEXOR
		connect(OM3_2_E, OM3_3, 1, 3)  # ToDo: EXTENSOR
		connect(OM3_2_F, OM3_3, 1, 3)  # ToDo: FLEXOR
		connect(OM3_3, OM3_1, 1, -5 * INH_COEF)
		connect(OM3_3, OM3_2_E, 1, -10 * INH_COEF)  # ToDo: EXTENSOR !!!
		connect(OM3_3, OM3_2_F, 1, -0.1 * INH_COEF)  # ToDo: FLEXOR
		# output to OM3, ToDo: FLEXOR
		connect(OM3_2_F, OM4_2, 1, 50)
		connect(OM3_2_E, IP_E, 3, 15, neurons_in_ip)  # ToDo: EXTENSOR
		connect(OM3_2_F, IP_F, 3, 6, neurons_in_ip)  # ToDo: FLEXOR

		# OM 4
		# input from EES group 4
		connect(E4, OM4_0_E, 3, 7)  # ToDo: EXTENSOR
		connect(E4, OM4_0_F, 1, 15)  # ToDo: FLEXOR
		# input from sensory [CV]
		connect_one_to_all(CV4, OM4_0_E, 0.5, 18)
		connect_one_to_all(CV5, OM4_0_E, 0.5, 18)
		# [INH]
		connect_one_to_all(CV5, OM4_3, 1, 80)
		# input from sensory [CD]
		connect_one_to_all(CD4, OM4_0_E, 1, 11)
		connect_one_to_all(CD5, OM4_0_E, 1, 11)
		# inner connectomes
		connect(OM4_0_E, OM4_1, 3, 50)  # ToDo: EXTENSOR
		connect(OM4_0_F, OM4_1, 3, 50)  # ToDo: FLEXOR
		connect(OM4_1, OM4_2, 1, 23)
		connect(OM4_1, OM4_3, 1, 3)
		connect(OM4_2, OM4_1, 2.5, 22)
		connect(OM4_2, OM4_3, 1, 3)
		connect(OM4_3, OM4_1, 1, -70 * INH_COEF)
		connect(OM4_3, OM4_2, 1, -70 * INH_COEF)
		# output to OM4
		connect(OM4_0_F, OM5_0, 1, 50)
		connect(OM4_2, IP_E, 3, 13, neurons_in_ip)
		connect(OM4_2, IP_F, 1, 6, neurons_in_ip)

		# OM 5
		# input from EES group 5
		connect(E5, OM5_0, 3, 7)
		# input from sensory [CV]
		connect_one_to_all(CV5, OM5_0, 0.5, 18)
		# input from sensory [CD]
		connect_one_to_all(CD5, OM5_0, 1, 11)
		# inner connectomes
		connect(OM5_0, OM5_1, 1, 50)
		connect(OM5_1, OM5_2, 1, 23)
		connect(OM5_1, OM5_3, 1, 3)
		connect(OM5_2, OM5_1, 2.5, 22)
		connect(OM5_2, OM5_3, 1, 3)
		connect(OM5_3, OM5_1, 1, -70 * INH_COEF)
		connect(OM5_3, OM5_2, 1, -70 * INH_COEF)
		# output to IP
		connect(OM5_2, IP_E, 2, 15, neurons_in_ip)
		connect(OM5_2, IP_F, 3, 3, neurons_in_ip)

		# inhibition by C=0: IP_E, Ia_Extensor
		connect_one_to_all(C_0, IP_E, 0.1, -g_bar)
		connect_one_to_all(C_0, Ia_Extensor, 0.1, -g_bar)
		# inhibition by C=0: extensor clones D1, D2, G3, D4
		connect_one_to_all(C_0, OM1_0_E, 0.1, -0.1)
		connect_one_to_all(C_0, OM2_0_E, 0.1, -g_bar)
		connect_one_to_all(C_0, OM3_2_E, 0.1, -g_bar)
		connect_one_to_all(C_0, OM4_0_E, 0.1, -g_bar)

		# inhibition by C=1: IP_F, Ia_Flexor
		connect_one_to_all(C_1, IP_F, 0.1, -g_bar)
		connect_one_to_all(C_1, Ia_Flexor, 0.1, -g_bar)
		# inhibition by C=0: flexor clones D1, D2, G3, D4
		connect_one_to_all(C_1, OM1_0_F, 0.1, -10)
		connect_one_to_all(C_1, OM2_0_F, 0.1, -g_bar)
		connect_one_to_all(C_1, OM3_2_F, 0.1, -g_bar)
		connect_one_to_all(C_1, OM4_0_F, 0.1, -g_bar)

		# reflex arc
		connect(EES, Ia_Extensor, 1, 500)
		connect(EES, Ia_Flexor, 1, 500)

		connect(IP_E, MP_E, 1, 2, neurons_in_moto)  # was 30
		connect(IP_E, Ia_E, 2.0, 20.0)
		connect(MP_E, R_E, 2.0, 20.0)

		connect(IP_F, MP_F, 1, 5, neurons_in_moto)
		connect(IP_F, Ia_F, 2.0, 20.0)
		connect(MP_F, R_F, 2.0, 20.0)

		connect(Ib_F, Ib_E, 2.0, -20 * INH_COEF)
		connect(Ib_F, MP_F, 2.0, -20 * INH_COEF)
		connect(Ib_E, Ib_F, 2.0, -20 * INH_COEF)
		connect(Ib_E, MP_E, 2.0, -5 * INH_COEF)

		connect(Ia_F, Ia_E, 2.0, -20 * INH_COEF)
		connect(Ia_F, MP_E, 2.0, -5 * INH_COEF)
		connect(Ia_E, Ia_F, 2.0, -20 * INH_COEF)
		connect(Ia_E, MP_F, 2.0, -20 * INH_COEF)

		connect(R_F, R_E, 2.0, -20 * INH_COEF)
		connect(R_F, Ia_F, 2.0, -20 * INH_COEF)
		connect(R_F, MP_F, 2.0, -20 * INH_COEF)

		connect(R_E, R_F, 2.0, -20 * INH_COEF)
		connect(R_E, Ia_E, 2.0, -20 * INH_COEF)
		connect(R_E, MP_E, 2.0, -5 * INH_COEF)

		connect(Ia_Flexor, MP_F, 1, 5, neurons_in_moto)
		connect(Ia, Ia_F, 1.0, 10.0)
		connect(Ia, Ib_F, 1.0, 10.0)

		connect(Ia_Extensor, MP_E, 1, 5, neurons_in_moto)
		connect(Ia, Ia_E, 1.0, 10.0)
		connect(Ia, Ib_E, 1.0, 10.0)


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
		if self.multitest and "MP_E" not in name:
			return gids

		mm_id = add_multimeter(name, record_from="V_m")
		sd_id = add_spike_detector(name)
		spikedetectors_dict[name] = sd_id
		multimeters_dict[name] = mm_id

		Connect(pre=mm_id, post=gids)
		Connect(pre=gids, post=sd_id)

		return gids


def rand(a, b):
	return random.uniform(a, b)


def __build_params():
	neuron_params = {'t_ref': rand(2.0, 4.0),
	                 'V_m': -70.0,  # [mV] starting value of membrane potential
	                 'E_L': rand(-75.0, -65.0),
	                 'g_L': 75.0,  # [nS] leak conductance
	                 'tau_syn_ex': rand(0.2, 0.35),  # [ms]
	                 'tau_syn_in': rand(2.5, 3.5),  # [ms]
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
	# initialize synapse specification
	weight_median = 10  # 3 10
	delay_median = 0.5 if delay > 1 else 0.25
	# initialize synapse specification
	if syn_spec:
		syn_spec['model'] = 'stdp_synapse'
		syn_spec['delay'] = {"distribution": "uniform",
		                     "low": float(delay) - delay_median,
		                     "high": float(delay) + delay_median}
		syn_spec['weight'] = {"distribution": "uniform",
		                      "low": float(weight) - weight_median,
		                      "high": float(weight) + weight_median}
	else:
		syn_spec = {'model': 'static_synapse',
		            'delay': {"distribution": "uniform",
		                      "low": float(delay) - delay_median,
		                      "high": float(delay) + delay_median},
		            'weight': {"distribution": "uniform",
		                       "low": float(weight) - weight_median,
		                       "high": float(weight) + weight_median}
		            }
	# initialize connection specification
	conn_spec = {'rule': 'fixed_outdegree',  # fixed outgoing synapse number
	             'outdegree': int(degree),  # number of synapses outgoing from PRE neuron
	             'multapses': True,  # allow recurring connections
	             'autapses': False}  # allow self-connection
	# NEST connection
	Connect(pre=pre, post=post, syn_spec=syn_spec, conn_spec=conn_spec)
