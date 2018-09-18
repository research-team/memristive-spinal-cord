import random
from enum import Enum
from nest import Create, Connect, CopyModel
from ..tools.multimeter import add_multimeter
from ..tools.spike_detector import add_spike_detector
from ..data import *

delay_module_number = 0
generator_module_number = 0
subthreshold_module_number = 0
monovibrator_module_number = 0

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


class Params(Enum):
	NUM_SUBLEVELS = 6
	NUM_SPIKES = 6
	RATE = 40
	SIMULATION_TIME = 125.
	INH_COEF = 1.


class EES:
	def __init__(self):
		self.ees = Create(
			model='spike_generator',
			params={
				'spike_times': [10. + i * round(1000. / Params.RATE.value, 1) for i in range(Params.NUM_SPIKES.value)],
				'spike_weights': [500. for _ in range(Params.NUM_SPIKES.value)]
			})

	def connect(self, post):
		"""
		Args:
			post:
		"""
		Connect(pre=self.ees,
		        post=post,
		        syn_spec={'model': 'static_synapse',
		                  'weight': 1.,
		                  'delay': .1},
		        conn_spec={'rule': 'fixed_outdegree',
		                   'outdegree': len(post),
		                   'autapses': False,
		                   'multapses': False}
		        )



class Delay:
	def __init__(self, time=None):
		global delay_module_number
		node1 = create_with_mmeter('Delay_N{}_node1'.format(delay_module_number))
		node2 = create_with_mmeter('Delay_N{}_node2'.format(delay_module_number))
		node3 = create_with_mmeter('Delay_N{}_node3'.format(delay_module_number))
		node4 = create_with_mmeter('Delay_N{}_node3'.format(delay_module_number))

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
		node1 = create_with_mmeter('Generator_N{}_node1'.format(generator_module_number))
		node2 = create_with_mmeter('Generator_N{}_node2'.format(generator_module_number))
		node3 = create_with_mmeter('Generator_N{}_node3'.format(generator_module_number))

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
		node1 = create_with_mmeter('Subthreshold_N{}_node1'.format(subthreshold_module_number))
		node2 = create_with_mmeter('Subthreshold_N{}_node2'.format(subthreshold_module_number))
		node3 = create_with_mmeter('Subthreshold_N{}_node3'.format(subthreshold_module_number))

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


class Monovibrator:
	def __init__(self):
		global monovibrator_module_number
		node1 = create_with_mmeter('Monovibrator_N{}_node1'.format(monovibrator_module_number))
		node2 = create_with_mmeter('Monovibrator_N{}_node2'.format(monovibrator_module_number))

		connect(node2, node1, weight=-10, delay=2)

		monovibrator_module_number += 1

	@property
	def output(self):
		return self.node1

	@property
	def input(self):
		return self.node1 + self.node2


class Topology:
	def __init__(self, multitest=False, test_iteration=0):
		self.multitest = multitest
		self.test_iteration = test_iteration
		sensory = create_with_mmeter('sensory', 60)
		ia_aff = create_with_mmeter('ia_aff', 169)
		pool = [create_with_mmeter('pool{}'.format(i)) for i in range(1, 7)]
		moto = create_with_mmeter('moto', 169)
		ees = EES()
		ees.connect(sensory)
		ees.connect(moto)

		for pool_nucleus in pool:
			connect(pool_nucleus, moto, weight=25, degree=20)
		connect(ia_aff, moto, weight=25, degree=20)

		D1 = Delay()

		S2 = Subthreshold()
		S3 = Subthreshold()
		S4 = Subthreshold()
		S5 = Subthreshold()

		G1 = Generator()
		G2 = Generator()
		G3 = Generator()
		G4 = Generator()
		G5 = Generator()

		M1 = Monovibrator()

		connect(sensory, D1.input, syn_spec=HEBBIAN, weight=10, delay=2)
		connect(D1.output, G1.input, weight=10, syn_spec=HEBBIAN, delay=2)
		connect(G1.output, pool[0], weight=100, delay=2)


		connect(sensory, node11, weight=10.)

		connect(node53, node35, Params.INH_COEF.value * -30, 60, delay=1.5)  # weight = -30 de 0.1
		connect(node53, node37, Params.INH_COEF.value * -30, 60, delay=1.5)  # weight = -30 de 0.1

		connect(node64, pool[5], weight=80)
		connect(node65, pool[5], weight=80)


def rand(a, b):
	return random.uniform(a, b)


def create(neuron_number):
	"""
	Function for creating new neruons without multimeter
	Args:
		neuron_number (int): number of neurons
	Returns:
		list: neurons GIDs
	"""
	neuron_model = 'hh_cond_exp_traub'
	neuron_params = {'t_ref': rand(1.9, 2.1) if self.multitest else 2.0,  # [ms] refractory period
	                 'V_m': -70.0,  # [mV] starting value of membrane potential
	                 'E_L': rand(-70.5, -69.5) if self.multitest else -70.0,  # [mV] leak reversal potential
	                 'g_L': 75.0,  # [nS] leak conductance
	                 'tau_syn_ex': 0.2,  # [ms]
	                 'tau_syn_in': 3.0}  # [ms]
	return Create(model=neuron_model, n=neuron_number, params=neuron_params)


def create_with_mmeter(name, n=40):
	"""
	Function for creating new neruons
	Args:
		name (str): neurons group name
		n (int): number of neurons
	Returns:
		list: global IDs of created neurons
	"""
	if self.multitest:
		name = "{}-{}".format(self.iteration, name)
	gids = create(n)
	# decrease useless data recording for 'multitest' case
	if self.multitest and "moto" not in name:
		return gids

	mm_id = add_multimeter(name)
	sd_id = add_spike_detector(name)
	spikedetectors_dict[name] = sd_id
	multimeters_dict[name] = mm_id

	Connect(pre=mm_id, post=gids)
	Connect(pre=gids, post=sd_id)
	return gids


def connect(pre, post, syn_spec=None, weight=0, degree=40, delay=1.0):
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
