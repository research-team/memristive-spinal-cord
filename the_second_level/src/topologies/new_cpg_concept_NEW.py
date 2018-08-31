import random
from enum import Enum
from nest import Create, Connect
from ..tools.multimeter import add_multimeter
from ..tools.spike_detector import add_spike_detector
from ..data import *


class Params(Enum):
	NUM_SUBLEVELS = 6
	NUM_SPIKES = 6
	RATE = 40
	SIMULATION_TIME = 175.
	INH_COEF = 1.
	PLOT_SLICES_SHIFT = 8.  # ms

	TO_PLOT = {
		'node1.1': 'Node 1.1',
		'node1.2': 'Node 1.2',
		'node1.3': 'Node 1.3',
		'node1.4': 'Node 1.4',
		'node2.1': 'Node 2.1',
		'node2.2': 'Node 2.2',
		'node2.3': 'Node 2.3',
		'node2.4': 'Node 2.4',
		'node2.5': 'Node 2.5',
		'node2.6': 'Node 2.6',
		'node2.7': 'Node 2.7',
		'node3.1': 'Node 3.1',
		'node3.2': 'Node 3.2',
		'node3.3': 'Node 3.3',
		'node3.4': 'Node 3.4',
		'node3.5': 'Node 3.5',
		'node3.6': 'Node 3.6',
		'node3.7': 'Node 3.7',
		'node4.1': 'Node 4.1',
		'node4.2': 'Node 4.2',
		'node4.3': 'Node 4.3',
		'node4.4': 'Node 4.4',
		'node4.5': 'Node 4.5',
		'node4.6': 'Node 4.6',
		'node4.7': 'Node 4.7',
		'node5.1': 'Node 5.1',
		'node5.2': 'Node 5.2',
		'node5.3': 'Node 5.3',
		'node5.4': 'Node 5.4',
		'node5.5': 'Node 5.5',
		'node5.6': 'Node 5.6',
		'node6.1': 'Node 6.1',
		'node6.2': 'Node 6.2',
		'node6.3': 'Node 6.3',
		'node6.4': 'Node 6.4',
		'node6.5': 'Node 6.5',
		'pool1': 'Pool1',
		'pool2': 'Pool2',
		'pool3': 'Pool3',
		'pool4': 'Pool4',
		'pool5': 'Pool5',
		'pool6': 'Pool6',
		'moto': 'Moto',
		}

	TO_PLOT_WITH_SLICES = {
		'moto': 6
	}


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
		Connect(
			pre=self.ees,
			post=post,
			syn_spec={
				'model': 'static_synapse',
				'weight': 1.,
				'delay': .1
			},
			conn_spec={
				'rule': 'fixed_outdegree',
				'outdegree': len(post),
				'autapses': False,
				'multapses': False
			})


class Topology:
	def __init__(self, multitest=False, iteration=0):
		self.multitest = multitest
		self.iteration = iteration
		sensory = self.create_with_mmeter('sensory', 60)
		ia_aff = self.create_with_mmeter('ia_aff', 169)
		pool = [self.create_with_mmeter('pool{}'.format(i)) for i in range(1, 7)]
		moto = self.create_with_mmeter('moto', 169)
		ees = EES()
		ees.connect(sensory)
		ees.connect(moto)

		for pool_nucleus in pool:
			self.connect(pool_nucleus, moto, 25, 20)
		self.connect(ia_aff, moto, 25, 20)

		node11 = self.create_with_mmeter('node1.1')
		node12 = self.create_with_mmeter('node1.2')
		node13 = self.create_with_mmeter('node1.3')
		node14 = self.create_with_mmeter('node1.4')

		node21 = self.create_with_mmeter('node2.1')
		node22 = self.create_with_mmeter('node2.2')
		node23 = self.create_with_mmeter('node2.3')
		node24 = self.create_with_mmeter('node2.4')
		node25 = self.create_with_mmeter('node2.5')
		node26 = self.create_with_mmeter('node2.6')
		node27 = self.create_with_mmeter('node2.7')

		node31 = self.create_with_mmeter('node3.1')
		node32 = self.create_with_mmeter('node3.2')
		node33 = self.create_with_mmeter('node3.3')
		node34 = self.create_with_mmeter('node3.4')
		node35 = self.create_with_mmeter('node3.5')
		node36 = self.create_with_mmeter('node3.6')
		node37 = self.create_with_mmeter('node3.7')

		node41 = self.create_with_mmeter('node4.1')
		node42 = self.create_with_mmeter('node4.2')
		node43 = self.create_with_mmeter('node4.3')
		node44 = self.create_with_mmeter('node4.4')
		node45 = self.create_with_mmeter('node4.5')
		node46 = self.create_with_mmeter('node4.6')
		node47 = self.create_with_mmeter('node4.7')

		node51 = self.create_with_mmeter('node5.1')
		node52 = self.create_with_mmeter('node5.2')
		node53 = self.create_with_mmeter('node5.3')
		node54 = self.create_with_mmeter('node5.4')
		node55 = self.create_with_mmeter('node5.5')
		node56 = self.create_with_mmeter('node5.6')

		node61 = self.create_with_mmeter('node6.1')
		node62 = self.create_with_mmeter('node6.2')
		node63 = self.create_with_mmeter('node6.3')
		node64 = self.create_with_mmeter('node6.4')
		node65 = self.create_with_mmeter('node6.5')

		# node level 1
		self.connect(sensory, node11, 10.)
		self.connect(node11, node12, 15., delay=2)
		self.connect(node11, node21, 15., delay=2)
		self.connect(node11, node23, 7., delay=0.1)
		self.connect(node12, node13, 15., delay=2.5)
		self.connect(node13, node14, 15., delay=2) # delay 0.1 for ONE impulse on slice 1
		# connect to the IP
		self.connect(node13, pool[0], weight=80, degree=30)
		self.connect(node14, pool[0], weight=80, degree=30)

		# node level 2
		self.connect(node21, node22, 20.)
		self.connect(node21, node23, 4.)
		self.connect(node22, node21, 20.)
		# connect(node22, node23, -1.)
		self.connect(node23, node31, 15.)
		self.connect(node23, node33, 6., delay=0.1)
		self.connect(node23, node24, 15., delay=2.5)
		self.connect(node24, node25, 15., delay=2.5)
		self.connect(node25, node27, 15., delay=0.1)
		# connect(node25, node26, 8., delay=1)
		# connect(node26, node25, -15., delay=1)
		# connect(node26, node27, -15., delay=1)
		# connect(node27, node25, 15., delay=1)
		# connect(node27, node26, 8., delay=1)
		# connect to the IP
		self.connect(node25, pool[1], weight=80)
		self.connect(node27, pool[1], weight=80)

		# node level 3
		self.connect(node31, node32, 17)
		self.connect(node31, node33, 4, delay=1.5)
		self.connect(node32, node31, 17)
		# connect(node32, node33, -4)
		self.connect(node33, node13, Params.INH_COEF.value * -40, 80, delay=1)
		self.connect(node33, node14, Params.INH_COEF.value * -40, 80, delay=1)
		self.connect(node33, node41, 15.)
		self.connect(node33, node43, 6, delay=0.1)
		self.connect(node33, node34, 15, delay=3) #d 2.5
		self.connect(node34, node35, 35, delay=3) # 35 d2.5
		self.connect(node35, node37, 25, delay=1.5) # 30 d 2
		# connect(node35, node36, 8., delay=1)
		# connect(node36, node35, -15., delay=1)
		# connect(node36, node37, -15., delay=1)
		# connect(node37, node35, 15., delay=1)
		# connect(node37, node36, 8., delay=1)
		# connect to the IP
		self.connect(node35, pool[2], weight=80)
		self.connect(node37, pool[2], weight=80)

		# node level 4
		self.connect(node41, node42, 17)
		self.connect(node41, node43, 4)
		self.connect(node42, node41, 17)
		# connect(node42, node43, -4)
		self.connect(node43, node25, Params.INH_COEF.value * -30, 60, delay=.1)
		self.connect(node43, node27, Params.INH_COEF.value * -30, 60, delay=.1)
		self.connect(node43, node51, 10, delay=1) #d 1 w 15
		self.connect(node43, node53, 9, delay=1) # 0.1
		self.connect(node43, node44, 15, delay=2.5)
		self.connect(node44, node45, 15, delay=2.5)
		self.connect(node45, node47, 15)
		# connect(node45, node46, 8., delay=1)
		# connect(node46, node45, -15., delay=1)
		# connect(node46, node47, -15., delay=1)
		# connect(node47, node45, 15., delay=1)
		# connect(node47, node46, 8., delay=1)
		# connect to the IP
		self.connect(node45, pool[3], weight=60, degree=60)
		self.connect(node47, pool[3], weight=60, degree=60) #w 80

		# node level 5
		self.connect(node51, node52, 17)
		self.connect(node51, node53, 4, delay=2)
		self.connect(node52, node51, 17)
		# connect(node52, node53, -4)
		self.connect(node53, node35, Params.INH_COEF.value * -30, 60, delay=1.5) # weight = -30 de 0.1
		self.connect(node53, node37, Params.INH_COEF.value * -30, 60, delay=1.5) # weight = -30 de 0.1
		self.connect(node53, node61, 15)
		self.connect(node53, node63, 6, delay=0.1)
		self.connect(node53, node54, 15, delay=2.5) #2
		self.connect(node54, node56, 20, delay=1.5) # d 0.1
		# connect(node54, node55, 8, delay=1)
		# connect(node55, node54, -15, delay=1)
		# connect(node55, node56, -15, delay=1)
		# connect(node56, node54, 15, delay=1)
		# connect(node56, node55, 8, delay=1)
		# connect to the IP
		self.connect(node54, pool[4], weight=80)
		self.connect(node56, pool[4], weight=80)

		# node level 6
		self.connect(node61, node62, 17)
		self.connect(node61, node63, 4, delay=2)
		self.connect(node62, node61, 17)
		# connect(node62, node63, -8)
		# to the group 3
		self.connect(node63, node35, Params.INH_COEF.value * -25, 60, delay=0.1)
		self.connect(node63, node37, Params.INH_COEF.value * -25, 60, delay=0.1)
		# to the group 4
		self.connect(node63, node45, Params.INH_COEF.value * -25, 60, delay=0.1)
		self.connect(node63, node47, Params.INH_COEF.value * -25, 60, delay=0.1)
		# to the group 5
		self.connect(node63, node54, Params.INH_COEF.value * -25, 60, delay=0.1)
		self.connect(node63, node56, Params.INH_COEF.value * -25, 60, delay=0.1)
		self.connect(node63, node64, 20., delay=3.5) # 15 d 2
		self.connect(node64, node65, 20., delay=2) # 15 d 2
		# connect to the IP
		self.connect(node64, pool[5], weight=80)
		self.connect(node65, pool[5], weight=80)


	def create(self, neuron_number):
		"""
		Args:
			neuron_number (int): number of neurons
		Returns:
			list: list of neurons ID
		"""
		return Create(
			model='hh_cond_exp_traub',
			n=neuron_number,
			params={
				't_ref': random.uniform(1.9, 2.1),       # [ms] refractory period
				'V_m': -70.0,       # [mV] starting value of membrane potential
				'E_L': random.uniform(-70.5, -69.5),       # [mV] leak reversal potential
				'g_L': 75.0,        # [nS] leak conductance
				'tau_syn_ex': .2,   # [ms]
				'tau_syn_in': 3.    # [ms]
			})

	def create_with_mmeter(self, name, n=40):
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
		gids = self.create(n)
		# decrease useless data recording
		if self.multitest and "moto" not in name:
			return gids
		mm_id = add_multimeter(name)
		sd_id = add_spike_detector(name)

		spikedetectors_dict[name] = sd_id
		multimeters_dict[name] = mm_id

		Connect(pre=mm_id, post=gids)
		Connect(pre=gids, post=sd_id)
		return gids

	def connect(self, pre, post, weight, degree=40, delay=1.):
		"""
		Connect group of neurons
		Args:
			pre (list):
			post (list):
			weight (float):
			degree (int):
			delay (float):
		"""
		Connect(
			pre=pre,
			post=post,
			syn_spec={
				'model': 'static_synapse',
				'delay': float(delay),
				'weight': float(weight),
			},
			conn_spec={
				'rule': 'fixed_outdegree',
				'outdegree': int(degree),
				'multapses': True,
				'autapses': True
			})
