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
	def __init__(self):
		sensory = create_with_mmeter('sensory', 60)
		ia_aff = create_with_mmeter('ia_aff', 169)
		pool = [create_with_mmeter('pool{}'.format(i)) for i in range(1, 7)]
		moto = create_with_mmeter('moto', 169)
		ees = EES()
		ees.connect(sensory)
		ees.connect(moto)

		for pool_nucleus in pool:
			connect(pool_nucleus, moto, 25, 20)
		connect(ia_aff, moto, 25, 20)

		node11 = create_with_mmeter('node1.1')
		node12 = create_with_mmeter('node1.2')
		node13 = create_with_mmeter('node1.3')
		node14 = create_with_mmeter('node1.4')

		node21 = create_with_mmeter('node2.1')
		node22 = create_with_mmeter('node2.2')
		node23 = create_with_mmeter('node2.3')
		node24 = create_with_mmeter('node2.4')
		node25 = create_with_mmeter('node2.5')
		node26 = create_with_mmeter('node2.6')
		node27 = create_with_mmeter('node2.7')

		node31 = create_with_mmeter('node3.1')
		node32 = create_with_mmeter('node3.2')
		node33 = create_with_mmeter('node3.3')
		node34 = create_with_mmeter('node3.4')
		node35 = create_with_mmeter('node3.5')
		node36 = create_with_mmeter('node3.6')
		node37 = create_with_mmeter('node3.7')

		node41 = create_with_mmeter('node4.1')
		node42 = create_with_mmeter('node4.2')
		node43 = create_with_mmeter('node4.3')
		node44 = create_with_mmeter('node4.4')
		node45 = create_with_mmeter('node4.5')
		node46 = create_with_mmeter('node4.6')
		node47 = create_with_mmeter('node4.7')

		node51 = create_with_mmeter('node5.1')
		node52 = create_with_mmeter('node5.2')
		node53 = create_with_mmeter('node5.3')
		node54 = create_with_mmeter('node5.4')
		node55 = create_with_mmeter('node5.5')
		node56 = create_with_mmeter('node5.6')

		node61 = create_with_mmeter('node6.1')
		node62 = create_with_mmeter('node6.2')
		node63 = create_with_mmeter('node6.3')
		node64 = create_with_mmeter('node6.4')
		node65 = create_with_mmeter('node6.5')

		# node level 1
		connect(sensory, node11, 10.)
		connect(node11, node12, 15., delay=2)
		connect(node11, node21, 15., delay=2)
		connect(node11, node23, 7., delay=0.1)
		connect(node12, node13, 15., delay=2)
		connect(node13, node14, 15., delay=2) # delay 0.1 for ONE impulse on slice 1
		# connect to the IP
		connect(node13, pool[0], weight=40, degree=30)
		connect(node14, pool[0], weight=40, degree=30)

		# node level 2
		connect(node21, node22, 20.)
		connect(node21, node23, 4.)
		connect(node22, node21, 20.)
		# connect(node22, node23, -1.)
		connect(node23, node31, 15.)
		connect(node23, node33, 6., delay=0.1)
		connect(node23, node24, 15., delay=2)
		connect(node24, node25, 15., delay=2)
		connect(node25, node27, 15., delay=0.1)
		# connect(node25, node26, 8., delay=1)
		# connect(node26, node25, -15., delay=1)
		# connect(node26, node27, -15., delay=1)
		# connect(node27, node25, 15., delay=1)
		# connect(node27, node26, 8., delay=1)
		# connect to the IP
		connect(node25, pool[1], weight=80)
		connect(node27, pool[1], weight=80)

		# node level 3
		connect(node31, node32, 17)
		connect(node31, node33, 4, delay=1.5)
		connect(node32, node31, 17)
		# connect(node32, node33, -4)
		connect(node33, node13, Params.INH_COEF.value * -40, 80, delay=1)
		connect(node33, node14, Params.INH_COEF.value * -40, 80, delay=1)
		connect(node33, node41, 15.)
		connect(node33, node43, 6, delay=0.1)
		connect(node33, node34, 15, delay=2)
		connect(node34, node35, 35, delay=2)
		connect(node35, node37, 30)
		# connect(node35, node36, 8., delay=1)
		# connect(node36, node35, -15., delay=1)
		# connect(node36, node37, -15., delay=1)
		# connect(node37, node35, 15., delay=1)
		# connect(node37, node36, 8., delay=1)
		# connect to the IP
		connect(node35, pool[2], weight=80)
		connect(node37, pool[2], weight=80)

		# node level 4
		connect(node41, node42, 17)
		connect(node41, node43, 4)
		connect(node42, node41, 17)
		# connect(node42, node43, -4)
		connect(node43, node25, Params.INH_COEF.value * -30, 60, delay=.1)
		connect(node43, node27, Params.INH_COEF.value * -30, 60, delay=.1)
		connect(node43, node51, 15)
		connect(node43, node53, 9, delay=0.1)
		connect(node43, node44, 15, delay=2)
		connect(node44, node45, 15, delay=2)
		connect(node45, node47, 15)
		# connect(node45, node46, 8., delay=1)
		# connect(node46, node45, -15., delay=1)
		# connect(node46, node47, -15., delay=1)
		# connect(node47, node45, 15., delay=1)
		# connect(node47, node46, 8., delay=1)
		# connect to the IP
		connect(node45, pool[3], weight=60, degree=60)
		connect(node47, pool[3], weight=80, degree=60)

		# node level 5
		connect(node51, node52, 17)
		connect(node51, node53, 4, delay=2)
		connect(node52, node51, 17)
		# connect(node52, node53, -4)
		connect(node53, node35, Params.INH_COEF.value * -30, 60, delay=1) # weight = -30 de 0.1
		connect(node53, node37, Params.INH_COEF.value * -30, 60, delay=1) # weight = -30 de 0.1
		connect(node53, node61, 15)
		connect(node53, node63, 6, delay=0.1)
		connect(node53, node54, 15, delay=2)
		connect(node54, node56, 20, delay=1.5) # d 0.1
		# connect(node54, node55, 8, delay=1)
		# connect(node55, node54, -15, delay=1)
		# connect(node55, node56, -15, delay=1)
		# connect(node56, node54, 15, delay=1)
		# connect(node56, node55, 8, delay=1)
		# connect to the IP
		connect(node54, pool[4], weight=80)
		connect(node56, pool[4], weight=80)

		# node level 6
		connect(node61, node62, 17)
		connect(node61, node63, 4, delay=2)
		connect(node62, node61, 17)
		# connect(node62, node63, -8)
		# to the group 3
		connect(node63, node35, Params.INH_COEF.value * -25, 60, delay=0.1)
		connect(node63, node37, Params.INH_COEF.value * -25, 60, delay=0.1)
		# to the group 4
		connect(node63, node45, Params.INH_COEF.value * -25, 60, delay=0.1)
		connect(node63, node47, Params.INH_COEF.value * -25, 60, delay=0.1)
		# to the group 5
		connect(node63, node54, Params.INH_COEF.value * -25, 60, delay=0.1)
		connect(node63, node56, Params.INH_COEF.value * -25, 60, delay=0.1)
		connect(node63, node64, 20., delay=3) # 15 d 2
		connect(node64, node65, 20., delay=3) # 15 d 2
		# connect to the IP
		connect(node64, pool[5], weight=60)
		connect(node65, pool[5], weight=80)


def create(neuron_number):
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
			't_ref': 2.0,       # [ms] refractory period
			'V_m': -70.0,       # [mV] starting value of membrane potential
			'E_L': -70.0,       # [mV] leak reversal potential
			'g_L': 75.0,        # [nS] leak conductance
			'tau_syn_ex': .2,   # [ms]
			'tau_syn_in': 3.    # [ms]
		})

def create_with_mmeter(name, n=40):
	"""
	Function for creating new neruons
	Args:
		name (str): neurons group name
		n (int): number of neurons
	Returns:
		list: global IDs of created neurons
	"""
	gids = create(n)
	mm_id = add_multimeter(name)
	sd_id = add_spike_detector(name)

	spikedetectors_dict[name] = sd_id
	multimeters_dict[name] = mm_id

	Connect(pre=mm_id, post=gids)
	Connect(pre=gids, post=sd_id)
	return gids

def connect(pre, post, weight, degree=40, delay=1.):
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

	"""
	def __init__(self):
		sensory = create_with_mmeter('sensory', 60)
		ia_aff = create_with_mmeter('ia_aff', 169)
		pool = [create_with_mmeter('pool{}'.format(i+1)) for i in range(6)]
		moto = create_with_mmeter('moto', 169)
		ees = EES()
		ees.connect(sensory)
		ees.connect(moto)

		for pool_nucleus in pool:
			connect(pool_nucleus, moto, weight=25, degree=20)
		connect(ia_aff, moto, 25, 20)

		# Nodes 1
		node11 = create_with_mmeter('node1.1')
		node12 = create_with_mmeter('node1.2')
		node13 = create_with_mmeter('node1.3')
		node14 = create_with_mmeter('node1.4')
		# Nodes 2
		node21 = create_with_mmeter('node2.1')
		node22 = create_with_mmeter('node2.2')
		node23 = create_with_mmeter('node2.3')
		node24 = create_with_mmeter('node2.4')
		node25 = create_with_mmeter('node2.5')
		node26 = create_with_mmeter('node2.6')
		node27 = create_with_mmeter('node2.7')
		# Nodes 3
		node31 = create_with_mmeter('node3.1')
		node32 = create_with_mmeter('node3.2')
		node33 = create_with_mmeter('node3.3')
		node34 = create_with_mmeter('node3.4')
		node35 = create_with_mmeter('node3.5')
		node36 = create_with_mmeter('node3.6')
		node37 = create_with_mmeter('node3.7')
		# Nodes 4
		node41 = create_with_mmeter('node4.1')
		node42 = create_with_mmeter('node4.2')
		node43 = create_with_mmeter('node4.3')
		node44 = create_with_mmeter('node4.4')
		node45 = create_with_mmeter('node4.5')
		node46 = create_with_mmeter('node4.6')
		node47 = create_with_mmeter('node4.7')
		# Nodes 5
		node51 = create_with_mmeter('node5.1')
		node52 = create_with_mmeter('node5.2')
		node53 = create_with_mmeter('node5.3')
		node54 = create_with_mmeter('node5.4')
		node55 = create_with_mmeter('node5.5')
		node56 = create_with_mmeter('node5.6')
		# Nodes 6
		node61 = create_with_mmeter('node6.1')
		node62 = create_with_mmeter('node6.2')
		node63 = create_with_mmeter('node6.3')
		node64 = create_with_mmeter('node6.4')
		node65 = create_with_mmeter('node6.5')

		# Connections
		# Node 1
		connect(sensory, node11, 10.)
		connect(node11, node12, 15., delay=2)
		connect(node11, node21, 15, delay=1)   # 7.5
		connect(node11, node23, 8., delay=1)
		connect(node12, node13, 10., delay=2.5)
		connect(node13, node14, 25., delay=0.1)
		# Group 1 to the IP
		for node in [node13, node14]:
			connect(node, pool[0], 60., degree=80) # w 40 d 30

		# Node 2
		connect(node21, node22, 15.)
		connect(node21, node23, 8.)
		connect(node22, node21, 15.)
		connect(node22, node23, -3.)
		connect(node23, node24, 15., delay=1)
		#connect(node23, node31, 20., delay=3)
		#connect(node23, node33, 8., delay=3)
		connect(node24, node25, 25.) # 15
		# connect(node25, node26, 25.) # 15
		connect(node25, node27, 15.) # 15
		# connect(node26, node25, -5.)
		# connect(node26, node27, -5.)
		connect(node27, node25, 15.)
		# # Group 2 to the IP
		#for node in [node25, node27]:
		#	connect(node, pool[1], 80)

		# # Node 3
		# connect(node31, node32, 20.) # 15
		# connect(node31, node33, 15., delay=1)
		# connect(node32, node31, 20.) # 15
		# connect(node32, node33, -0.1)
		# # Node 3.3 -> Group 1
		# for node in [node13, node14]:
		# 	connect(node33, node, -20, degree=80, delay=2)
		# connect(node33, node34, 10., delay=4) #4
		# connect(node33, node41, 15.)
		# connect(node33, node43, 10.)
		# connect(node34, node35, 10., delay=4) #4
		# connect(node35, node36, 15.)
		# connect(node35, node37, 15., delay=0.1)
		# connect(node36, node35, -2.)
		# connect(node36, node37, -2.)
		# connect(node37, node35, 15.)
		# connect(node37, node36, 15.)
		# Group 3 to the IP
		# for node in [node35, node37]:
		# 	connect(node, pool[2], 80)
##
		# # Node 4
		# connect(node41, node42, 17.)
		# connect(node41, node43, 4.)
		# connect(node42, node41, 17.)
		# connect(node42, node43, -0.5)
		# # Node 4.3 -> Nodes 2.5-2.7
		# for node in [node25, node27]:
		# 	connect(node43, node, -10, degree=60, delay=1)
		# connect(node43, node44, 15.)
		# connect(node43, node51, 15.)
		# connect(node43, node53, 15., delay=3)
		# connect(node44, node45, 15.)
		# connect(node45, node46, 15.)
		# connect(node45, node47, 15.)
		# connect(node46, node45, -5)
		# connect(node46, node47, -5)
		# connect(node47, node45, 15.)
		# connect(node47, node46, 15.)
		# # Group 4 to the IP
		# for node in [node45, node47]:
		# 	connect(node, pool[3], 60, degree=60)
#
		# # Node 5
		# connect(node51, node52, 17.)
		# connect(node51, node53, 4.)
		# connect(node52, node51, 17.)
		# connect(node52, node53, -17)
		# # Node 5.3 -> Group 3
		# for node in [node35, node37]:
		# 	connect(node53, node, -20, degree=60, delay=5)
		# connect(node53, node54, 15., delay=2)
		# connect(node53, node61, 15.)
		# connect(node53, node63, 15.)
		# connect(node54, node55, 15.)
		# connect(node54, node56, 25.)
		# connect(node55, node54, -5.)
		# connect(node55, node56, -5.)
		# connect(node56, node55, 15.)
		# connect(node56, node56, 15.)
		# # Group 5 to the IP
		# connect(node54, pool[4], 40)
		# connect(node56, pool[4], 80)
#
		# # Node 6
		# connect(node61, node62, 17.)
		# connect(node61, node63, 4.)
		# connect(node62, node61, 17.)
		# connect(node62, node63, 17.)
		# # Node 6.3 -> Group 3
		# for node in [node35, node37]:
		# 	connect(node63, node, -3, degree=60, delay=.1)
		# # Node 6.3 -> Group 4
		# for node in [node45, node47]:
		# 	connect(node63, node, -3, degree=60, delay=.1)
		# # Node 6.3 -> Group 5
		# for node in [node54, node56]:
		# 	connect(node63, node, -3, degree=60, delay=.1)
		# connect(node63, node64, 15., delay=2.)
		# connect(node64, node65, 15., delay=2.)
		# # Group 6 to the IP
		# connect(node64, pool[5], 60., delay=1)
		# connect(node65, pool[5], 80., delay=1)
"""
