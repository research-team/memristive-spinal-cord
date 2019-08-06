import os
import nest
import numpy as np
from random import normalvariate
from collections import defaultdict

syn_outdegree = 27

class Parameters:
	__slots__ = ['tests', 'steps', 'cms', 'EES', 'inh',
	             'ped', 'ht5', 'save_all', 'step_cycle',
	             'resolution', 'T_sim', 'skin_stim']

	def __init__(self):
		self.tests: int
		self.steps: int
		self.cms: int
		self.EES: int
		self.inh: int
		self.ped: int
		self.ht5: int
		self.save_all: int
		self.step_cycle: int
		self.resolution: float
		self.T_sim: float
		self.skin_stim: float


class Functions:
	"""
	TODO add info
	"""
	def __init__(self, P):
		"""
		Args:
			P (Parameters):
		"""
		stim = {21: 25,
		        15: 50,
		        6: 125}
		P.skin_stim = stim[P.cms]
		# init T of muslce activation time
		extensor_time = 6 * P.skin_stim
		flexor_time = (5 if P.ped == 2 else 7) * P.skin_stim
		# init global T of simulation
		P.step_cycle = extensor_time + flexor_time
		P.T_sim = float(P.step_cycle * P.steps)

		self.P = P


	def __build_params(self):
		"""
		Returns:
			dict:
		"""
		# V_adj = -63.0;           // adjusts threshold to around -50 mV
		# g_bar = 1500;            // [nS] the maximal possible conductivity

		neuron_params = {'t_ref': normalvariate(3.0, 0.4),  # [ms] refractory period
		                 'V_m': -70.0,      # [mV] starting value of membrane potential
		                 'E_L': -72.0,      # [mV] Reversal potential for the leak current
		                 'g_Na': 20000.0,   # [nS] Maximal conductance of the Sodium current
		                 'g_K': 6000.0,     # [nS] Maximal conductance of the Potassium current
		                 'g_L': 30.0,       # [nS] Conductance of the leak current
		                 'E_Na': 50.0,      # [mV] Reversal potential for the Sodium current
		                 'E_K': -100.0,     # [mV] Reversal potential for the Potassium current
		                 'E_ex': 0.0,       # [mV] Reversal potential for excitatory input
		                 'E_in': -80.0,     # [mV] Reversal potential for excitatory input
		                 'tau_syn_ex': 0.2, # [ms] Decay time of excitatory synaptic current (ms)
		                 'tau_syn_in': 2.0, # [ms] Decay time of inhibitory synaptic current (ms)
		                 'C_m': normalvariate(200.0, 6)}  # [pF]

		return neuron_params


	def save(self):
		folder = nest.GetKernelStatus()['data_path']
		if not folder:
			folder = os.getcwd()
		prefix = nest.GetKernelStatus()['data_prefix']

		raise NotImplemented


	def add_multimeter(self, name, record_from):
		"""
		Function for creating NEST multimeter node
		Args:
			name (str):
				name of the node to which will be connected the multimeter
			record_from (str):
				Extracellular or V_m (intracelullar) recording variants
		Returns:
			tuple: global NEST ID of the multimeter
		"""
		if record_from not in ['Extracellular', 'V_m']:
			raise NotImplemented(f"The '{record_from}' parameter is not implemented "
			                     "for membrane potential recording")
		mm_params = {'label': name,
		             'record_from': [record_from, "g_ex", "g_in"],
		             'withgid': True,
		             'withtime': True,
		             'interval': nest.GetKernelStatus()['resolution'],
		             'to_file': True,
		             'to_memory': False}

		return nest.Create(model='multimeter', n=1, params=mm_params)


	def add_spike_detector(self, name):
		"""
		ToDo add info
		Args:
			name: neurons group name
	    Returns:
			list: list of spikedetector GID
	    """
		detector_params = {'label': name,
		                   'withgid': True,
		                   'to_file': True,
		                   'to_memory': False}

		return nest.Create(model='spike_detector', n=1, params=detector_params)


	def form_group(self, name, nrn_number=20):
		"""
		Function for creating new neruons
		Args:
			name (str): neurons group name
			nrn_number (int): number of neurons
		Returns:
			list: global IDs of created neurons
		"""
		neuron_model = 'hh_cond_exp_traub'
		gids = [nest.Create(model=neuron_model, n=1, params=self.__build_params())[0] for _ in range(nrn_number)]

		if self.P.tests > 1 and name not in ["MN_E", "MN_F"]:
			return gids

		multimeter_id = self.add_multimeter(name, record_from="V_m")
		spikedetector_id = self.add_spike_detector(name)

		nest.Connect(pre=multimeter_id, post=gids)
		nest.Connect(pre=gids, post=spikedetector_id)

		return gids


	def connect_spike_generator(self, node, rate, t_start=None, t_end=None, offset=0):
		"""
		TODO add info
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
		if not t_start:
			t_start = nest.GetKernelStatus()['resolution']
		if not t_end:
			t_end = self.P.T_sim

		# total possible number of spikes without t_start and t_end at current rate
		num_spikes = self.P.T_sim // (1000 / rate)
		spike_times = np.arange(num_spikes) * round(1000 / rate, 2) + offset
		spike_times = spike_times[(t_start <= spike_times) & (spike_times < t_end)]

		# parameters
		spike_gen_params = {'spike_times': spike_times,
		                    'spike_weights': [450.] * len(spike_times)}

		syn_spec = {'model': 'static_synapse',
		            'weight': 1.0,
		            'delay': 0.1}

		conn_spec = {'rule': 'all_to_all',
		             'multapses': False}

		# create the spike generator
		spike_generator = nest.Create(model='spike_generator', n=1, params=spike_gen_params)
		# connect the spike generator with node
		nest.Connect(pre=spike_generator, post=node, syn_spec=syn_spec, conn_spec=conn_spec)


	def connect_noise_generator(self, node, rate, t_start=None, t_end=None):
		"""
		TODO add info
		Args:
			node (tuple or list):
				GIDs of the neurons
			t_start (int):
				start stimulation time
			t_end (int):
				end stimulation time
			rate (int):
				frequency rate
		"""
		if not t_start:
			t_start = nest.GetKernelStatus()['resolution']
		if not t_end:
			t_end = self.P.T_sim

		# parameters
		spike_gen_params = {'rate': float(rate),
		                    'start': float(t_start),
		                    'stop': float(t_end)}

		syn_spec = {'model': 'static_synapse',
		            'weight': {"distribution": "normal",
		                       "mu": float(50),
		                       "sigma": float(50) / 10},
		            'delay': 0.1}

		conn_spec = {'rule': 'all_to_all',
		             'multapses': False}

		# create the spike generator
		spike_generator = nest.Create(model='poisson_generator', n=1, params=spike_gen_params)
		# connect the spike generator with node
		nest.Connect(pre=spike_generator, post=node, syn_spec=syn_spec, conn_spec=conn_spec)


	def __connect(self, pre_ids, post_ids, syn_delay, syn_weight, conn_spec, no_distr):
		delay_distr = {"distribution": "normal",
		               "mu": float(syn_delay),
		               "sigma": float(syn_delay) / 5}
		weight_distr = {"distribution": "normal",
		                "mu": float(syn_weight),
		                "sigma": abs(syn_weight) / 10}
		# initialize synapse specification
		syn_spec = {'model': 'static_synapse',
		            'delay': float(syn_delay) if no_distr else delay_distr,
		            'weight': float(syn_weight) if no_distr else weight_distr}

		# NEST connection
		nest.Connect(pre=pre_ids, post=post_ids, syn_spec=syn_spec, conn_spec=conn_spec)


	def connect_one_to_all(self, pre_ids, post_ids, syn_delay, syn_weight, no_distr=False):
		"""
		Connect group of neurons
		Args:
			pre_ids (list): pre neurons GIDs
			post_ids (list): pre neurons GIDs
			syn_delay (float): synaptic delay
			syn_weight (float): synaptic strength
			no_distr (bool): disable distribution or no
		"""
		conn_spec = {'rule': 'all_to_all',  # fixed outgoing synapse number
		             'multapses': True,  # allow recurring connections
		             'autapses': False}  # allow self-connection
		self.__connect(pre_ids, post_ids, syn_delay, syn_weight, conn_spec, no_distr)


	def connect_fixed_outdegree(self, pre_ids, post_ids, syn_delay, syn_weight, outdegree=syn_outdegree, no_distr=False):
		"""
		Connect group of neurons
		Args:
			pre_ids (list): pre neurons GIDs
			post_ids (list): pre neurons GIDs
			syn_delay (float): synaptic delay
			syn_weight (float): synaptic strength
			outdegree (int): number of outgoing synapses from PRE neurons
			no_distr (bool): disable distribution or no
		"""
		# initialize connection specification
		conn_spec = {'rule': 'fixed_outdegree',  # fixed outgoing synapse number
		             'outdegree': int(outdegree),  # number of synapses outgoing from PRE neuron
		             'multapses': True,  # allow recurring connections
		             'autapses': False}  # allow self-connection
		self.__connect(pre_ids, post_ids, syn_delay, syn_weight, conn_spec, no_distr)


	def simulate(self):
		# simulate the topology
		nest.Simulate(self.P.T_sim)
