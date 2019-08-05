import nest
from random import random

syn_outdegree = 27

class Functions:
	"""
	TODO add info
	"""
	def __init__(self, multitest):
		self.multitest = multitest
		self.multimeters_dict = {}
		self.spikedetectors_dict = {}


	def __build_params(self):
		"""
		Returns:
			dict:
		"""
		neuron_params = {'t_ref': random.normal(3, 0.4),  # [ms] refractory period
		                 'V_m': -70.0,  # [mV] starting value of membrane potential
		                 'E_L': random.normal(-75.0, -65.0),  # [mV]
		                 'g_L': 75.0,  # [nS] leak conductance
		                 'tau_syn_ex': random.normal(0.2, 0.35),  # [ms]
		                 'tau_syn_in': random.normal(2.5, 3.5),  # [ms]
		                 'C_m': random.normal(200, 6)}  # [pF]

		return neuron_params


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

		return nest.Create(model='multimeter', n=1, params={'label': name,
		                                                    'record_from': [record_from, "g_ex", "g_in"],
		                                                    'withgid': True,
		                                                    'withtime': True,
		                                                    'interval': 0.1,
		                                                    'to_file': True,
		                                                    'to_memory': True})


	def add_spike_detector(self, name):
		"""
		ToDo add info
		Args:
			name: neurons group name
	    Returns:
			list: list of spikedetector GID
	    """
		return nest.Create(model='spike_detector', n=1, params={'label': name,
		                                                        'withgid': True,
		                                                        'to_file': True,
		                                                        'to_memory': True})


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

		if self.multitest and name not in ["MN_E", "MN_F"]:
			return gids

		multimeter_id = self.add_multimeter(name, record_from="V_m")
		spikedetector_id = self.add_spike_detector(name)

		self.spikedetectors_dict[name] = spikedetector_id
		self.multimeters_dict[name] = multimeter_id

		nest.Connect(pre=multimeter_id, post=gids)
		nest.Connect(pre=gids, post=spikedetector_id)

		return gids


	def connect_spike_generator(self, node, t_start, t_end, rate, offset=0):
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
		spike_generator = nest.Create(model='spike_generator', params=spike_generator_params)
		# connect the spike generator with node
		nest.Connect(pre=spike_generator, post=node, syn_spec=syn_spec, conn_spec=conn_spec)


	def __connect(self, pre_ids, post_ids, syn_delay, syn_weight, conn_spec, no_distr):
		delay_distr = {"distribution": "normal",
		               "mu": float(syn_delay),
		               "sigma": float(syn_delay) / 5}
		weight_distr = {"distribution": "normal",
		                "mu": float(syn_weight),
		                "sigma": float(syn_weight) / 10}
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
		conn_spec = {'rule': 'one_to_all',  # fixed outgoing synapse number
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