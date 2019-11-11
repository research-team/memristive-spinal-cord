import os
import math
import nest
import numpy as np
from random import normalvariate
from collections import defaultdict
import re

syn_outdegree = 27
nrns_in_group = 40


class Parameters:
	__slots__ = ['tests', 'steps', 'cms', 'EES', 'inh', 'ped', 'ht5', 'save_all', 'step_cycle',
	             'resolution', 'T_sim', 'skin_stim', 'extensor_time', 'flexor_time', 'air', 'toe']

	def __init__(self):
		self.tests: int
		self.steps: int
		self.cms: int
		self.EES: int
		self.inh: int
		self.ped: int
		self.ht5: int
		self.save_all: int
		self.step_cycle: float
		self.resolution: float
		self.T_sim: float
		self.skin_stim: float
		self.extensor_time: int
		self.flexor_time: int
		self.air: bool
		self.toe: bool


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
		P.resolution = nest.GetKernelStatus()['resolution']
		# init T of muslce activation time
		P.extensor_time = 6 * P.skin_stim
		P.flexor_time = (7 if P.ped == 4 else 5) * 25
		# init global T of simulation
		P.step_cycle = P.extensor_time + P.flexor_time
		P.T_sim = float(P.step_cycle * P.steps)
		self.P = P
		self.multimeters = []
		self.cv_generators = []
		self.spikedetectors = []

	def __build_params(self):
		"""
		ToDo add info
		Returns:
			dict: formed neuron's params with randomization
		"""
		neuron_params = {'t_ref': normalvariate(3, 0.4),  # [ms] refractory period
		                 'V_m': -70.0,  # [mV] starting value of membrane potential
		                 'E_L': -72.0,  # [mV] Reversal potential for the leak current
		                 'g_Na': 20000.0,  # [nS] Maximal conductance of the Sodium current
		                 'g_K': 6000.0,  # [nS] Maximal conductance of the Potassium current
		                 'g_L': 30.0,  # [nS] Conductance of the leak current
		                 'E_Na': 50.0,  # [mV] Reversal potential for the Sodium current
		                 'E_K': -100.0,  # [mV] Reversal potential for the Potassium current
		                 'E_ex': 0.0,  # [mV] Reversal potential for excitatory input
		                 'E_in': -80.0,  # [mV] Reversal potential for excitatory input
		                 'tau_syn_ex': 0.2,  # [ms] Decay time of excitatory synaptic current (ms)
		                 'tau_syn_in': 2.0,  # [ms] Decay time of inhibitory synaptic current (ms)
		                 'C_m': normalvariate(200, 6)}  # [pF] capacity of membrane

		return neuron_params

	def create_multimeter(self, name, record_from):
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
		             'start': 0.,
		             'file_extension': 'mm',
		             'record_from': [record_from, "g_ex", "g_in"],
		             'withgid': False,
		             'withtime': True,
		             'interval': nest.GetKernelStatus('resolution'),
		             'to_file': True,
		             'to_memory': False}

		return nest.Create(model='multimeter', n=1, params=mm_params)

	def create_spikedetector(self, name):
		"""
		ToDo add info
		Args:
			name: neurons group name
		Returns:
			list: list of spikedetector GID
		"""
		detector_params = {'label': name,
		                   'withgid': False,
		                   'file_extension': 'sd',
		                   'to_file': True,
		                   'to_memory': False}

		return nest.Create(model='spike_detector', n=1, params=detector_params)

	def form_group(self, name, nrn_number=nrns_in_group, d_distr="uniform"):
		"""
		Function for creating new neruons
		Args:
			name (str): neurons group name
			nrn_number (int): number of neurons
			d_distr (str): distribution type
		Returns:
			list: global IDs of created neurons
		"""
		sp_C_m = 2  # pF/um²
		sp_g_Na = 0.5  # nS/um²
		sp_g_K = 3  # nS/um²
		sp_g_L = 0.02  # nS/um²
		neuron_model = 'hh_cond_exp_traub'
		neuron_params = {  # t_ref            # [ms] refractory period
			# C_m              # [pF] capacity of membrane
			# g_Na             # [nS] Maximal conductance of the Sodium current
			# g_K              # [nS] Maximal conductance of the Potassium current
			# g_L              # [nS] Conductance of the leak current
			'V_m': -70.0,  # [mV] starting value of membrane potential
			'E_Na': 50.0,  # [mV] Reversal potential for the Sodium current
			'E_K': -100.0,  # [mV] Reversal potential for the Potassium current
			'E_L': -70.0,  # [mV] Reversal potential for the leak current
			'E_ex': 0.0,  # [mV] Reversal potential for excitatory input
			'E_in': -80.0,  # [mV] Reversal potential for excitatory input
			'tau_syn_ex': 0.2,  # [ms] Decay time of excitatory synaptic current (ms)
			'tau_syn_in': 2.0,  # [ms] Decay time of inhibitory synaptic current (ms)
		}
		# create neurons with default common parameters
		gids = nest.Create(model=neuron_model, n=nrn_number, params=neuron_params)
		# generate refractory period
		t_refs = np.random.normal(3, 0.4, nrn_number)
		# generate neuron diameters
		if d_distr == "bimodal":
			# for motoneurons
			standby_percent = 70
			standby_size = int(nrn_number * standby_percent / 100)
			active_size = nrn_number - standby_size

			loc_active, scale_active = 27, 3  # 18 - 35, med=27
			loc_standby, scale_standby = 57, 6  # 38.5 - 78, med=57

			diameters = np.concatenate([np.random.normal(loc=loc_active, scale=scale_active, size=active_size),
			                            np.random.normal(loc=loc_standby, scale=scale_standby, size=standby_size)])
		elif d_distr == "normal":
			raise NotImplemented
		elif d_distr == "uniform":
			# for interneurons
			if 'OM' in name:
				diameters = np.random.uniform(low=1, high=10, size=nrn_number)
			else:
				diameters = np.random.uniform(low=3, high=8, size=nrn_number)
		else:
			raise NotImplemented("Can not recognize distribution name")

		# update neuron parameters based on distributions
		for gid, diameter, t_ref in zip(gids, diameters, t_refs):
			params = {'t_ref': t_ref,
			          'C_m': sp_C_m * diameter,
			          'g_Na': sp_g_Na * diameter,
			          'g_K': sp_g_K * diameter,
			          'g_L': sp_g_L * diameter}
			nest.SetStatus([gid], params)

		target_groups = [
			# "OM1_2_E",  # "OM1_0", "OM1_1", "OM1_2_F", "OM1_3",
			# "OM2_2_E",  # "OM2_0", "OM2_1", "OM2_2_F", "OM2_3",
			# "OM3_2_E",  # "OM3_0", "OM3_1", "OM3_2_F", "OM3_3",
			# "OM4_2_E",  # "OM4_0", "OM4_1", "OM4_2_F", "OM4_3",
			# "OM5_2_E",  # "OM5_0", "OM5_1", "OM5_2_F", "OM5_3",
			# "iIP_E", "iIP_F",
			# "eIP_E_0", "eIP_E_1", "eIP_E_2", "eIP_E_3", "eIP_E_4", "eIP_E_5", "eIP_E_6", "eIP_F",
			# "eIP_E",
			# "Ia_E_aff", "Ia_F_aff", "Ia_E_pool", "Ia_F_pool",
			"MN_E", "MN_F",
			# "R_E", "R_F"
		]

		if self.P.save_all or name in target_groups:
			mm_device = self.create_multimeter(name, record_from="V_m")
			sd_device = self.create_spikedetector(name)

			self.multimeters.append(mm_device)
			self.spikedetectors.append(sd_device)

			nest.Connect(pre=mm_device, post=gids)
			nest.Connect(pre=gids, post=sd_device)

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
		resolution = nest.GetKernelStatus()['resolution']
		if not t_start:
			t_start = resolution
		if not t_end:
			t_end = self.P.T_sim

		# total possible number of spikes without t_start and t_end at current rate
		num_spikes = int(self.P.T_sim / (1000 / rate))
		spike_times = np.around(int(1000 / rate) * np.arange(num_spikes) + offset, decimals=1).astype(float)
		if spike_times[0] == 0:
			spike_times[0] = resolution
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

	def connect_noise_generator(self, node, rate, t_start=None, t_end=None, weight=None):
		"""
		TODO add info
		Args:
			node (list):
				GIDs of the neurons
			t_start (float):
				start stimulation time
			t_end (float):
				end stimulation time
			rate (float):
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
		            'weight': weight if weight is not None else 300.0,
		            'delay': 0.1}

		conn_spec = {'rule': 'all_to_all',
		             'multapses': False}

		# create the spike generator
		spike_generator = nest.Create(model='poisson_generator', n=1, params=spike_gen_params)
		# connect the spike generator with node
		nest.Connect(pre=spike_generator, post=node, syn_spec=syn_spec, conn_spec=conn_spec)

		self.cv_generators.append(spike_generator)

	def __connect(self, pre_ids, post_ids, syn_delay, syn_weight, conn_spec, no_distr):
		delay_distr = {"distribution": "normal",
		               "mu": float(syn_delay),
		               "sigma": float(syn_delay) / 5}

		weight_distr = {"distribution": "normal",
		                "mu": float(syn_weight),
		                "sigma": abs(syn_weight) / 5,
		                }
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
		             'multapses': False,  # allow recurring connections
		             'autapses': False}  # allow self-connection
		self.__connect(pre_ids, post_ids, syn_delay, syn_weight, conn_spec, no_distr)

	def connect_fixed_outdegree(self, pre_ids, post_ids, syn_delay, syn_weight, outdegree=syn_outdegree,
	                            no_distr=False):
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
		"""
		ToDo add info
		"""
		# simulate the topology by step cycle
		for step_index in range(self.P.steps):
			# update CV generators time
			for gen_device in self.cv_generators:
				new_start = nest.GetStatus(gen_device, 'start')[0] + step_index * self.P.step_cycle
				new_stop = nest.GetStatus(gen_device, 'stop')[0] + step_index * self.P.step_cycle
				nest.SetStatus(gen_device, {'start': new_start, 'stop': new_stop})
			# simulate one step cycle
			nest.Simulate(self.P.step_cycle)

	def check(self, data):
		for d in data:
			if len(d) == 4:
				yield d

	def resave(self):
		"""
		ToDo add info
		"""
		k_times = 0
		k_volts = 1
		k_g_exc = 2
		k_g_inh = 3

		folder = nest.GetKernelStatus()['data_path']
		if not folder:
			folder = os.getcwd()
		prefix = nest.GetKernelStatus()['data_prefix']

		volt_parent_files = defaultdict(list)
		spikes_parent_files = defaultdict(list)

		# voltages
		for filename in filter(lambda f: f.startswith(prefix) and f.endswith('.mm'), os.listdir(folder)):
			parent_name = filename.split("-")[0]
			volt_parent_files[parent_name].append(filename)

		# spikes
		for filename in filter(lambda f: f.startswith(prefix) and f.endswith('.sd'), os.listdir(folder)):
			parent_name = filename.split("-")[0]
			spikes_parent_files[parent_name].append(filename)

		for name in volt_parent_files:
			volt_data = None

			filenames_dat = volt_parent_files[name]
			filenames_gdf = spikes_parent_files[name]

			if len(filenames_dat) == 1:
				continue

			for filename in filenames_dat:
				print(filename)
				with open(f"{folder}/{filename}") as file:
					filedata = self.check((line.split("\t")[:-1] for line in file.readlines()))
					filedata = np.array(list(filedata)).astype(float)
					if len(filedata) > 0:
						if volt_data is None:
							volt_data = filedata
						else:
							volt_data = np.concatenate((volt_data, filedata), axis=0)

			volt_data = volt_data[volt_data[:, k_times].argsort()]
			time_parts = len(np.unique(volt_data[:, k_times]))

			volts = np.mean(np.split(volt_data[:, k_volts], time_parts), axis=1)
			g_exc = np.mean(np.split(volt_data[:, k_g_exc], time_parts), axis=1)
			g_inh = np.mean(np.split(volt_data[:, k_g_inh], time_parts), axis=1)

			del volt_data

			expected_length = int(self.P.T_sim / self.P.resolution)

			# fill missing volts values
			for _ in range(expected_length - len(volts)):
				volts = np.append(volts, volts[-1])
			# fill missing g_exc values
			for _ in range(expected_length - len(g_exc)):
				g_exc = np.append(g_exc, g_exc[-1])
			# fill missing g_inh values
			for _ in range(expected_length - len(g_inh)):
				g_inh = np.append(g_inh, g_inh[-1])

			spikes = []
			for filename in filenames_gdf:
				with open(f"{folder}/{filename}") as file:
					spikes += list(map(float, file.readlines()))

			spikes = sorted(spikes)

			for filename in filenames_dat:
				os.remove(f"{folder}/{filename}")
			for filename in filenames_gdf:
				os.remove(f"{folder}/{filename}")

			with open(f"{folder}/{name}.dat", 'w') as file:
				file.write(" ".join(map(str, volts)))
				file.write("\n")
				file.write(" ".join(map(str, g_exc)))
				file.write("\n")
				file.write(" ".join(map(str, g_inh)))
				file.write("\n")
				file.write(" ".join(map(str, spikes)))
