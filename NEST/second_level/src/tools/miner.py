import os
import itertools
import numpy as np
import logging as log

from nest import GetStatus
from collections import defaultdict
from NEST.second_level.src.data import *
from NEST.second_level.src.paths import raw_data_path, spiketimes_path


class Miner:
	def __init__(self):
		log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
		self.logger = log.getLogger('Miner')


	def gather_mean_voltage(self, test_name, from_memory=False):
		"""
		Gather voltage from files and calculate mean for each milliseconds
		Args:
			test_name (str):
				neurons group name
			from_memory (bool):
				True - get data from the RAM, False - get data from files
		Returns:
			dict[float, float]: mean voltage for each millisecond
		"""
		self.logger.info("get data for '{}' from [{}]".format(test_name, "MEM" if from_memory else "FILE"))

		voltage_data = {}
		g_ex_data = {}
		g_in_data = {}

		if from_memory:
			# get values from memory
			record_from = 'Extracellular'
			try:
				GetStatus(multimeters_dict[test_name])[0]['events'][record_from]
			except Exception:
				record_from = "V_m"

			voltages = GetStatus(multimeters_dict[test_name])[0]['events'][record_from]
			g_ex = GetStatus(multimeters_dict[test_name])[0]['events']['g_ex']
			g_in = GetStatus(multimeters_dict[test_name])[0]['events']['g_in']

			times = GetStatus(multimeters_dict[test_name])[0]['events']['times']
			senders = GetStatus(multimeters_dict[test_name])[0]['events']['senders']

			voltage_data = {k: 0 for k in np.unique(times)}
			g_ex_data = {k: 0 for k in np.unique(times)}
			g_in_data = {k: 0 for k in np.unique(times)}
			senders_number = len(np.unique(senders))

			# calculate mean value of voltage by each unique time
			for time, voltage, g_ex_elem, g_in_elem in zip(times, voltages, g_ex, g_in):
				voltage_data[time] += voltage
				g_ex_data[time] += g_ex_elem
				g_in_data[time] += g_in_elem
			for time in voltage_data:
				voltage_data[time] /= senders_number
				g_ex_data[time] /= senders_number
				g_in_data[time] /= senders_number
		else:
			for filename in os.listdir(raw_data_path):
				if filename.startswith(test_name) and filename.endswith('.dat'):
					self.logger.info('\t Gathering data from {}'.format(filename))
					with open(os.path.join(raw_data_path, filename)) as file:
						for line in file:
							gid, time, volt, g_ex, g_in = line.split()
							time = float(time)
							volt = float(volt)
							g_ex = float(g_ex)
							g_in = float(g_in)
							if time not in voltage_data.keys():
								voltage_data[time] = []
								g_ex_data[time] = []
								g_in_data[time] = []
							voltage_data[time].append(volt)
							g_ex_data[time].append(g_ex)
							g_in_data[time].append(g_in)
			# calculate mean value of Voltage
			for time in voltage_data.keys():
				voltage_data[time] = np.mean(voltage_data[time])
		return voltage_data, g_ex_data, g_in_data


	def gather_spikes(self, node_name):
		"""
		Gather all spike times from .gdf files
		Args:
			node_name (str):
				name of the node
		Returns:
			dict[int, list]: GID of neuron and spiketimes
		"""
		spikes_data = {}
		self.logger.info('Searching for {}'.format(node_name))
		for filename in os.listdir(raw_data_path):
			if node_name in filename and filename.endswith('.gdf'):
				self.logger.info('\t Gathering data from {}'.format(filename))
				with open(os.path.join(raw_data_path, filename)) as file:
					for line in file:
						gid, time = line.split()
						gid = int(gid)
						time = float(time)
						if gid not in spikes_data.keys():
							spikes_data[gid] = []
						spikes_data[gid].append(time)
		return spikes_data
