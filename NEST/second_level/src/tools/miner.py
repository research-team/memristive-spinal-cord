import os
import itertools
import numpy as np
import logging as log
from ..data import *
from nest import GetStatus
from collections import defaultdict
from ..paths import raw_data_path, spiketimes_path

class Miner:
	@staticmethod
	def gather_mean_voltage(test_name, from_memory=False):
		"""
		Gather voltage from files and calculate mean for each milliseconds
		Args:
			test_name (str): neurons group name
			from_memory (bool):
		Returns:
			list: mean voltage for each millisecond
		"""
		log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
		logger = log.getLogger('Miner')
		logger.info('[{}] from "{}"'.format("MEM" if from_memory else "FILE", test_name))

		data = {}

		if from_memory:
			# get values from memory
			voltages = GetStatus(multimeters_dict[test_name])[0]['events']['Extracellular']
			times = GetStatus(multimeters_dict[test_name])[0]['events']['times']
			gids = GetStatus(multimeters_dict[test_name])[0]['events']['senders']
			# calculate mean value of voltage by each unique time
			for key, group in itertools.groupby(zip(times, voltages, gids), key=lambda x: x[0]):
				# FixMe: a bug of groupby -- can't use 'group' twice. The next invoke is empty. Use the list()
				tmp = list(group)
				data[float(key)] = sum([volt for time, volt, gid in tmp]) / len([gid for time, volt, gid in tmp])
		else:
			neurons = set()
			for filename in os.listdir(raw_data_path):
				if filename.startswith(test_name) and filename.endswith('.dat'):
					logger.info('  Gathering data from {}'.format(filename))
					with open(os.path.join(raw_data_path, filename)) as filedata:
						for line in filedata:
							gid, time, volt = line.split()
							time = float(time)
							neurons.add(gid)
							if time not in data.keys():
								data[time] = 0.
							data[time] += float(volt)
			# calculate mean value of Voltage
			for time in data.keys():
				data[time] = round(data[time] / len(neurons), 5)
		return data

	@staticmethod
	def gather_spikes(name):
		"""
		Args:
			name (str):
		Returns:
			list:
		"""
		values = dict()
		logger.info('Searching for {}'.format(name))
		for datafile in os.listdir(raw_data_path):
			if name in datafile and '.gdf' in datafile:
				logger.info('Gathering data from {}'.format(datafile))
				with open(os.path.join(raw_data_path, datafile)) as data:
					for line in data:
						gid, time = line.split()
						if int(gid) not in values.keys():
							values[int(gid)] = []
						values[int(gid)].append(float(time))
		return values

	@staticmethod
	def has_spikes(name):
		"""
		Args:
			name (str):
		Returns:
			bool:
		"""
		logger.info('Check {} for spikes'.format(name))
		for datafile in os.listdir(raw_data_path):
			if name in datafile and '.gdf' in datafile:
				with open(os.path.join(raw_data_path, datafile)) as data:
					for line in data:
						if line != '':
							logger.info('{}: spikes found'.format(name))
							return True
		logger.info('{}: spikes not found'.format(name))
		return False

	@staticmethod
	def gather_spiketimes(name):
		"""
		Args:
			name (str):
		"""
		logger.info('Gathering spikes for {}'.format(name))
		for datafile in os.listdir(raw_data_path):
			if name in datafile and '.gdf' in datafile and os.path.getsize(os.path.join(raw_data_path, datafile)) > 0:
				with open(spiketimes_path, 'a') as spikes:
					with open(os.path.join(raw_data_path, datafile)) as data:
						spikes.write('{}\n'.format(name))
						for line in data:
							spikes.write('{}'.format(line))
