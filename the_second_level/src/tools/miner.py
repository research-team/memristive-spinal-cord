import os
import logging as log
import numpy as np
from ..data import *
from nest import GetStatus
from collections import defaultdict
from the_second_level.src.paths import raw_data_path, spiketimes_path

class Miner:
	@staticmethod
	def gather_voltage(group_name, from_memory=False):
		"""
		Gather voltage from files and calculate mean for each milliseconds
		Args:
			group_name (str): neurons group name
			from_memory (bool):
		Returns:
			list: mean voltage for each millisecond
		"""
		data = {}
		log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
		logger = log.getLogger('Miner')
		logger.info('searching for "{}"'.format(group_name))

		if from_memory:
			# get values from memory
			voltage = GetStatus(multimeters_dict[group_name])[0]['events']['V_m']
			times = GetStatus(multimeters_dict[group_name])[0]['events']['times']
			gids = np.unique(GetStatus(multimeters_dict[group_name])[0]['events']['senders'])
			# initialize dict with zeroes by each unique time
			for i in np.unique(times):
				data[float(i)] = 0
			# fill data with voltage by each time
			for i in range(len(times)):
				data[float(times[i])] += float(voltage[i])
			# calculate mean value of voltage by each unique time
			for time in np.unique(times):
				data[time] /= len(gids)
		else:
			neurons = []
			for filename in os.listdir(raw_data_path):
				if filename.startswith(group_name) and filename.endswith('.dat'):
					logger.info('  Gathering data from {}'.format(filename))
					with open(os.path.join(raw_data_path, filename)) as filedata:
						for line in filedata:
							gid, time, value = line.split()
							if gid not in neurons:
								neurons.append(gid)
							if float(time) not in data.keys():
								data[float(time)] = 0.
							data[float(time)] += float(value)
			# calculate mean value of Voltage
			for time in data.keys():
				data[float(time)] /= float(len(neurons))
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