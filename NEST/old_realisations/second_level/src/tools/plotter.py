import os
import sys
import pylab
import datetime
import numpy as np
import logging as log
import matplotlib.patches as mpatches

sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-4]))
print('/'.join(os.path.realpath(__file__).split('/')[:-5]))

from nest import GetStatus
from collections import defaultdict
from NEST.second_level.src.data import *
from NEST.second_level.src.namespace import *
from NEST.second_level.src.tools.miner import Miner
from NEST.second_level.src.paths import img_path, topologies_path


log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotter')

boldline = 1.5
thickline = 0.8
sim_step = 0.1

class Plotter:
	def __init__(self, simulation_parameters):
		"""
		Init variables and create an instance of the Miner
		Args:
			simulation_parameters (dict):
				parameters of the simulation
		"""
		self.model = simulation_parameters[Params.MODEL.value]
		self.ees_rate = simulation_parameters[Params.EES_RATE.value]
		self.inh_coef = simulation_parameters[Params.INH_COEF.value]
		self.speed = simulation_parameters[Params.SPEED.value]
		self.sim_time = simulation_parameters[Params.SIM_TIME.value]
		self.record_from = simulation_parameters[Params.RECORD_FROM.value]
		self.miner = Miner()

		log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
		self.logger = log.getLogger('Plotter')


	def __split_to_slices(self, test_names, from_memory):
		"""
		Slice data to slices by 25 ms
		Args:
			test_names (list):
				 test names
		Returns:
			dict[int, dict[str, list]]: data structure for keeping for slice -> (test -> voltages)
		"""
		num_slices = self.sim_time // 25
		voltages = {k: defaultdict(list) for k in range(num_slices) }
		for test_name in test_names:
			self.logger.info("Plot the figure {}".format(test_name))
			#test_voltage_data, g_ex, g_in = self.miner.gather_mean_voltage(test_name, from_memory=from_memory)
			test_voltage_data = self.miner.gather_mean_voltage(test_name, from_memory=from_memory)[0]
			# split data for slices
			for time in sorted(test_voltage_data.keys()):
				slice_number = int(time // 25)
				voltages[slice_number][test_name].append(test_voltage_data[time])
		return voltages


	def plot_slices(self, tests_number=1, from_memory=True, test_name="MP_E"):
		"""
		Method for search and plotting results data
		Args:
			tests_number (int):
				number of tests
			from_memory (bool):
				True - get results from the RAM, False - from the files
		"""

		yticks = []

		# collect data for Motoneuron pool
		test_names = ["{}-{}".format(test_number, test_name) for test_number in range(tests_number)]
		moto_voltage = self.__split_to_slices(test_names, from_memory)

		# plot data
		pylab.figure(figsize=(10, 5))
		pylab.suptitle("Model {} Speed {} cm/s Rate {} Hz Inh {}%".format(self.model,
		                                                                  self.speed,
		                                                                  self.ees_rate,
		                                                                  100 * self.inh_coef), fontsize=11)

		# plot each slice
		for slice_number, tests in moto_voltage.items():
			if self.record_from == 'V_m':
				offset = -slice_number * 10
			else:
				offset = -slice_number * 1500

			i = -1 if self.record_from == 'V_m' else 1

			yticks.append(i * tests[list(tests.keys())[0]][0] + offset)
			# collect mean data: sum values (mV) step by step (ms) per test and divide by test number
			mean_data = list(map(lambda elements: np.mean(elements), zip(*tests.values())))  #
			times = [time * sim_step for time in range(len(mean_data))]   # divide by 10 to convert to ms step

			means = [i * voltage + offset for voltage in mean_data]
			minimal_per_step = [min(a) for a in zip(*tests.values())]
			maximal_per_step = [max(a) for a in zip(*tests.values())]
			# plot mean with shadows
			pylab.plot(times, means, linewidth=0.5, color='k')
			pylab.fill_between(times,
							   [i * mini + offset for mini in minimal_per_step],  # the minimal values + offset (slice number)
							   [i * maxi + offset for maxi in maximal_per_step],  # the maximal values + offset (slice number)
							   alpha=0.35)

			if self.speed == 6:
				num = 5
			elif self.speed == 15:
				num = 2
			elif self.speed == 21:
				num = 1
			else:
				raise Exception("Can't recognize the speed")
			# plot lines to see when activity should be started
		pylab.axvline(x=13, color='r', linewidth=thickline)
		pylab.axvline(x=15, color='r', linewidth=thickline)
		pylab.axvline(x=17, color='r', linewidth=thickline)
		pylab.axvline(x=21, color='r', linewidth=thickline)

		# global plot settings
		pylab.xlim(0, 25)
		pylab.xticks(range(26), [i if i % 5 == 0 else "" for i in range(26)])
		pylab.yticks(yticks, range(1, self.sim_time // 25 + 1))
		pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
		#pylab.subplots_adjust(left=0.03, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)

		# save the plot
		fig_name = 'sim_{}_eesF{}_i{}_s{}cms_{}'.format(self.model,
														self.ees_rate,
														self.inh_coef * 100,
														self.speed,
														datetime.datetime.today().strftime('%Y-%m-%d'))
		pylab.savefig(os.path.join(img_path, '{}.png'.format(fig_name)), dpi=200)
		pylab.close('all')
		logger.info("saved to '{}'".format(os.path.join(img_path, '{}.png'.format(fig_name))))


	def plot_voltage(self, group_name, with_spikes=False):
		"""
		Plotting membrane potential of the nodes
		Args:
			group_name (str):
				name of the node
			with_spikes (bool):
				True - additional plotting of spiking activity,
				False - plot only voltage
		"""
		try:
			GetStatus(multimeters_dict[group_name])
			GetStatus(spikedetectors_dict[group_name])
		except Exception:
			logger.info("Not found device '{}'".format(group_name))
			return

		# plot results
		voltage_values, g_ex_data, g_in_data = self.miner.gather_mean_voltage(group_name, from_memory=True)
		self.logger.info("plot '{}'".format(group_name))

		pylab.figure(figsize=(10, 5))
		pylab.suptitle(group_name)

		# VOLTAGE
		pylab.subplot(2, 1, 1)
		pylab.plot(voltage_values.keys(), voltage_values.values())

		# plot spikes
		if with_spikes:
			spike_values = GetStatus(spikedetectors_dict[group_name])[0]['events']['times']
			pylab.plot(spike_values, [0] * len(spike_values), ".", color='r', markersize=1)
		# plot the slice border
		for i in np.arange(0, self.sim_time, 25):
			pylab.axvline(x=i, linewidth=boldline, color='k')
		pylab.ylabel('Voltage [mV]')
		pylab.xticks(range(0, self.sim_time + 1, 5),
					 [""] * ((self.sim_time + 1) // 5))
		pylab.xlim(0, self.sim_time)
		pylab.grid()

		# CURRENTS
		pylab.subplot(2, 1, 2)
		pylab.plot(g_ex_data.keys(), g_ex_data.values(), color='r')
		pylab.plot(g_in_data.keys(), g_in_data.values(), color='b')
		# plot the slice border
		for i in np.arange(0, self.sim_time, 25):
			pylab.axvline(x=i, linewidth=boldline, color='k')

		pylab.xlabel('Time [ms]')
		pylab.ylabel('Currents [pA]')
		pylab.xlim(0, self.sim_time)
		pylab.xticks(range(0, self.sim_time + 1, 5),
					 ["{}\n{}".format(global_time - 25 * (global_time // 25), global_time)
					  for global_time in range(0, self.sim_time + 1, 5)],
					 fontsize=8)
		pylab.yticks(fontsize=8)
		pylab.grid()
		pylab.subplots_adjust(hspace=0.05)
		pylab.savefig(os.path.join(img_path, '{}.png'.format(group_name)), dpi=200)
		pylab.close('all')


if __name__ == "__main__":
	simulation_params = {
		Params.MODEL.value: "extensor",
		Params.EES_RATE.value: 40,
		Params.RECORD_FROM.value: 'V_m',
		Params.INH_COEF.value: 1,
		Params.SPEED.value: 21,
		Params.C_TIME.value: 25,
		Params.SIM_TIME.value: 25 * 6,  # flexor 5, extensor 6
		Params.ESS_THRESHOLD.value: True,
		Params.MULTITEST.value: True
	}

	k = Plotter(simulation_params)
	k.plot_slices(tests_number=5, from_memory=False, test_name='MP_E')
