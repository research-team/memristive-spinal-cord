import os
import sys
import pylab
import datetime
import numpy as np
import logging as log
from nest import GetStatus
from collections import defaultdict
from ..data import *
from ..namespace import *

from second_level.src.tools.miner import Miner
from second_level.src.paths import img_path, topologies_path


log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotter')

boldline = 1.5
thickline = 0.8


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
		self.miner = Miner()

		log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
		self.logger = log.getLogger('Plotter')


	def plot_slices(self, num_slices=6, tests_number=1, from_memory=True):
		"""
		Method for search and plotting results data
		Args:
			num_slices (int):
				number of slices
			tests_number (int):
				number of tests
			from_memory (bool):
				True - get results from the RAM, False - from the files
		"""
		slices = {k: defaultdict(list) for k in range(num_slices) }
		yticks = []

		# collect data from each test
		test_names = ["{}-IP_E".format(test_number) for test_number in range(tests_number)]
		for test_name in test_names:
			self.logger.info("Plot the figure {}".format(test_name))
			test_voltage_data = self.miner.gather_mean_voltage(test_name, from_memory=from_memory)
			# split data for slices
			for time in sorted(test_voltage_data.keys()):
				slice_number = int(time // 25)
				slices[slice_number][test_name].append(test_voltage_data[time])

		#for slice_index in range(num_slices):
		#	print(slice_index)
		#	for test_name in test_names:
		#		print("\t", test_name, len(slices[slice_index][test_name]), slices[slice_index][test_name])
		#raise Exception
		# plot data
		pylab.figure(figsize=(16, 9))
		pylab.suptitle("Rate {} Hz, inh {}%".format(self.ees_rate, 100 * self.inh_coef), fontsize=11)



		# plot each slice
		for slice_number, tests in slices.items():
			offset = slice_number * 30
			yticks.append(-tests[list(tests.keys())[0]][0] - offset)
			# collect mean data: sum values (mV) step by step (ms) per test and divide by test number
			mean_data = list(map(lambda elements: np.mean(elements), zip(*tests.values())))  #
			times = [time / 10 for time in range(len(mean_data))]   # divide by 10 to convert to ms step

			means = [-voltage - offset for voltage in mean_data]
			minimal_per_step = [min(a) for a in zip(*tests.values())]
			maximal_per_step = [max(a) for a in zip(*tests.values())]
			# plot mean with shadows
			pylab.plot(times, means, linewidth=1, color='gray')
			pylab.fill_between(times,
			                   [i + offset for i in minimal_per_step],  # the minimal values + offset (slice number)
			                   [i + offset for i in maximal_per_step],  # the maximal values + offset (slice number)
			                   alpha=0.35)
			# plot lines to see when activity should be started
			#if slice_number in [0, 1]:
			#	pylab.plot([13, 13], [means[0]+med, means[0]-med], color='r', linewidth=boldline)
			#if slice_number == 2:
			#	pylab.plot([15, 15], [means[0]+med, means[0]-med], color='r', linewidth=boldline)
			#if slice_number in [3, 4]:
			#	pylab.plot([17, 17], [means[0]+med, means[0]-med], color='r', linewidth=boldline)
			#if slice_number == 5:
			#	pylab.plot([21, 21], [means[0]+med, means[0]-med], color='r', linewidth=boldline)
		pylab.axvline(x=5, linewidth=thickline, color='r')
		# global plot settings
		pylab.xlim(0, 25)
		pylab.xticks(range(26), [i if i % 5 == 0 else "" for i in range(26)])
		pylab.yticks(yticks, range(1, num_slices+1))
		pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
		pylab.subplots_adjust(left=0.03, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
		#pylab.gca().invert_yaxis()


		# save the plot
		fig_name = 'sim_{}_eesF{}_i{}_s{}cms_{}'.format(self.model,
		                                                self.ees_rate,
		                                                self.inh_coef * 100,
		                                                self.speed,
		                                                datetime.datetime.today().strftime('%Y-%m-%d'))
		pylab.savefig(os.path.join(img_path, '{}.png'.format(fig_name)), dpi=200)
		pylab.show()
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

		voltage_values = self.miner.gather_mean_voltage(group_name, from_memory=True)
		self.logger.info("Plot the figure")

		pylab.figure(figsize=(9, 5))
		pylab.plot(voltage_values.keys(), voltage_values.values())

		# plot spikes
		if with_spikes:
			spike_values = GetStatus(spikedetectors_dict[group_name])[0]['events']['times']
			pylab.plot(spike_values, [0 for _ in spike_values], ".", color='r')

		pylab.xlabel('Time, ms')
		pylab.xlim(0, self.sim_time + 1)
		for i in np.arange(0, self.sim_time, 5):
			pylab.axvline(x=i, linewidth=thickline, color='gray')
		for i in np.arange(0, self.sim_time, 25):
			pylab.axvline(x=i, linewidth=boldline, color='k')
		# pylab.ylim(-900, 300)
		pylab.legend()
		pylab.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
		pylab.savefig(os.path.join(img_path, '{}.png'.format(group_name)), dpi=200)
		pylab.close('all')
