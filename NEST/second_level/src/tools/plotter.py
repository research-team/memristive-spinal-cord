import os
import sys
import pylab
import logging as log
import numpy as np
from nest import GetStatus
from collections import defaultdict
from ..data import *
from the_second_level.src.tools.miner import Miner
from the_second_level.src.paths import img_path, topologies_path


topology = __import__('{}.{}'.format(topologies_path, sys.argv[1]), globals(), locals(), ['Params'], 0)
Params = topology.Params

thickline = 0.8
boldline = 1.5

log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotter')

class Plotter:
	@staticmethod
	def plot_all_voltages():
		for sublevel in range(Params.NUM_SUBLEVELS.value):
			pylab.subplot(Params.NUM_SUBLEVELS.value + 2, 1, Params.NUM_SUBLEVELS.value - sublevel)
			for group in ['right', 'left']:
				pylab.ylim([-80., 60.])
				Plotter.plot_voltage('{}{}'.format(group, sublevel), '{}{}'.format(group, sublevel+1))
			pylab.legend()
			pylab.grid()
		pylab.subplot(Params.NUM_SUBLEVELS.value + 2, 1, Params.NUM_SUBLEVELS.value + 1)
		pylab.ylim([-80., 60.])
		Plotter.plot_voltage('pool', 'Pool')
		pylab.legend()
		pylab.subplot(Params.NUM_SUBLEVELS.value + 2, 1, Params.NUM_SUBLEVELS.value + 2)
		pylab.ylim([-80., 60.])
		Plotter.plot_voltage('moto', 'Motoneuron')
		pylab.legend()
		Plotter.save_voltage('main_voltages')

		for sublevel in range(Params.NUM_SUBLEVELS.value):
			pylab.subplot(Params.NUM_SUBLEVELS.value, 1, Params.NUM_SUBLEVELS.value - sublevel)
			for group in ['hight', 'heft']:
				pylab.ylim([-80., 60.])
				Plotter.plot_voltage('{}{}'.format(group, sublevel), '{}{}'.format(group, sublevel+1))
			pylab.legend()
		Plotter.save_voltage('hidden_tiers')

	@staticmethod
	def plot_10test(num_slices=6, name='moto', plot_mean=False, from_memory=True):
		"""
		Method for search and plotting test data
		Args:
			num_slices (int): number of slices
			name (str): neuron group name
			plot_mean (bool): choose what to plot: mean (True) or all tests (False)
		"""
		step = 0.1
		test_number = 10
		period = 1000 / Params.RATE.value
		num_dots = int(1 / step * num_slices * period)
		shift = Params.PLOT_SLICES_SHIFT.value
		shift_dots = int(1 / step * shift)
		slices = {}
		test_colors = {}

		# create the main body of the figure and it's subplots
		fig = pylab.figure(figsize=(9, 3))
		fig.suptitle("Rate {} Hz, Inh {}%".format(Params.RATE.value, 100 * Params.INH_COEF.value), fontsize=11)
		# collect data from each test
		test_names = ["{}-{}".format(test_n, name) for test_n in range(test_number)]

		# every slice keep 10 tests, each test contain voltage results
		for index, test_name in enumerate(test_names):
			test_colors[test_name] = colors[index]
			test_data = Miner.gather_mean_voltage(test_name, from_memory=from_memory)
			raw_times = sorted(test_data.keys())[shift_dots-1:num_dots + shift_dots]    # make [8, 33]
			fraction = float(len(raw_times)) / num_slices
			for slice_number in range(num_slices):
				start_time = int(slice_number * fraction)
				end_time = int((slice_number + 1) * fraction) if slice_number < num_slices - 1 else len(raw_times) - 1
				values = [test_data[time] for time in raw_times[start_time:end_time]]
				if slice_number not in slices.keys():
					slices[slice_number] = dict()
				slices[slice_number][test_name] = values
		yticks = []
		for slice_number, tests in slices.items():
			offset = slice_number * 40
			yticks.append(tests[list(tests.keys())[0]][0] + offset)
			local_offset = np.arange(-14, 14, 2.9)
			index = 0
			for test_name, voltages in tests.items():
				pylab.plot([time / 10 for time in range(len(voltages))],
				           [v + offset + local_offset[index] for v in voltages],
				           linewidth=0.5,
				           color='gray')
				index += 1
			if plot_mean:
				mean_data = list(map(lambda elements: sum(elements) / test_number,
				                     zip(*[voltages for test_name, voltages in tests.items()])))
				pylab.plot([time / 10 for time in range(len(mean_data))],
				           [v + offset for v in mean_data],
				           color='gray',
				           linewidth=2,
				           linestyle='--')
			if slice_number in [0, 1]:
				pylab.plot([13, 13], [-100, 250], linewidth=boldline, color='r')
			if slice_number == 2:
				pylab.plot([15, 15], [-10, 250], linewidth=boldline, color='r')
			if slice_number in [3, 4]:
				pylab.plot([17, 17], [30, 250], linewidth=boldline, color='r')
			if slice_number == 5:
				pylab.plot([21, 21], [120, 250], linewidth=boldline, color='r')
		pylab.axvline(x=5, linewidth=thickline, color='r')

		# global plot settings
		#pylab.yticks(yticks, range(1, num_slices+1))
		pylab.xticks(range(26), [i if i % 5 == 0 else "" for i in range(26)])
		pylab.xlim(0, 25)
		pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
		pylab.subplots_adjust(left=0.03, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
		pylab.gca().invert_yaxis()
		pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)

		pylab.show()
		# save the plot
		fig_name = 'slices{}Hz-{}Inh-{}sublevels_mean'.format(Params.RATE.value,
		                                                      Params.INH_COEF.value,
		                                                      Params.NUM_SUBLEVELS.value)
		pylab.savefig(os.path.join(img_path, '{}.png'.format(fig_name)), dpi=200)
		pylab.close('all')
		logger.info('saved to "{}"'.format(os.path.join(img_path, '{}.png'.format(fig_name))))


	@staticmethod
	def plot_slices(num_slices=6, name='moto', only_show=False):
		"""
		Args:
			num_slices (int):
			name (str):
			only_show (bool):
		"""
		pylab.figure(figsize=(9, 3))
		period = 1000 / Params.RATE.value
		step = 0.1
		shift = Params.PLOT_SLICES_SHIFT.value
		interval = period
		# get data from memory/hard disk
		data = Miner.gather_mean_voltage(name, from_memory=True)
		num_dots = int(1 / step * num_slices * interval)
		shift_dots = int(1 / step * shift) - 1 # make to [0,8)
		# slice unnecessary times
		raw_times = sorted(data.keys())[shift_dots:num_dots + shift_dots]
		fraction = float(len(raw_times)) / num_slices

		pylab.suptitle("Rate {} Hz, Inh {}%".format(Params.RATE.value, 100 * Params.INH_COEF.value), fontsize=10)
		yticks = []
		for slice_n in range(num_slices):
			start = int(slice_n * fraction)
			end = int((slice_n + 1) * fraction) if slice_n < num_slices - 1 else len(raw_times) - 1
			voltages = [data[time] for time in raw_times[start:end]]
			offset = slice_n*30
			pylab.plot([time / 10 for time in range(len(voltages))],
			           [v + offset for v in voltages],
			           color='gray',
			           linewidth=1)
			yticks.append(voltages[0] + offset)
		pylab.yticks(yticks, range(1, num_slices+1))
		pylab.xticks(range(26), [i if i % 5 == 0 else "" for i in range(26)])
		pylab.xlim(0, 25)
		pylab.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
		pylab.subplots_adjust(left=0.03, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
		pylab.gca().invert_yaxis()

		if only_show:
			pylab.show()
		else:
			name = 'slices{}Hz-{}Inh-{}sublevels'.format(Params.RATE.value, Params.INH_COEF.value, Params.NUM_SUBLEVELS.value)
			pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=200)
			pylab.close('all')
			logger.info('saved to "{}"'.format(os.path.join(img_path, '{}.png'.format(name))))

	@staticmethod
	def plot_voltage(group_name, label, with_spikes=False):
		pylab.figure(figsize=(9, 5))
		try:
			GetStatus(multimeters_dict[group_name])
			GetStatus(spikedetectors_dict[group_name])
		except Exception:
			print("Not found devices with this name", group_name)
			return
		voltage_values = Miner.gather_mean_voltage(group_name, from_memory=True)
		pylab.plot(voltage_values.keys(), voltage_values.values(), label=label)

		if with_spikes:
			spike_values = GetStatus(spikedetectors_dict[group_name])[0]['events']['times']
			pylab.plot(spike_values, [0 for _ in spike_values], ".", color='r')

	@staticmethod
	def save_voltage(name):
		pylab.xlabel('Time, ms')
		pylab.xlim(0, Params.SIMULATION_TIME.value + 1)
		for i in np.arange(0, Params.SIMULATION_TIME.value, 5):
			pylab.axvline(x=i, linewidth=thickline, color='gray')
		pylab.ylim(-90, 50)
		pylab.legend()
		pylab.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
		pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=200)
		pylab.close('all')

	@staticmethod
	def plot_all_spikes():
		for sublevel in range(Params.NUM_SUBLEVELS.value):
			pylab.subplot(Params.NUM_SUBLEVELS.value + 2, 1, Params.NUM_SUBLEVELS.value - sublevel)
			pylab.title('sublevel {}'.format(sublevel+1))
			if Plotter.has_spikes('{}{}'.format('right', sublevel)):
				pylab.xlim([0., simulation_time])
				Plotter.plot_spikes('{}{}'.format('right', sublevel), color='b')
			if Plotter.has_spikes('{}{}'.format('left', sublevel)):
				pylab.xlim([0., simulation_time])
				Plotter.plot_spikes('{}{}'.format('left', sublevel), color='r')
			pylab.legend()
		pylab.subplot(Params.NUM_SUBLEVELS.value + 2, 1, Params.NUM_SUBLEVELS.value + 1)
		pylab.title('Pool')
		if Plotter.has_spikes('pool'):
			pylab.xlim([0., simulation_time])
			Plotter.plot_spikes('pool', color='b')
			pylab.legend()
		pylab.subplot(Params.NUM_SUBLEVELS.value + 2, 1, Params.NUM_SUBLEVELS.value + 2)
		pylab.title('Motoneuron')
		if Plotter.has_spikes('moto'):
			pylab.xlim([0., simulation_time])
			Plotter.plot_spikes('moto', color='r')
			pylab.legend()
		Plotter.save_spikes('spikes')

	@staticmethod
	def plot_spikes(name, color):
		results = Miner.gather_spikes(name)
		gids = sorted(list(results.keys()))
		events = [results[gid] for gid in gids]
		pylab.eventplot(events, lineoffsets=gids, linestyles='dotted', color=color)

	@staticmethod
	def save_spikes(name: str):
		pylab.xlabel('Time, ms')
		pylab.subplots_adjust(hspace=0.7)
		pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=300)
		pylab.close('all')

	@staticmethod
	def has_spikes(name):
		"""
		Args:
			name: neurons group name
		Returns:
			bool:
		"""
		return Miner.has_spikes(name)
