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


topology = __import__('{}.{}'.format(topologies_path, "new_cpg_concept_NEW"), globals(), locals(), ['Params'], 0)
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
	def plot_10test(num_slices=6, name='moto', plot_mean=False):
		"""
		Method for search and plotting test data
		Args:
			num_slices (int): number of slices
			name (str): neuron group name
			plot_mean (bool): choose what to plot: mean (True) or all tests (False)
		"""
		times = []
		all_data = defaultdict(list)
		mean_data = defaultdict(list)
		step = 0.1
		raw_times = 0
		test_number = 10
		period = 1000 / Params.RATE.value
		shift = Params.PLOT_SLICES_SHIFT.value
		# create the main body of the figure and it's subplots
		fig = pylab.figure(figsize=(9, 9))
		subplots = [fig.add_subplot(num_slices, 1, slice_number + 1) for slice_number in range(num_slices)]
		fig.suptitle("Rate {}Hz, Inh {}%".format(Params.RATE.value, 100 * Params.INH_COEF.value), fontsize=11)
		# collect data from each test
		test_names = ["{}-{}".format(test_n, name) for test_n in range(test_number)]
		for test_name in test_names:
			data = Miner.gather_mean_voltage(test_name)
			num_dots = int(1 / step * num_slices * period)
			shift_dots = int(1 / step * shift)
			raw_times = sorted(data.keys())[shift_dots:num_dots + shift_dots]
			fraction = float(len(raw_times)) / num_slices

			for slice_number in range(num_slices):
				start_time = int(slice_number * fraction)
				end_time = int((slice_number + 1) * fraction) if slice_number < num_slices - 1 else len(raw_times) - 1
				times.append(raw_times[start_time:end_time])
				values = [data[time] for time in raw_times[start_time:end_time]]
				all_data[test_name].append(values)

		# setting plot for each slice
		for slice_number in range(num_slices):
			fraction = float(len(raw_times)) / num_slices
			start_time = int(slice_number * fraction)
			end_time = int((slice_number + 1) * fraction) if slice_number < num_slices - 1 else len(raw_times) - 1

			# subplot settings
			start_xlim = round(start_time / 10 + shift)
			end_xlim = round(end_time / 10 + shift)
			xticks = np.arange(start_xlim, end_xlim + 0.1, step=1.0)
			subplots[slice_number].set_xticks(xticks)
			subplots[slice_number].set_xticklabels([index if index % 5 == 0 else "" for index in range(len(xticks))]
			                                       if slice_number == num_slices-1 else
			                                       ["" for _ in xticks])
			subplots[slice_number].set_ylabel("{}".format(slice_number + 1), fontsize=14, rotation='horizontal')
			subplots[slice_number].set_xlim(start_xlim, end_xlim)
			subplots[slice_number].set_ylim(-100, 60)
			subplots[slice_number].invert_yaxis()
			subplots[slice_number].grid()

		if plot_mean:
			# get data from each SLICE and calculate their mean in each TEST for INDEX by INDEX
			for slice_n in range(num_slices):
				mean_data[slice_n] = list(map(lambda elements: sum(elements) / test_number,
				                              zip(*[all_data[test_name][slice_n] for test_name in test_names])))
		# plot each slice
		for slice_number in range(num_slices):
			# Draw vertical lines
			fraction = float(len(raw_times)) / num_slices
			start_time = int(slice_number * fraction)
			start_xlim = round(start_time / 10 + shift)
			subplots[slice_number].axvline(x=start_xlim + 5, linewidth=thickline, color='r')
			if slice_number in [0, 1]:
				subplots[slice_number].axvline(x=start_xlim + 13, linewidth=boldline, color='r')
			if slice_number == 2:
				subplots[slice_number].axvline(x=start_xlim + 15, linewidth=boldline, color='r')
			if slice_number in [3, 4]:
				subplots[slice_number].axvline(x=start_xlim + 17, linewidth=boldline, color='r')
			if slice_number == 5:
				subplots[slice_number].axvline(x=start_xlim + 21, linewidth=boldline, color='r')
			if plot_mean:
				subplots[slice_number].plot(times[slice_number], mean_data[slice_number])
			else:
				for test_name in ["{}-{}".format(n, name) for n in range(test_number)]:
					subplots[slice_number].plot(times[slice_number], all_data[test_name][slice_number], linewidth=2)
		pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
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
		pylab.figure(figsize=(9, 9))
		period = 1000 / Params.RATE.value
		step = .1
		shift = Params.PLOT_SLICES_SHIFT.value
		interval = period
		# get data from memory/hard disk
		data = Miner.gather_mean_voltage(name, from_memory=True)
		num_dots = int(1 / step * num_slices * interval)
		shift_dots = int(1 / step * shift)
		raw_times = sorted(data.keys())[shift_dots:num_dots + shift_dots]
		fraction = float(len(raw_times)) / num_slices

		pylab.suptitle('Params.RATE.value = {}Hz, Inh = {}%'.format(Params.RATE.value,
		                                                            100 * Params.INH_COEF.value), fontsize=10)
		# im = pylab.imread('/home/alex/Downloads/HTR1 -_ Membrane Transport.png')
		# pylab.imshow(im) #, zorder=0, extent=[0.5, 8.0, 1.0, 7.0])
		for slice_n in range(num_slices):
			# logging.warning('Plotting slice {}'.format(s))
			pylab.subplot(num_slices, 1, slice_n + 1)
			start = int(slice_n * fraction)
			end = int((slice_n + 1) * fraction) if slice_n < num_slices - 1 else len(raw_times) - 1
			# logging.warning('starting = {} ({}); end = {} ({})'.format(start, start / 10, end, end / 10))
			times = raw_times[start:end]
			values = [data[time] for time in times]
			start_xlim = round(start / 10 + shift)
			end_xlim = round(end / 10 + shift)
			ticks = np.arange(start_xlim, end_xlim+0.1, step=1.0)
			# Draw vertical lines
			pylab.axvline(x=start_xlim + 5, linewidth=thickline, color='r')
			if slice_n in [0, 1]:
				pylab.axvline(x=start_xlim + 13, linewidth=boldline, color='r')
			if slice_n == 2:
				pylab.axvline(x=start_xlim + 15, linewidth=boldline, color='r')
			if slice_n in [3, 4]:
				pylab.axvline(x=start_xlim + 17, linewidth=boldline, color='r')
			if slice_n == 5:
				pylab.axvline(x=start_xlim + 21, linewidth=boldline, color='r')
			if slice_n == num_slices-1:
				pylab.xticks(ticks, [str(i) if i % 5 == 0 else "" for i in range(26)])
			else:
				pylab.xticks(ticks, ["" for _ in ticks])
			pylab.grid()
			pylab.ylabel("{}".format(slice_n+1), fontsize=14, rotation='horizontal')
			pylab.xlim(start_xlim, end_xlim)
			pylab.ylim(-100, 60)
			pylab.gca().invert_yaxis()
			pylab.plot(times, values, label='moto')

		pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
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
		logger.info('saved to "{}"'.format(os.path.join(img_path, '{}.png'.format(name))))

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
