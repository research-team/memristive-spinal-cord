import os
import sys
import pylab
import logging as log
import numpy as np
from nest import GetStatus
from ..data import *
from the_second_level.src.tools.miner import Miner
from the_second_level.src.paths import img_path, topologies_path


topology = __import__('{}.{}'.format(topologies_path, "new_cpg_concept_NEW"), globals(), locals(), ['Params'], 0)
Params = topology.Params

thickline = 0.7
boldline = 3

log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotter')

class Plotter:
	@staticmethod
	def plot_all_voltages():
		for sublevel in range(Params.NUM_SUBLEVELS.value):
			pylab.subplot(Params.NUM_SUBLEVELS.value + 2, 1, Params.NUM_SUBLEVELS.value - sublevel)
			for group in ['right', 'left']:
				pylab.ylim([-80., 60.])
				Plotter.plot_voltage('{}{}'.format(group, sublevel),
					'{}{}'.format(group, sublevel+1))
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
	def plot_slices(num_slices=6, name='moto'):
		"""
		Args:
			num_slices (int):
			name (str):
		"""
		pylab.figure(figsize=(9, 12))
		period = 1000 / Params.RATE.value
		step = .1
		shift = Params.PLOT_SLICES_SHIFT.value
		interval = period
		# get data from memory/hard disk
		data = Miner.gather_voltage(name, from_memory=True)
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

			start_xlim = start / 10 + shift
			end_xlim = end / 10 + shift
			ticks = np.arange(start_xlim, end_xlim, step=1.0)
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
				pylab.xticks(ticks, range(26))
			else:
				pylab.xticks(ticks, ["" for _ in ticks])
			pylab.grid()
			pylab.ylabel("{}".format(slice_n+1), fontsize=14, rotation='horizontal')
			pylab.xlim(round(start_xlim), round(end_xlim))
			pylab.ylim(-100, 60)
			pylab.gca().invert_yaxis()
			pylab.plot(times, values, label='moto')

		pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.93, wspace=0.0, hspace=0.09)
		#pylab.show()
		name = 'slices{}Hz-{}Inh-{}sublevels'.format(Params.RATE.value, Params.INH_COEF.value, Params.NUM_SUBLEVELS.value)
		pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=200)
		pylab.close('all')
		logger.info('saved to "{}"'.format(os.path.join(img_path, '{}.png'.format(name))))
		#Plotter.save_voltage()

	@staticmethod
	def plot_voltage(group_name, label, with_spikes=False):
		pylab.figure(figsize=(12, 5))
		try:
			GetStatus(multimeters_dict[group_name])
			GetStatus(spikedetectors_dict[group_name])
		except Exception:
			print("Not found devices with this name", group_name)
			return
		voltage_values = Miner.gather_voltage(group_name, from_memory=True)
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

"""
import pylab
import os
import sys
from the_second_level.src.tools.miner import Miner
from the_second_level.src.paths import img_path, topologies_path
topology = __import__('{}.{}'.format(topologies_path, sys.argv[1]), globals(), locals(), ['Params'], 0)
Params = topology.Params
import logging


class Plotter:

	@staticmethod
	def plot_all_voltages():
		for sublevel in range(Params.NUM_SUBLEVELS.value):
			pylab.subplot(Params.NUM_SUBLEVELS.value + 2, 1, Params.NUM_SUBLEVELS.value - sublevel)
			for group in ['right', 'left']:
				pylab.ylim([-80., 60.])
				Plotter.plot_voltage('{}{}'.format(group, sublevel),
					'{}{}'.format(group, sublevel+1))
			pylab.legend()
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
				Plotter.plot_voltage('{}{}'.format(group, sublevel),
					'{}{}'.format(group, sublevel+1))
			pylab.legend()
		Plotter.save_voltage('hidden_tiers')

	@staticmethod
	def plot_slices(num_slices: int=7, name: str='moto'):
		from collections import defaultdict
		all_datas = defaultdict(list)

		fig = pylab.figure()

		timesGlobal = []
		startGlobal = []
		endGlobal = []

		shift = Params.PLOT_SLICES_SHIFT.value

		subplots = []
		for s in range(num_slices):
			subplots.append(fig.add_subplot(num_slices, 1, s + 1))

		for name in ["{}-".format(n) for n in range(10)]:
			period = 1000 / Params.RATE.value
			step = .1
			interval = period
			data = Miner.gather_voltage(name)
			num_dots = int(1 / step * num_slices * interval)
			shift_dots = int(1 / step * shift)
			raw_times = sorted(data.keys())[shift_dots:num_dots + shift_dots]
			fraction = float(len(raw_times)) / num_slices

			#pylab.suptitle('Params.RATE.value = {}Hz, Inh = {}%'.format(Params.RATE.value, 100 * Params.INH_COEF.value), fontsize=14)

			for s in range(num_slices):
				start = int(s * fraction)
				startGlobal.append(start)

				end = int((s + 1) * fraction) if s < num_slices - 1 else len(raw_times) - 1
				endGlobal.append(end)

				timesGlobal.append(raw_times[start:end])
				times = raw_times[start:end]
				values = [data[time] for time in times]
				subplots[s].set_ylim(-90, 60)
				subplots[s].set_xlim(start / 10 + shift, end / 10 + shift)
				#subplots[s].plot(times, values, linewidth=0.5)

				all_datas[name].append(values)

		## mean data
		new_mean = defaultdict(list)

		for slice in range(6):
			tmp_slice = [0 for _ in all_datas["0-"][slice]]
			for experiment in all_datas:
				# sum elements
				for index in range(len(all_datas[experiment][slice])):
					tmp_slice[index] += all_datas[experiment][slice][index]

			# divide by 6
			tmp_slice = [elem / 10 for elem in tmp_slice]
			new_mean[slice] = list(tmp_slice)


		pylab.suptitle('Params.RATE.value = {}Hz, Inh = {}%'.format(Params.RATE.value, 100 * Params.INH_COEF.value), fontsize=14)
		for s in range(num_slices):
			subplots[s].set_xlim(startGlobal[s] / 10 + shift, endGlobal[s] / 10 + shift)
			subplots[s].set_ylim(-90, 60)
			subplots[s].plot(timesGlobal[s], new_mean[s], linewidth=0.5)

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
	def plot_voltage(name, label):
		results = Miner.gather_voltage(name)
		times = sorted(results.keys())
		values = [results[time] for time in times]
		pylab.plot(times, values, label=label)

	@staticmethod
	def save_voltage(name):
		pylab.xlabel('Time, ms')
		pylab.rcParams['font.size'] = 4
		pylab.ylim(-90, 50)
		pylab.legend()
		pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=520)
		pylab.close('all')

	@staticmethod
	def plot_spikes(name, color):
		results = Miner.gather_spikes(name)
		gids = sorted(list(results.keys()))
		events = [results[gid] for gid in gids]
		pylab.eventplot(events, lineoffsets=gids, linestyles='dotted', color=color)

	@staticmethod
	def save_spikes(name: str):
		pylab.rcParams['font.size'] = 4
		pylab.xlabel('Time, ms')
		pylab.subplots_adjust(hspace=0.7)
		pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=120)
		pylab.close('all')

	@staticmethod
	def has_spikes(name: str) -> bool:
		return Miner.has_spikes(name)

"""