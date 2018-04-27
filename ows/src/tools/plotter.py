import pylab
import os
from ows.src.tools.miner import Miner
from ows.src.paths import img_path
from ows.src.params import num_sublevels, simulation_time, rate, inh_coef
import logging


class Plotter:

    @staticmethod
    def plot_all_voltages():
        for sublevel in range(num_sublevels):
            pylab.subplot(num_sublevels + 2, 1, num_sublevels - sublevel)
            for group in ['right', 'left']:
                pylab.ylim([-80., 60.])
                Plotter.plot_voltage('{}{}'.format(group, sublevel),
                    '{}{}'.format(group, sublevel+1))
            pylab.legend()
        pylab.subplot(num_sublevels + 2, 1, num_sublevels + 1)
        pylab.ylim([-80., 60.])
        Plotter.plot_voltage('pool', 'Pool')
        pylab.legend()
        pylab.subplot(num_sublevels + 2, 1, num_sublevels + 2)
        pylab.ylim([-80., 60.])
        Plotter.plot_voltage('moto', 'Motoneuron')
        pylab.legend()
        Plotter.save_voltage('main_voltages')

        for sublevel in range(num_sublevels):
            pylab.subplot(num_sublevels, 1, num_sublevels - sublevel)
            for group in ['hight', 'heft']:
                pylab.ylim([-80., 60.])
                Plotter.plot_voltage('{}{}'.format(group, sublevel),
                    '{}{}'.format(group, sublevel+1))
            pylab.legend()
        Plotter.save_voltage('hidden_tiers')

    @staticmethod
    def plot_slices(num_slices: int=7, name: str='extensor_moto'):
        period = 1000 / rate
        step = .1
        shift = period
        interval = period
        data = Miner.gather_voltage(name)
        num_dots = int(1 / step * num_slices * interval)
        shift_dots = int(1 / step * shift)
        raw_times = sorted(data.keys())[shift_dots:num_dots + shift_dots]
        fraction = float(len(raw_times)) / num_slices

        pylab.suptitle('Rate = {}Hz, Inh = {}%'.format(rate, 100 * inh_coef), fontsize=14)

        for s in range(num_slices):
            logging.warning('Plotting slice {}'.format(s))
            pylab.subplot(num_slices, 1, s + 1)
            start = int(s * fraction)
            end = int((s + 1) * fraction) if s < num_slices - 1 else len(raw_times) - 1
            logging.warning('starting = {} ({}); end = {} ({})'.format(start, start / 10, end, end / 10))
            times = raw_times[start:end]
            values = [data[time] for time in times]
            pylab.ylim(-80, 60)
            pylab.xlim(start / 10 + shift, end / 10 + shift)
            pylab.plot(times, values)

        Plotter.save_voltage('slices{}Hz-{}Inh-{}sublevels'.format(rate, 100 * inh_coef, num_sublevels))

    @staticmethod
    def plot_all_spikes():
        for sublevel in range(num_sublevels):
            pylab.subplot(num_sublevels + 2, 1, num_sublevels - sublevel)
            pylab.title('sublevel {}'.format(sublevel+1))
            if Plotter.has_spikes('{}{}'.format('right', sublevel)):
                pylab.xlim([0., simulation_time])
                Plotter.plot_spikes('{}{}'.format('right', sublevel), color='b')
            if Plotter.has_spikes('{}{}'.format('left', sublevel)):
                pylab.xlim([0., simulation_time])
                Plotter.plot_spikes('{}{}'.format('left', sublevel), color='r')
            pylab.legend()
        pylab.subplot(num_sublevels + 2, 1, num_sublevels + 1)
        pylab.title('Pool')
        if Plotter.has_spikes('pool'):
            pylab.xlim([0., simulation_time])
            Plotter.plot_spikes('pool', color='b')
            pylab.legend()
        pylab.subplot(num_sublevels + 2, 1, num_sublevels + 2)
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
        pylab.legend()
        pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=120)
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
