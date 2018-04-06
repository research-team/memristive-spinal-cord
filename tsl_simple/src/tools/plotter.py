import pylab
import os
from tsl_simple.src.tools.miner import Miner
from tsl_simple.src.paths import img_path
from tsl_simple.src.params import num_sublevels, simulation_time


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
        Plotter.save_voltage('voltages')

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
        pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=500)
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
        pylab.subplots_adjust(hspace=0.6)
        pylab.savefig(os.path.join(img_path, '{}.png'.format(name)), dpi=500)
        pylab.close('all')

    @staticmethod
    def has_spikes(name: str) -> bool:
        return Miner.has_spikes(name)
