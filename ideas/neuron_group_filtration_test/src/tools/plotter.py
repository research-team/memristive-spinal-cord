import pylab
import os
from neuron_group_filtration_test.src.tools.miner import Miner
from neuron_group_filtration_test.src.paths import img_path


class Plotter:

    @staticmethod
    def plot_voltage(name, label):
        results = Miner.gather_voltage(name)
        times = sorted(results.keys())
        values = [results[time] for time in times]
        pylab.plot(times, values, label=label)

    @staticmethod
    def save_voltage(name):
        pylab.ylabel('Average membrane potential, mV')
        pylab.xlabel('Time, ms')
        pylab.legend()
        pylab.savefig(os.path.join(img_path, '{}.png'.format(name)))
        pylab.close('all')

    @staticmethod
    def plot_spikes(name, colors):
        results = Miner.gather_spikes(name)
        gids = sorted(list(results.keys()))
        events = [results[gid] for gid in gids]
        pylab.eventplot(events, lineoffsets=gids, linestyles='dotted', colors=colors)

    @staticmethod
    def save_spikes():
        pylab.ylabel('Neuron gid')
        pylab.xlabel('Time, ms')
        pylab.xlim([0., 200.])
        pylab.savefig(os.path.join(img_path, 'spikes.png'))
        pylab.close('all')
