import pylab
import os
from tsl_simple.src.tools.miner import Miner
from tsl_simple.src.paths import img_path


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
    def plot_spikes(name, color):
        results = Miner.gather_spikes(name)
        gids = sorted(list(results.keys()))
        events = [results[gid] for gid in gids]
        pylab.eventplot(events, lineoffsets=gids, linestyles='dotted', color=color)

    @staticmethod
    def save_spikes(name: str):
        pylab.ylabel('Neuron gid')
        pylab.xlabel('Time, ms')
        # pylab.xlim([0., 100.])
        pylab.savefig(os.path.join(img_path, '{}.png'.format(name)))
        pylab.close('all')

    @staticmethod
    def has_spikes(name: str) -> bool:
        return Miner.has_spikes(name)
