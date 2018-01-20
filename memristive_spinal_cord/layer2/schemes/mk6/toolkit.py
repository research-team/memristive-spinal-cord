import shutil
import sys

from memristive_spinal_cord.layer2.toolkit import ToolKit
import os
import pylab


class Plotter(ToolKit):
    def plot_tier(self, *tiers):
        for tier in tiers:
            # data = {'e{}'.format(str(i)): {'times': [], 'voltages': []} for i in range(6)}
            # data.update({'i{}'.format(str(i)): {'times': [], 'voltages': []} for i in range(2)})

            for index in range(6):
                times = []
                voltages = []
                try:
                    raw_data = open(os.path.join(self.path, self.raw_data_dirname, 'Tier{}E{} [Glu].dat'.format(tier, index)), 'r')
                except FileNotFoundError:
                    print(sys.exc_info()[1])
                    pass
                for line in raw_data.readlines():
                    time, voltage = [float(value) for value in line.split()[1:]]
                    times.append(time)
                    voltages.append(voltage)
                pylab.subplot(4, 2, index + 1)
                pylab.title('Tier{}E{}'.format(tier, index))
                pylab.plot(times, voltages)
            for index in range(2):
                times = []
                voltages = []
                try:
                    raw_data = open(os.path.join(self.path, self.raw_data_dirname, 'Tier{}I{} [GABA].dat'.format(tier, index)), 'r')
                except FileNotFoundError:
                    print(sys.exc_info()[1])
                for line in raw_data.readlines():
                    time, voltage = [float(value) for value in line.split()[1:]]
                    times.append(time)
                    voltages.append(voltage)
                pylab.subplot(4, 2, index + 7)
                pylab.title('Tier{}I{}'.format(tier, index))
                pylab.plot(times, voltages)

            pylab.savefig(fname=os.path.join(self.path, '{}/pool.png'.format(self.figures_dirname)))
            pylab.close('all')