import shutil
import os
import pylab


class ToolKit():

    def __init__(self, path, raw_data_dirname, figures_dirname):
        self.path = path
        self.raw_data_dirname = raw_data_dirname
        self.figures_dirname = figures_dirname

        self.clear_results()
        self.clear_results(figures_dirname)

    def clear_results(self, dirname=None):
        if not dirname: dirname = self.raw_data_dirname
        try:
            shutil.rmtree(path=os.path.join(self.path, dirname))
            os.mkdir(path=os.path.join(self.path, dirname))
        except FileNotFoundError:
            pass

    def plot_column(self, show_results: bool=False, column: str='Left'):
        for tier in range(6, 0, -1):
            filename = os.path.join(self.raw_data_dirname, 'Tier{}{} [Glu].dat'.format(str(tier), column))
            with open(filename) as data:
                voltage = []
                time = []
                for line in data.readlines():
                    time.append(float(line.split()[1]))
                    voltage.append(float(line.split()[2]))
            pylab.subplot(6, 1, 7 - tier)
            pylab.plot(time, voltage)
            title = 'Tier{}{}'.format(str(tier), column)
            pylab.title(title)
        if show_results:
            pylab.show()
        else:
            pylab.savefig(fname=os.path.join(self.path, '{}/{}_column.png'.format(self.figures_dirname, column)))
        pylab.close('all')

    def plot_interneuronal_pool(self, show_results: bool=False):
        __pool_slices = {}
        __stimuli_slices = {}
        pool_times = 'pool_times'
        pool_voltages = 'pool_voltages'
        stimuli_times = 'stimuli_times'
        stimuli_voltages = 'stimuli_voltages'

        pool = open('{}/{}'.format(os.path.join(self.path, self.raw_data_dirname), 'InterneuronalPool [Glu].dat'))
        stimuli = open('{}/{}'.format(os.path.join(self.path, self.raw_data_dirname), 'Mediator [Glu].dat'))

        for line in pool.readlines():
            time, voltage = [float(value) for value in line.split()][1:]
            slice_time = int(time // 25.0)
            if slice_time not in __pool_slices.keys():
                __pool_slices[slice_time] = {pool_times: [], pool_voltages: []}
            __pool_slices[slice_time][pool_times].append(time % 25.0)
            __pool_slices[slice_time][pool_voltages].append(voltage)

        for line in stimuli.readlines():
            time, voltage = [float(value) for value in line.split()][1:]
            slice_time = int(time // 25.0)
            if slice_time not in __stimuli_slices.keys():
                __stimuli_slices[slice_time] = {stimuli_times: [], stimuli_voltages: []}
            __stimuli_slices[slice_time][stimuli_times].append(time % 25.0)
            __stimuli_slices[slice_time][stimuli_voltages].append(voltage)

        graphs_num = len(__pool_slices)
        for i in range(len(__pool_slices)):
            pylab.subplot(graphs_num, 1, i + 1)
            pylab.axis([0, 25, -80, -55])
            pylab.plot(__pool_slices[i][pool_times], __pool_slices[i][pool_voltages])
            pylab.plot(__stimuli_slices[i][stimuli_times], __stimuli_slices[i][stimuli_voltages])
            pylab.title('Slice{}'.format(str(i)))

        if show_results:
            pylab.show()
        else:
            pylab.savefig(fname=os.path.join(self.path, '{}/pool.png'.format(self.figures_dirname)))
        pylab.close('all')
