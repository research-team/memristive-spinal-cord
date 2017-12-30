import shutil
import os
import pylab


class ToolKit():

    def __init__(self, path, dirname):
        self.path = path
        self.dirname = dirname

    def clear_results(self, dirname=None):
        if not dirname: dirname = self.dirname
        try:
            shutil.rmtree(path=os.path.join(self.path, dirname))
            os.mkdir(path=os.path.join(self.path, dirname))
        except FileNotFoundError:
            pass

    def plot_left_column(self):
        pass # TODO implement plot_left_column()

    def plot_interneuronal_pool(self, show_results: bool=False):
        __pool_slices = {}
        __stimuli_slices = {}
        pool_times = 'pool_times'
        pool_voltages = 'pool_voltages'
        stimuli_times = 'stimuli_times'
        stimuli_voltages = 'stimuli_voltages'

        pool = open('{}/{}'.format(os.path.join(self.path, self.dirname), 'InterneuronalPool [Glu].dat'))
        stimuli = open('{}/{}'.format(os.path.join(self.path, self.dirname), 'Mediator [Glu].dat'))

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

        if show_results: pylab.show()
        self.clear_results('images')
        os.mkdir(os.path.join(self.path, 'images/pool'))
        pylab.savefig(fname=os.path.join(self.path, 'images/pool/pool.png'))