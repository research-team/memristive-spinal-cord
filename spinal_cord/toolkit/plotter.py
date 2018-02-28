import shutil
from pkg_resources import resource_filename
import os
import pylab
from spinal_cord.toolkit.data_miner import DataMiner


def clear_results():
    results_dir_filename = resource_filename('spinal_cord', 'results')
    if os.path.isdir(results_dir_filename):
        shutil.rmtree(results_dir_filename)
        os.mkdir(results_dir_filename)
        os.mkdir(os.path.join(results_dir_filename, 'img'))
        os.mkdir(os.path.join(results_dir_filename, 'raw_data'))
    else:
        os.mkdir(results_dir_filename)
        os.mkdir(os.path.join(results_dir_filename, 'img'))
        os.mkdir(os.path.join(results_dir_filename, 'raw_data'))


class ResultsPlotter:
    def __init__(self, rows_number, title, filename):
        self.rows_number = rows_number
        self.cols_number = 1
        self.plot_index = 1
        self.title = title
        self.filename = filename
        w, h = pylab.figaspect(.25)
        self.a = pylab.figure(figsize=(w, h))
        self.a.suptitle(title)

    def save(self):
        pylab.subplots_adjust(left=0.05, right=0.99, hspace=0.1*self.rows_number)
        pylab.xlabel('ms')
        pylab.savefig(os.path.join(resource_filename('spinal_cord', 'results'), 'img', self.filename))

    def subplot(self, title: str, first, first_label: str, second=None, second_label: str=None):
        if self.plot_index > self.rows_number:
            raise ValueError("Too many subplots!")
        pylab.subplot(self.rows_number, self.cols_number, self.plot_index)
        self.plot_index += 1

        data = DataMiner.get_average_voltage(first)
        times = sorted(list(data.keys()))
        values = [data[time] for time in times]
        pylab.plot(
            times,
            values,
            'r--',
            label=first_label)
        data = DataMiner.get_average_voltage(second)
        times = sorted(list(data.keys()))
        values = [data[time] for time in times]
        pylab.plot(
            times,
            values,
            'b:',
            label=second_label)

        pylab.ylabel(title, fontsize=11)
        pylab.legend()
