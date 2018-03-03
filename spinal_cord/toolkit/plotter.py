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
        pylab.subplots_adjust(left=0.05, right=0.99, hspace=0.15*self.rows_number)
        pylab.xlabel('ms')
        pylab.savefig(os.path.join(resource_filename('spinal_cord', 'results'), 'img', self.filename))

    def subplot(self, title: str, first=None, first_label: str=None, second=None, second_label: str=None):
        if self.plot_index > self.rows_number:
            raise ValueError("Too many subplots!")
        pylab.subplot(self.rows_number, self.cols_number, self.plot_index)
        self.plot_index += 1

        if first:
            data = DataMiner.get_average_voltage(first)
            times = sorted(list(data.keys()))
            values = [data[time] for time in times]
            pylab.plot(
                times,
                values,
                'r--',
                label=first_label)
        if second:
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

    def subplot_with_slices(self, slices: int, title: str, first=None, first_label: str=None, second=None, second_label: str=None, third=None, third_label=None):
        if slices > self.rows_number:
            raise ValueError('Too much subplots')

        data1 = DataMiner.get_average_voltage(first)
        times1 = sorted(data1.keys())
        data2 = DataMiner.get_average_voltage(second)
        data3 = DataMiner.get_average_voltage(third)
        fraction = len(data1) / slices
        for slice in range(slices):
            pylab.subplot(self.rows_number, self.cols_number, self.plot_index)
            self.plot_index += 1
            start = int(slice * fraction)
            end = int((slice + 1) * fraction) if slice < 6 else len(times1) - 1
            times = times1[start:end]
            values = [data1[time] for time in times]
            pylab.plot(
                times,
                values,
                'b:',
                label=first_label)
            values = [data2[time] for time in times]
            pylab.plot(
                times,
                values,
                'r--',
                label=second_label)
            values = [data3[time] for time in times]
            pylab.plot(
                times,
                values,
                'g',
                label=third_label)

        pylab.ylabel(title, fontsize=11)
        pylab.legend()
