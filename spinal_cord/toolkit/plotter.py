import shutil
from pkg_resources import resource_filename
import os
import pylab
import matplotlib
matplotlib.use('Agg')

from spinal_cord.params import Params
from spinal_cord.toolkit.data_miner import DataMiner
from datetime import datetime

def clear_results(name=None):
    results_dir_filename = resource_filename('spinal_cord', 'results')
    if not name:
        if os.path.isdir(results_dir_filename):
            shutil.rmtree(results_dir_filename)
        os.mkdir(results_dir_filename)
        os.mkdir(os.path.join(results_dir_filename, 'img'))
        os.mkdir(os.path.join(results_dir_filename, 'raw_data'))
    else:
        dt = datetime.now()
        name = '{}_{}'.format(dt, name)
        if os.path.isdir(results_dir_filename):
            if not os.path.isdir(os.path.join(results_dir_filename, 'img')):
                os.mkdir(os.path.join(results_dir_filename, 'img'))
            os.mkdir(os.path.join(results_dir_filename, 'img', name))
            if os.path.isdir(os.path.join(results_dir_filename, 'raw_data')):
                shutil.rmtree(os.path.join(results_dir_filename, 'raw_data'))
            os.mkdir(os.path.join(results_dir_filename, 'raw_data'))

class ResultsPlotter:
    def __init__(self, rows_number, title, filename):
        pylab.ioff()
        params = {'legend.fontsize': 'x-small',
                  'axes.labelsize': 'x-small',
                  'axes.titlesize': 'x-small',
                  'xtick.labelsize': 'x-small'}
        pylab.rcParams.update(params)
        if rows_number == 7:
            pylab.rcParams.update({'ytick.labelsize': 4})
            self.blue_line_style = 'b'
            self.red_line_style = 'r'
        else:
            pylab.rcParams.update({'ytick.labelsize': 8})
            self.blue_line_style = 'b:'
            self.red_line_style = 'r--'
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
        pylab.savefig(os.path.join(resource_filename('spinal_cord', 'results'), 'img', '{}_{}Hz'.format(self.filename, Params.rate.value)))

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
                self.red_line_style,
                label=first_label
            )
        if second:
            data = DataMiner.get_average_voltage(second)
            times = sorted(list(data.keys()))
            values = [data[time] for time in times]
            pylab.plot(
                times,
                values,
                self.blue_line_style,
                label=second_label,
                linewidth=1.
            )

        pylab.ylabel(title, fontsize=11)
        pylab.legend(fontsize=11)

    def subplot_with_slices(self, slices: int, title: str, first=None, first_label: str=None, second=None, second_label: str=None, third=None, third_label=None):
        if slices > self.rows_number:
            raise ValueError('Too much subplots')
        step = .1
        shift = 10.
        interval = 1000 / Params.rate.value
        data1 = DataMiner.get_average_voltage(first)
        number_of_dots = int(1 / step * slices * interval)
        shift_dots = int(1 / step * shift)
        times1 = sorted(data1.keys())[shift_dots:number_of_dots+shift_dots]
        data2 = DataMiner.get_average_voltage(second)
        if third:
            data3 = DataMiner.get_average_voltage(third)
        fraction = len(times1) / slices
        for slice in range(slices):
            # pylab.ylim(ymax=-60)
            pylab.subplot(self.rows_number, self.cols_number, self.plot_index)
            self.plot_index += 1
            start = int(slice * fraction)
            end = int((slice + 1) * fraction) if slice < 6 else len(times1) - 1
            times = times1[start:end]
            values = [data1[time] for time in times]
            pylab.plot(
                times,
                values,
                self.blue_line_style,
                label=first_label
            )
            values = [data2[time] for time in times]
            pylab.plot(
                times,
                values,
                self.red_line_style,
                label=second_label
            )
            if third:
                values = [data3[time] for time in times]
                pylab.plot(
                    times,
                    values,
                    'g',
                    label=third_label
                )
            pylab.ylabel(title, fontsize=8)
            pylab.legend(fontsize=6)
