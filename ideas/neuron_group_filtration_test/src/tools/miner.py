import os
import logging
from neuron_group_filtration_test.src.paths import raw_data_path


class Miner:

    @staticmethod
    def gather_voltage(name):
        values = dict()
        for datafile in os.listdir(raw_data_path):
            if name in datafile:
                logging.warning('Gathering data from {}'.format(datafile))
                with open(os.path.join(raw_data_path, datafile)) as data:
                    for line in data:
                        gid, time, value = line.split()
                        if float(time) not in values.keys():
                            values[float(time)] = 0.
                        values[float(time)] += float(value)
        for time in values.keys():
            values[float(time)] /= 20.
        return values

    @staticmethod
    def gather_spikes(name):
        values = dict()
        for datafile in os.listdir(raw_data_path):
            if name in datafile:
                logging.warning('Gathering data from {}'.format(datafile))
                with open(os.path.join(raw_data_path, datafile)) as data:
                    for line in data:
                        gid, time = line.split()
                        if int(gid) not in values.keys():
                            values[int(gid)] = []
                        values[int(gid)].append(float(time))
        return values
