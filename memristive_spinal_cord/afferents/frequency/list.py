import random
import logging
from logging.config import fileConfig


class FrequencyList:
    """
    List of frequencies at given time interval.

    Attributes:
        interval (int): In milliseconds. Time interval between frequencies.
        list (:obj:`list` of :obj:`int`): list of frequencies.
        name (str, optional): Name of the list.
    """

    def __init__(self, interval, list, name=''):
        fileConfig('../../logging_config.ini', disable_existing_loggers=False)
        self.logger = logging.getLogger('FrequencyList')
        self.interval = interval
        self.list = list
        self.name = name

    def __len__(self):
        return len(self.list)

    def generate_spikes(self):
        """
        Generates a list of spikes by given frequencies.
        :return: the list of spike times
        """
        self.logger.info('Spike generation started')
        self.logger.debug('Using frequency list: ' + str(self.list))
        spike_times = []
        # initial time
        time = 0.0
        for frequency in self.list:
            spikes_at_interval = int(self.interval / 1000 * frequency)

            # fraction used as a probability of the additional spike
            if self.interval / 1000 * frequency - spikes_at_interval > random.random():
                spikes_at_interval += 1

            if spikes_at_interval > 0:
                time_between_spikes = self.interval / spikes_at_interval
                time -= time_between_spikes / 2  # shifting time to place spikes closer to the center
                spike_times.extend(
                    [time + time_between_spikes * (n + 1) for n in range(spikes_at_interval)])
                time += time_between_spikes / 2  # shifting back
            time += self.interval
        self.logger.info('Spike generation finished')
        self.logger.debug('Spike times: ' + str(spike_times))
        self.logger.debug('Total spikes generated: ' + str(len(spike_times)))
        return spike_times


class FrequencyListFile(FrequencyList):
    """
    List of frequencies at given time interval.
    Receives data from a specific file

    Attributes:
        :param interval (int): In milliseconds. Time interval between frequencies.
        :param afferent_index (int): Actually the number of the required line in a file with data, starts with 1
        :param filename (str): The name of the file with frequency data
        :param name (str, optional): Name of the list.
    """

    def __init__(self, interval, afferent_index, filename, name=''):
        fileConfig('../../logging_config.ini', disable_existing_loggers=False)
        self.logger = logging.getLogger('FrequencyListFile')
        f = open('frequency_data/' + filename, mode='r')
        self.logger.info('File opened')
        frequency_list = []
        for index, line in enumerate(f):
            self.logger.info('Getting line ' + str(index))
            if index == afferent_index - 1:
                self.logger.info('Found line ' + str(index))
                frequency_list = [float(value) for value in line.split()]
                break
        f.close()
        self.logger.info('File closed')
        super().__init__(interval=interval, list=frequency_list, name=name)


if __name__ == '__main__':
    flf = FrequencyListFile(20, 1, 'fr_Ia_GM_speed15.txt')
    flf.generate_spikes()
