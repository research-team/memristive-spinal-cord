import random
import logging
import re


class FrequencyList:
    """
    List of frequencies at given time interval.

    Attributes:
        interval (int): In milliseconds. Time interval between frequencies.
        list (:obj:`list` of :obj:`int`): list of frequencies.
        name (str, optional): Name of the list.

    """

    def __init__(self, interval, list, name=''):
        self.logger = logging.getLogger('FrequencyList')
        self.interval = interval
        self.list = list
        self.name = name

    def __len__(self):
        return len(self.list)

    def generate_spikes(self):
        """
        Generates a list of spikes by using its own frequency list.

        Returns:
            list: the list of spike times

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
    Receives data from a specific file.
    The filename has to contain a word 'interval' and an integer number after it where
    the number is a value of the interval in ms.
    If several intervals mentioned then the first one will be used

    If there are several lines in the file then the first one will be read

    Attributes:
        filename (str): The name of the file with frequency data
        name (str, optional): Name of the list.

    """

    def __init__(self, filename, name=''):

        f = open(filename, mode='r')
        interval = int(re.search('interval(?P<interval>[\d]+)', filename, flags=re.IGNORECASE).group('interval'))
        frequency_list = [float(value) for value in f.readline().split()]
        f.close()
        super().__init__(interval=interval, list=frequency_list, name=name)
