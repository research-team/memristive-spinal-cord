import random


class FrequencyList:
    """
    List of frequencies at given time interval.

    Attributes:
        interval (int): In milliseconds. Time interval between frequencies.
        list (:obj:`list` of :obj:`int`): list of frequencies.
        name (str, optional): Name of the list.
    """

    def __init__(self, interval, list, name=''):
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
        spike_times = []
        time = 0.0
        for frequency in self.list:
            spikes_at_interval = int(self.interval / 1000 * frequency)
            if spikes_at_interval < 1 and self.interval / 1000 * frequency > random.random():
                spikes_at_interval = spikes_at_interval + 1
            if spikes_at_interval > 0:
                spikes_at_interval = int(spikes_at_interval)
                time_between_spikes = self.interval / spikes_at_interval
                spike_times.extend(
                    [time + time_between_spikes * (n + 1) for n in range(spikes_at_interval)])
            time += self.interval
        print('Spike times: ' + str(spike_times))
        return spike_times
        # raise NotImplementedError("generateSpikes() is not implemented.")
