from memristive_spinal_cord.afferents.frequency.list import FrequencyList


class FrequencyListGenerator:
    """Represents a single rule (formula) for generating frequencies over some time period"""
    def __init__(self):
        self.frequencies = [50]

    def generate(self, time, interval):
        """
        Generates a list of frequencies.
        :param time: In seconds. How long should we generator for. Example: 60 => the generated list covers 1 minute time
        period.
        :param interval: In milliseconds. Interval between times when frequencies have to be updated. Example: 10ms means that every 10ms
        we need to add a new frequency value to the generated list.
        :return: instance of FrequencyList
        :rtype: FrequencyList
        """
        number_of_intervals = time * 1000 // interval
        frequency_list = []
        for i in range(number_of_intervals):
            frequency_list.append(self.frequencies[i % len(self.frequencies)])
        print('Frequency list: ' + str(frequency_list))
        return FrequencyList(
            interval=interval,
            list=frequency_list,
        )
