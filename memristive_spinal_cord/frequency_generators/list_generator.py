import logging
from memristive_spinal_cord.frequency_generators.list import FrequencyList


class FrequencyListGenerator:
    """Represents a single rule (formula) for generating frequencies over some time period"""
    def __init__(self):
        self.frequencies = [50]
        self.logger = logging.getLogger('FrequencyListGenerator')

    def generate(self, time, interval):
        """
        Generates a list of frequencies.

        Args:
            time (int): In seconds. How long should we generator for. Example: 60 => the generated list covers 1 minute time period.
            interval (int): In milliseconds. Interval between times when frequencies have to be updated. Example: 10ms means that every 10ms
            we need to add a new frequency value to the generated list.

        Returns:
            FrequencyList: an instance of the FrequencyList which contain a list of frequencies

        """

        self.logger.info('Generation started')
        number_of_intervals = time * 1000 // interval
        self.logger.debug('Total intervals: ' + str(number_of_intervals))
        frequency_list = []
        for i in range(number_of_intervals):
            frequency_list.append(self.frequencies[i % len(self.frequencies)])
        self.logger.info('Generation finished')
        self.logger.debug('FrequencyList: ' + str(frequency_list))
        return FrequencyList(
            interval=interval,
            list=frequency_list,
        )
