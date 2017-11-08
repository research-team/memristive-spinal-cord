class FrequencyListGenerator:
    """Represents a single rule (formula) for generating frequencies over some time period"""

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
        raise NotImplementedError("generate() is not implemented.")
