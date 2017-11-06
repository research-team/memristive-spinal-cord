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

    def generateSpikes(self):
        """
        Generates a list of spikes by given frequencies.
        :return: the list of spike times
        """
        raise NotImplementedError("generateSpikes() is not implemented.")
