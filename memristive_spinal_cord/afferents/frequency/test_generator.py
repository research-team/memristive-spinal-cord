import math
import sys
import os
import logging
from logging.config import fileConfig
sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-3]))
from memristive_spinal_cord.afferents.frequency.list_generator import FrequencyListGenerator


class TestGenerator(FrequencyListGenerator):

    def __init__(self):
        super().__init__()
        # this frequencies is used one by one while the frequency list is filling
        fileConfig(fname='../../logging_config.ini', disable_existing_loggers=False)
        self.logger = logging.getLogger('TestGenerator')
        self.frequencies = [50, 11, 17]
        self.logger.debug('Using frequencies: ' + str(self.frequencies))


if __name__ == '__main__':
    fileConfig('../../logging_config.ini', disable_existing_loggers=False)
    main_logger = logging.getLogger('main')
    assert len(sys.argv) == 3, '2 arguments needed simulation time and interval'
    testTime = int(sys.argv[1])
    testInterval = int(sys.argv[2])
    main_logger.info('Parameters (time, interval): ' + str(testTime) + ' ' + str(testInterval))

    testGenerator = TestGenerator()
    main_logger.debug('TestGenerator constructed')
    frequencyList = testGenerator.generate(testTime, testInterval)
    spikeList = frequencyList.generate_spikes()

    numberOfIntervals = testTime * 1000 // testInterval
    assert numberOfIntervals == len(frequencyList), 'frequency list size must be equal to the number of intervals'

    # storage for number of spikes per interval
    spikesPerInterval = [0] * numberOfIntervals

    # collecting number of spikes per interval
    for spikeTime in spikeList:
        spikeIndex = int(spikeTime / testInterval)
        spikesPerInterval[spikeIndex] += 1
    main_logger.info('Spikes per interval: ' + str(spikesPerInterval))

    # assert that number of spikes corresponds to their frequency
    acceptableErrorNumberOfSpikes = 2
    for i in range(numberOfIntervals):
        intervalFrequency = frequencyList.list[i] / numberOfIntervals * testTime
        assert math.fabs(intervalFrequency - spikesPerInterval[i]) < acceptableErrorNumberOfSpikes,\
            'number of spikes corresponds to their frequency'

    # assert that total number of spike corresponds to average frequency
    average_frequency = sum(frequencyList.list) / len(frequencyList)
    main_logger.debug('Expected number of spikes: ' + str(average_frequency * testTime))
    main_logger.debug('Actual number of spikes: ' + str(len(spikeList)))
    assert math.fabs(average_frequency * testTime - len(spikeList)) < acceptableErrorNumberOfSpikes,\
        'too much difference between expected and actual numbers of spikes'
