from memristive_spinal_cord.afferents.frequency.list_generator import FrequencyListGenerator
from math import *

class TestGenerator(FrequencyListGenerator):
    def generate(self, time, interval):
        raise NotImplementedError("generate() is not implemented.")


testTime = 1
testInterval = 100
testGenerator = TestGenerator()
frequencyList = testGenerator.generate(testTime, testInterval)
spikeList = frequencyList.generateSpikes()

numberOfIntervals = testTime * 1000 / testInterval
assert numberOfIntervals == len(frequencyList), 'frequency list size must be equal to the number of intervals'

# storage for number of spikes per interval
spikesPerInterval = []
for i in xrange(numberOfIntervals):
    spikesPerInterval[i] = 0

# collecting number of spikes per interval
for spikeTime in xrange(spikeList):
    spikeIndex = spikeTime / testInterval
    spikesPerInterval[spikeIndex] += 1

# assert that number of spikes corresponds to their frequency
acceptableErrorNumberOfSpikes = 2
for i in xrange(numberOfIntervals):
    intervalFrequency = frequencyList[i] / numberOfIntervals
    assert math.fabs(intervalFrequency - spikesPerInterval[i]) < acceptableErrorNumberOfSpikes,\
        'number of spikes corresponds to their frequency'


