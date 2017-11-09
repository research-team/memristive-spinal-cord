import math
import sys
import os
sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-3]))
from memristive_spinal_cord.afferents.frequency.list_generator import FrequencyListGenerator


class TestGenerator(FrequencyListGenerator):

    def __init__(self):
        super().__init__()
        # this frequencies is used one by one while the frequency list is filling
        self.frequencies = [5, 15, 50]


if __name__ == '__main__':
    assert len(sys.argv) == 3, '2 arguments needed simulation time and interval'
    testTime = int(sys.argv[1])
    testInterval = int(sys.argv[2])
else:
    testTime = 1
    testInterval = 20

testGenerator = TestGenerator()
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
print('Spikes per interval: ' + str(spikesPerInterval))

# assert that number of spikes corresponds to their frequency
acceptableErrorNumberOfSpikes = 2
for i in range(numberOfIntervals):
    intervalFrequency = frequencyList.list[i] / numberOfIntervals * testTime
    assert math.fabs(intervalFrequency - spikesPerInterval[i]) < acceptableErrorNumberOfSpikes,\
        'number of spikes corresponds to their frequency'


