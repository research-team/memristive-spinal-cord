from FrequencyListGenerator import FrequencyListGenerator
from FrequencyList import FrequencyList
import math


class TestGenerator(FrequencyListGenerator):
    def generate(self, time, interval):
        numberOfIntervals = time * 1000 // testInterval
        frequencyList = []
        for i in range(numberOfIntervals):
            frequencyList.append(frequencies[i % len(frequencies)])
        print('Frequency list: ' + str(frequencyList))
        return FrequencyList(
            interval=interval,
            list=frequencyList,
        )
        # raise NotImplementedError("generate() is not implemented.")


frequencies = [5, 15, 50]
testTime = 10
testInterval = 100
testGenerator = TestGenerator()
frequencyList = testGenerator.generate(testTime, testInterval)
spikeList = frequencyList.generate_spikes()

numberOfIntervals = testTime * 1000 // testInterval
assert numberOfIntervals == len(frequencyList), 'frequency list size must be equal to the number of intervals'

# storage for number of spikes per interval
spikesPerInterval = [0] * numberOfIntervals

# collecting number of spikes per interval
for spikeTime in spikeList:
    spikeIndex = int(spikeTime / testInterval) if not spikeTime % testInterval == 0 \
        else int(spikeTime / testInterval) - 1
    spikesPerInterval[spikeIndex] += 1
print('Spikes per interval: ' + str(spikesPerInterval))

# assert that number of spikes corresponds to their frequency
acceptableErrorNumberOfSpikes = 2
for i in range(numberOfIntervals):
    intervalFrequency = frequencyList.list[i] / numberOfIntervals * testTime
    assert math.fabs(intervalFrequency - spikesPerInterval[i]) < acceptableErrorNumberOfSpikes,\
        'number of spikes corresponds to their frequency: ' \
        + str(math.fabs(intervalFrequency - spikesPerInterval[i])) \
        + ' ' + str(numberOfIntervals) \
        + ' ' + str(intervalFrequency) \
        + ' ' + str(spikesPerInterval[i])


