from nest import Create, Connect, SetStatus, ResetKernel, SetKernelStatus
from random import random, randint


class ProkhazkaAfferents:


    def __init__(self, n: int=60, weight: float=500., end_time: float=1000., stepcycle: float=1000., step: float=10.):

        self.n = n
        self.weight = weight
        self.stepcycle = stepcycle
        self.step = step
        self.end_time = end_time
        self.spikes_per_step = 2

        self.generators = [Create('spike_generator', 1) for _ in range(self.n)]

        for generator in self.generators:
            spike_times = sorted(self.generate_spike_times())
            spike_weights = [weight for _ in spike_times]
            SetStatus(generator, {'spike_times': spike_times, 'spike_weights': spike_weights})

    def generate_spike_times(self):

        spike_times = list()
        cur_time = 0.1
        while cur_time < self.end_time:
            for _ in range(self.spikes_per_step):
                spike_times.append(round(cur_time + (self.step - .1) * random(), 1))
            cur_time += self.step
        return spike_times


def test_afferents():
    
    pa = ProkhazkaAfferents()
    print('Ok!')
