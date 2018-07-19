from nest import Create, Connect, SetStatus, ResetKernel, SetKernelStatus
from random import random, randint
from math import ceil
import pylab


class ProkhazkaAfferents:


    def __init__(self, n: int=60, weight: float=500., end_time: float=1000., stepcycle: float=1000., step: float=20.):

        self.n = n
        self.weight = weight
        self.stepcycle = stepcycle
        self.step = step
        self.end_time = end_time
        self.spikes_per_step = 2
        self.cur_time = .1

        self.generators = [Create('spike_generator', 1) for _ in range(self.n)]
        self.spike_times = [[] for _ in range(self.n)]

        while self.cur_time < self.end_time:
            for i in range(self.calculate_activated_fibers()):
                spike_times = sorted(self.generate_spike_times(i))
                spike_weights = [weight for _ in spike_times]
                self.spike_times[i].extend(spike_times)
            self.cur_time += step

    def generate_spike_times(self, i: int):

        spike_times = list()
        
        for _ in range(self.spikes_per_step):
            spike_times.append(round(self.cur_time + (self.step - .1) * random(), 1))
        return spike_times

    def calculate_activated_fibers(self):
        x = 0
        if self.cur_time < 250 or self.cur_time > 600:
            rate = 50 + 4 * random()
            x = 1
        elif 250 < self.cur_time < 500:
            rate = 0.75 * self.cur_time - 125.
            x = 2
        else:
            rate = -1.5 * self.cur_time + 950 
            x = 3
        spikes = rate * self.step / 1000. * 15
        activated_fibers = min(int(spikes / self.spikes_per_step), 60) 
        print('Activated fibers on time {} for rate {}: {} where x = {}'.format(self.cur_time, rate, activated_fibers, x))
        return activated_fibers


def test_afferents(interval: float=10., time: float=1000.):
    
    pa = ProkhazkaAfferents()
    print('Ok!')

    rates = dict()
    a = .0
    b = interval

    while a < time:
        num_spikes = 0
        for generator in pa.spike_times:
            for spike_time in generator:
                if a < spike_time < b:
                    num_spikes += 1
        rates[(a + b) / 2] = round(num_spikes / interval * 1000., 1)
        a += interval
        b += interval

    xvalues = sorted(rates.keys())
    yvalues = [rates[x] for x in xvalues]
    print(yvalues)
    pylab.plot(xvalues, yvalues)
    pylab.show()