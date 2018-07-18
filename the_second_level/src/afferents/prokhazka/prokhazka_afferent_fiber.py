from nest import Create, SetStatus
from random import randint, random
import pylab


class ProkhazkaAfferentFiber:
	def __init__(self, n: int=60, weight: float=300., stepcycle: float=1000., step: float: 5.):
		self.fiber = Create('spike_generator', n)
		self.spiketimes = [[] for i in range(n)]
		self.base_freq = 40.
		self.max_freq = 200.
		self.startpoint = .3
		self.endpoint = .5
		self.maxpoint = .45
		self.spikes_per_fiber = 2
		self.total_steps = int(stepcycle / step)

	def generateSpikeTimes(self, stepcycle, step):
		spike_interval = step / self.spikes_per_step 
		base_spiketimes = [spike_interval / 2. + spike_interval * i for i in range(self.spikes_per_step)]
		for i in range(self.total_steps):	
			self.busy_fibers = [0] * 60
			generateNoise()
			generateActivity()


	def generateNoise(self):
		freq = self.base_freq + self.base_freq * .5 * random()


	def generateActivity(self):
		pass

	def plotFrequencies(self):
		pass
