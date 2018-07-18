from nest import Create, SetStatus
from random import randint, random
import pylab


class ProkhazkaAfferentFiber:
	def __init__(self, n: int=60, weight: float=300., stepcycle: float=1000., step: float=5.):
		self.num_afferents = n
		self.cur_time = 0.
		self.fiber = Create('spike_generator', self.num_afferents)
		self.spiketimes = [[] for i in range(n)]
		self.base_freq = 40.
		self.max_freq = 200.
		self.startpoint = .3
		self.endpoint = .5
		self.maxpoint = .45
		self.spikes_per_fiber = 2
		self.total_steps = int(stepcycle / step)

	def generate_spiketimes(self, stepcycle: float, step: float):
		spike_interval = step / self.spikes_per_step 
		self.base_spiketimes = [spike_interval / 2. + spike_interval * i for i in range(self.spikes_per_step)]
		for i in range(self.total_steps):
			generateNoise()
			generateActivity()


	def generate_noise(self):
		freq = self.base_freq + self.base_freq * .5 * random()
		num_spikes = 1000. / freq
		activated_fibers = int(num_spikes / self.spikes_per_fiber)
		cur_spiketimes = [self.cur_time + spiketime for spiketime in self.base_spiketimes]
		for i in range(activated_fibers):
			self.spiketimes[i].append(cur_spiketimes)

	def generate_activity(self):
		pass


	def connect(self, post):
		Connect(
			pre=self.fiber,
			post=post,
			syn_spec={'model:': 'static_synapse'},
			conn_spec={'rule': 'one_to_one'})
