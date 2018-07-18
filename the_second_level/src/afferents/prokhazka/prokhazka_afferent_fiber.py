from nest import Connect, Create, SetStatus
from random import randint, random


class ProkhazkaAfferentFiber:
	def __init__(self, n: int=60, weight: float=500., stepcycle: float=1000., step: float=5.):
		self.num_afferents = n
		self.cur_time = 0.
		self.fiber = Create('spike_generator', self.num_afferents)
		self.spiketimes = [[] for i in range(n)]
		self.base_freq = 40.
		self.max_freq = 200.
		self.startpoint = .3
		self.endpoint = .5
		self.maxpoint = .45
		self.spikes_per_step = 2
		self.total_steps = int(stepcycle / step)
		self.generate_spiketimes(stepcycle, step)
		spike_weights = [[weight for _ in self.spiketimes[i]] for i in range(self.num_afferents)]
		for i in range(len(self.fiber)):
			SetStatus([self.fiber[i],], {'spike_times': self.spiketimes[i], 'spike_weights': spike_weights[i]})

		self.neurons = Create(
			model='hh_cond_exp_traub',
	        n=n,
	        params={
            't_ref': 2.,
            'V_m': -70.0,
            'E_L': -70.0,
            'g_L': 50.0,
            'tau_syn_ex': .2,
            'tau_syn_in': 1.})

		self.connect(self.neurons)

	def generate_spiketimes(self, stepcycle: float, step: float):
		spike_interval = step / self.spikes_per_step 
		self.base_spiketimes = [spike_interval / 2. + spike_interval * i for i in range(self.spikes_per_step)]
		for i in range(self.total_steps):
			self.generate_noise()
			self.generate_activity()
			self.cur_time += step


	def generate_noise(self):
		freq = self.base_freq + self.base_freq * .5 * random()
		num_spikes = 1000. / freq
		activated_fibers = int(num_spikes / self.spikes_per_step)
		cur_spiketimes = [round(self.cur_time + spiketime, 1) for spiketime in self.base_spiketimes]
		for i in range(activated_fibers):
			self.spiketimes[i].extend(cur_spiketimes)

	def generate_activity(self):
		pass


	def connect(self, post):
		Connect(
			pre=self.fiber,
			post=post,
			syn_spec={'model': 'static_synapse'},
			conn_spec={'rule': 'one_to_one'})

def test_freq():
	from nest import SetKernelStatus, Simulate, ResetKernel, raster_plot
	ResetKernel()
	SetKernelStatus({
	    'total_num_virtual_procs': 2,
	    'print_time': True,
	    'resolution': 0.1})

	paf = ProkhazkaAfferentFiber()
	sd = Create('spike_detector', params={'withgid': True, 'withtime': True})
	print(sd)
	Connect(paf.neurons, sd)
	Simulate(1000.)
	raster_plot.from_device(sd, hist=True)
	raster_plot.show()