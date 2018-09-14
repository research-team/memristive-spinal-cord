"""
Script for testing 'Extracellular' recording.
Use only with neuron model that have Extracellular recording realisation.
"""

__author__ = "Alexey Panzer"
__version__ = "1.7"
__tested___ = "14.09.2018 NEST 2.16.0 (with own module) Python 3"

import nest
import pylab

T_sim = 25.0
spike_times = [2.5, 10.0, 17.0]
spike_weights = [400.0, 300.0, 250.0]

nest.ResetKernel()
nest.SetKernelStatus({
		'total_num_virtual_procs': 2,
		'print_time': False,
		'resolution': 0.1})

nrn_params = {
	't_ref': 2.0,       # [ms] refractory period
	'V_m': -70.,        # [mV] membrane potential
	'E_L': -70.,        # [mV] leak reversal potential
	'E_K': -77.,        # [mV] potassium reversal potential
	'g_L': 30.,         # [nS] leak conductance
	'g_Na': 12000.,     # [nS] sodium peak conductance
	'g_K': 3600.,       # [nS] potassium peak conductance
	'C_m': 134.,        # [pf] capacity of membrane
	'tau_syn_ex': 0.2,  # [ms] time of excitatory action
	'tau_syn_in': 2.0,  # [ms] time of inhibitory action
}

multimeter_params = {
	'record_from': ['Extracellular', 'V_m', "Act_h", "Act_m", "Inact_n"],
	'withgid': True,
	'withtime': True,
	'interval': 0.1,
	'to_file': False,
	'to_memory': True
}

neuron = nest.Create("hh_cond_exp_traub", 1, nrn_params)
multimeter = nest.Create("multimeter", 1, multimeter_params)
generator = nest.Create("spike_generator", 1, {'spike_weights': spike_weights, 'spike_times': spike_times})

nest.Connect(generator, neuron, syn_spec={'weight': 1.0, 'delay': 0.1})
nest.Connect(multimeter, neuron)

nest.Simulate(T_sim)

voltages_extracellular = nest.GetStatus(multimeter, "events")[0]['Extracellular']
voltages_intracellular = nest.GetStatus(multimeter, "events")[0]['V_m']
channel_leakage = nest.GetStatus(multimeter, "events")[0]['Act_h']
channel_K = nest.GetStatus(multimeter, "events")[0]['Inact_n']
channel_Na = nest.GetStatus(multimeter, "events")[0]['Act_m']

times_extracellular = [i / 10 for i in range(len(voltages_extracellular))]
times_intracellular = [i / 10 for i in range(len(voltages_intracellular))]

pylab.figure(figsize=(16, 9))
pylab.subplot(311)
for st in spike_times:
	pylab.axvline(x=st, color='gray', linestyle="--")
pylab.xlim(0, T_sim)
pylab.ylabel("uV")
pylab.plot(times_extracellular, voltages_extracellular, label="Extracellular")
pylab.legend()

pylab.subplot(312)
for st in spike_times:
	pylab.axvline(x=st, color='gray', linestyle="--")
pylab.xlim(0, T_sim)
pylab.ylabel("mV")
pylab.plot(times_intracellular, voltages_intracellular, label="Intracellular")
pylab.legend()

pylab.subplot(313)
for st in spike_times:
	pylab.axvline(x=st, color='gray', linestyle="--")
pylab.xlim(0, T_sim)
pylab.ylabel("Probability")
pylab.xlabel("Time (ms)")
pylab.plot(times_intracellular, channel_Na, label="Na+")
pylab.plot(times_intracellular, channel_K, label="K+")
pylab.plot(times_intracellular, channel_leakage, label="Na+/K+ pump")
pylab.legend()

pylab.show()
