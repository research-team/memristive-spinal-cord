"""
Script for creating and testing four STDP learning rules:
1. Asymetric Hebbian
2. Asymetric anti-Hebbian
3. Sombrero (Symmetric Hebbian)
4. Anti-Sombrero (Symmetric anti-Hebbian)

Based on the article:
Yi Li, Yingpeng Zhong, Jinjian Zhang, et al.
Activity-Dependent Synaptic Plasticity of a Chalcogenide Electronic Synapse for Neuromorphic Systems
2014
URL: www.nature.com/articles/srep04906
DOI: 10.1038/srep04906
"""

import os
import nest
import numpy as np
import pylab as plt

plot_per_delta = False
multimeters = {}
spike_detectors = {}
weight_recorders = {}

original_syn_weight = 50.0
pre_spike_times = [float(100) for _ in range(101)]
post_spike_times = []

offset = 50
for index, pre_spike in enumerate(pre_spike_times):
	post_spike_times.append(pre_spike + offset)
	offset -= 1

# params
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

static_synapse_spec = {
	'weight': 1.0,
	'delay': 0.1
}

wr_params = {
	"to_file": False,
	"to_memory": True
}

detector_params = {
	'withgid': True,
	'to_file': False,
	'to_memory': True
}

multimeter_params = {
	'record_from': ['V_m'],
	'withgid': True,
	'withtime': True,
	'interval': 0.1,
	'to_file': False,
	'to_memory': True
}

common_stdp_parameters = {
	'delay': 0.1,      # synaptic delay
	'Wmax': original_syn_weight * 10,   # maximal allowed weight
	'weight': original_syn_weight,      # synaptic strength
	'mu_plus': 0.0,    # Weight dependence exponent, potentiation
	'mu_minus': 0.0    # Weight dependence exponent, depression
}

hebbian_stdp = {
	'alpha': 1.0,   # Asymmetry parameter (scales depressing increments as  alpha*lambda)
	'lambda': 0.1   # Step size
}

anti_hebbian_stdp = {
	'alpha': 1.0,
	'lambda': -0.1
}

sombrero_stdp = {
	'alpha': -1.0,
	'lambda': 0.1
}

anti_sombrero_stdp = {
	'alpha': -1.0,
	'lambda': -0.1
}

dict_of_models = {
	"Hebbian": dict(hebbian_stdp, **common_stdp_parameters),
	"anti-Hebbian": dict(anti_hebbian_stdp, **common_stdp_parameters),
	"Sombrero": dict(sombrero_stdp, **common_stdp_parameters),
	"anti-Sombrero": dict(anti_sombrero_stdp, **common_stdp_parameters)
}


def reser_kernel():
	nest.ResetKernel()
	nest.SetKernelStatus({
		'total_num_virtual_procs': 4,
		'print_time': True,
		'resolution': 0.1,
		'overwrite_files': True})

def build_topology(syn_specification):
	# create neurons
	neurons_pre = nest.Create("hh_cond_exp_traub", 101, nrn_params)
	neurons_post = nest.Create("hh_cond_exp_traub", 101, nrn_params)
	# create weight recorder
	wr = nest.Create('weight_recorder', n=1, params=wr_params)
	syn_specification['weight_recorder'] = wr[0]
	nest.CopyModel("stdp_synapse", "stdp_synapse_rec", params=syn_specification)
	# pair by pair
	for nrn_pre, nrn_post, pre_spike_time, post_spike_time in zip(neurons_pre, neurons_post,
	                                                              pre_spike_times, post_spike_times):
		# create generators
		# the spikes at the end of the simulation is need to re-calculate weights of synapses
		generator_pre = nest.Create('spike_generator', n=1, params={'spike_times': [pre_spike_time, 225.0],
		                                                            'spike_weights': [350.0, 350.0]})
		generator_post = nest.Create('spike_generator', n=1, params={'spike_times': [post_spike_time, 225.0],
		                                                             'spike_weights': [350.0, 350.0]})
		nest.Connect(generator_pre, [nrn_pre], syn_spec=static_synapse_spec)
		nest.Connect(generator_post, [nrn_post], syn_spec=static_synapse_spec)
		# create detector
		detector = nest.Create('spike_detector', n=1, params=detector_params)
		nest.Connect([nrn_pre, nrn_post], detector)
		spike_detectors[(nrn_pre, nrn_post)] = detector
		# create multimeter
		multimeter = nest.Create('multimeter', n=1, params=multimeter_params)
		nest.Connect(multimeter, [nrn_pre, nrn_post])
		multimeters[(nrn_pre, nrn_post)] = multimeter
		# connect neurons
		nest.Connect([nrn_pre], [nrn_post], syn_spec="stdp_synapse_rec")
	return neurons_pre, neurons_post, wr

def simulate(T):
	nest.Simulate(float(T))

def plot_results(model_name, neurons_pre, neurons_post, wr):
	weights_list = {}
	# get weights
	senders = nest.GetStatus(wr, 'events')[0]['senders']
	targets = nest.GetStatus(wr, 'events')[0]['targets']
	weights = nest.GetStatus(wr, 'events')[0]['weights']
	for s, t, w in sorted(zip(senders, targets, weights)):
		# FixMe: for the anti-models we need to use only first weight values
		if (s, t) not in weights_list.keys():
			weights_list[(s, t)] = w
		# FixMe: for the non anti-models we need to use second weight values
		if model_name in ["Sombrero", "Hebbian"]:
			weights_list[(s, t)] = w
	weight_values = [100 * v / original_syn_weight - 100 for k, v in weights_list.items()]
	weight_times = range(50, -51, -1)

	for nrn_pre, nrn_post, pre_spike_time, post_spike_time in zip(neurons_pre, neurons_post,
	                                                              pre_spike_times, post_spike_times):
		# get voltages
		mm = multimeters[(nrn_pre, nrn_post)]
		events_senders = nest.GetStatus(mm)[0]['events']['senders']
		events_voltages = nest.GetStatus(mm)[0]['events']['V_m']
		events_times = nest.GetStatus(mm)[0]['events']['times']
		# fill the dict with data
		nrn_values = {sender: {"volt": [],
		                       "times": []}
		              for sender in [nrn_pre, nrn_post]}
		for s, t, v in zip(events_senders, events_times, events_voltages):
			nrn_values[s]["times"].append(t)
			nrn_values[s]["volt"].append(v)
		# get spikes
		sd = spike_detectors[(nrn_pre, nrn_post)]
		spike_times = nest.GetStatus(sd)[0]['events']['times']

		print("Pre (ID {:0>3}): {:<5} ms Post (ID {:<3}): {:<5} ms  Δt (ms) {:<4}  ΔW (%) {:<6.3f}".format(
			nrn_pre,
			pre_spike_time,
			nrn_post,
			post_spike_time,
			post_spike_time - pre_spike_time,
			100 * weights_list[(nrn_pre, nrn_post)] / original_syn_weight - 100))

		# plot results
		if plot_per_delta:
			plt.figure()
			plt.suptitle("Δt {} ms. ΔW {:.3f} %".format(
				post_spike_time - pre_spike_time,
				100 * weights_list[(nrn_pre, nrn_post)] / original_syn_weight - 100
			))
			# plot voltages and spikes
			for nrn_id in [nrn_pre, nrn_post]:
				plt.plot(nrn_values[nrn_id]["times"],
				         nrn_values[nrn_id]["volt"],
				         label="PRE" if nrn_id < nrn_post else "POST")
			plt.plot(spike_times, [40 for _ in spike_times], ".", color='r')
			plt.xlim(0, 200)
			plt.ylim(-80, 50)
			plt.legend()
			plt.savefig("{}/{}_{}_{}".format(os.getcwd(), model_name, nrn_pre, nrn_post), dpi=120)
			plt.close('all')

	# plot the learning model
	plt.figure()
	plt.suptitle("Model: {}".format(model_name))
	plt.plot(weight_times, weight_values)
	plt.plot(weight_times, weight_values, ".", color='r', markersize=2)
	plt.axvline(0, linestyle="--", color="gray")
	plt.axhline(0, linestyle="--", color="gray")
	plt.xlabel("Δt (ms)")
	plt.ylabel("ΔW (%)")
	plt.xlim(-50, 50)
	plt.xticks(range(-50, 51, 10))
	plt.ylim(-100, 100)
	plt.savefig("{}/{}".format(os.getcwd(), model_name), dpi=120)
	plt.close('all')

def main():
	T = 230
	for model_name, model_params in dict_of_models.items():
		reser_kernel()
		nrns_pre, nrns_post, wr = build_topology(model_params)
		simulate(T)
		plot_results(model_name, nrns_pre, nrns_post, wr)


if __name__ == "__main__":
	main()
