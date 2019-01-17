#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include "Neuron.cpp"

class Synapse {
public:
	Synapse() = default;

	Synapse(Neuron *pre_neuron, Neuron* post_neuron, float syn_delay, float weight) {
		this->pre_neuron = pre_neuron;
		this->post_neuron = post_neuron;
		this->syn_delay = ms_to_step(syn_delay);
		this->weight = weight;
	};

	Neuron* pre_neuron{};		// [pointer] pre neuron
	Neuron* post_neuron{};		// [pointer] post neuron

	int syn_delay{};			// [step] synaptic delay. Converts from ms to steps
	int syn_delay_timer = -1;	// [step] timer of synaptic delay
	float weight{};				// [pA] synaptic weight

	__device__
	void update(int sim_iter, int thread_id) {
		// FixMe (bursting?)
		// if neuron has spike and synapse is 'free' for sending spike
		if (pre_neuron->has_spike and syn_delay_timer == -1) {
			syn_delay_timer = syn_delay;
		}
		// send spike event after synaptic delay
		if (syn_delay_timer == 0) {
			post_neuron->spike_event(weight);
			// set synapse state as 'free' for spiking (no synaptic delay timer)
			syn_delay_timer = -1;
		}
		// decrement timer
		if (syn_delay_timer > 0) {
			syn_delay_timer--;
		}

		// FixMe: remove in production
		syn_delay += 10;

		#ifdef DEBUG
			printf("S: %d, T: %d, SYN %p \n", sim_iter, thread_id, this);
		#endif
	}
};